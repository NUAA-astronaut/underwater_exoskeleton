"""Independent assistance checks; all generated files stay in assistance_output."""
import argparse
import ast
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import numpy as np
import mujoco
import simulate as legacy
import validate as legacy_validation
from assistance_sim import run_experiment, safe_output, MODES
from assistance_analysis import load_csv, rms
from assistance_controllers import G1,G2,MotorResponse,SensorObservation

ROOT=legacy.ROOT


def fingerprints():
    # 收集根目录下需要重点保护的核心文件。
    paths=[ROOT/n for n in ['simulate.py','validate.py','config.json','README.md']]
    # 递归收集 output、validation 目录下的所有文件(仅文件,排除目录)。
    for name in ['output','validation']:
        paths.extend(p for p in (ROOT/name).rglob('*') if p.is_file())
    # 返回 相对路径 -> SHA-256 哈希 的映射,用于前后比对,防止遗留文件/输出被篡改。
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def motor_checks(ac):
    # 分别测试电机响应延迟关闭(lag=0)和开启(lag=0.03s)两种工况。
    maxima=[]
    for lag in [0,.03]:
        # 基于配置构造电机模型,记录上一步力矩,并初始化饱和/速率限制触发标志。
        cfg=dict(ac['motor'],response_time_s=lag); m=MotorResponse(cfg); previous=np.zeros(2); saturated=False; limited=False
        for k in range(1000):
            # 前 500 步请求 [+100,-100] Nm,后 500 步反向,以同时触发正负向限制。
            requested=np.array([100.,-100.])*(1 if k<500 else -1)
            actual=m.step(requested,.001)
            # 断言 1:实际力矩幅值不得超过扭矩上限。
            assert np.max(abs(actual))<=cfg['torque_limit_Nm']+1e-12
            # 断言 2:单步力矩变化量不得超过 slew 限速 × 步长。
            assert np.max(abs(actual-previous))<=cfg['slew_limit_Nm_s']*.001+1e-12
            previous=actual.copy(); saturated |= bool(m.saturated.any()); limited |= bool(m.slew_limited.any())
        # 必须实际触发过饱和与速率限制,否则断言失败(说明测试信号不够强)。
        assert saturated and limited
        maxima.append(float(np.max(abs(actual))))
    return {'amplitude_and_slew_limits_passed':True,'lag_disabled_and_enabled_passed':True,'final_absolute_torque_Nm':maxima}


def direction_checks(c):
    # 构建 MuJoCo 模型,并关闭重力与摩擦损失,以便孤立地验证力矩方向与传递。
    m=mujoco.MjModel.from_xml_string(legacy.build(c)[0]); m.opt.gravity[:]=0; m.dof_frictionloss[:]=0
    # 所有人形关节的自由度地址,用于后续断言电机不直接驱动人体关节。
    h=[m.joint(j).dofadr[0] for j in legacy.JOINTS]
    results={}
    for side in ['L','R']:
        # 取该侧外骨骼髋关节自由度地址与对应电机执行器 id。
        dof=m.joint('exo_'+side+'_hip').dofadr[0]; aid=m.actuator('exo_'+side+'_hip_motor').id
        for sign in [-1,1]:
            # 施加 ±1 单位控制量,前向运动学求加速度与执行器力矩。
            d=mujoco.MjData(m); d.ctrl[aid]=sign; mujoco.mj_forward(m,d)
            # 断言 1:外骨骼髋关节加速度方向与控制量符号一致。
            assert sign*d.qacc[dof]>0
            # 断言 2:执行器不直接对任何人形关节施加力矩(仅作用于外骨骼)。
            assert np.max(abs(d.qfrc_actuator[h]))==0
            # 断言 3:外骨骼髋关节处的执行器力矩恰等于控制量 sign。
            assert np.isclose(d.qfrc_actuator[dof],sign)
            # 先验证有符号响应被孤立,再通过有限刚度绑带将其传递到人体。
            jp=np.zeros((3,m.nv)); jq=np.zeros((3,m.nv)); jr=np.zeros((3,m.nv))
            # 迭代 30 步:每步重新计算绑带力并施加到 qfrc_applied,再积分一步。
            for _ in range(30):
                mujoco.mj_forward(m,d); d.qfrc_applied[:]=0
                for n in ['L_thigh','L_shank','R_thigh','R_shank']:
                    # 取人体侧与外骨骼侧绑带连接点的 site id。
                    si=m.site(n+'_strap').id; sj=m.site('exo_'+n+'_strap').id
                    # 计算两 site 的位置雅可比(转置后用于把笛卡尔力映射到关节力矩)。
                    mujoco.mj_jacSite(m,d,jp,jr,si); mujoco.mj_jacSite(m,d,jq,jr,sj)
                    # 绑带力 = 刚度×位移差 + 阻尼×相对速度。
                    force=c['strap_stiffness_N_m']*(d.site_xpos[sj]-d.site_xpos[si])+c['strap_damping_Ns_m']*((jq-jp)@d.qvel)
                    # 将绑带力通过雅可比转置投影为关节广义力并累加。
                    d.qfrc_applied[:]+=(jp-jq).T@force
                mujoco.mj_step(m,d)
            hd=m.joint(side+'_hip').dofadr[0]
            # 断言 4:经过绑带传递后,外骨骼与人体髋关节位移方向均与 sign 一致。
            assert sign*d.qpos[dof]>0 and sign*d.qpos[hd]>0
            results[f'{side}_{sign:+}']={'exo_displacement_rad':float(d.qpos[dof]),'human_displacement_rad':float(d.qpos[hd]),'direct_human_actuator_force':0.0}
    return results


def controller_isolation(ac, data):
    # 从 CSV 重建仅含可观测量(时间、左右髋 IMU 角度/角速度)的观测序列。
    obs=[SensorObservation(float(t),(float(l),float(r)),(float(lv),float(rv))) for t,l,r,lv,rv in zip(
        data['time_s'],data['L_hip_imu_rad'],data['R_hip_imu_rad'],data['L_hip_imu_rad_s'],data['R_hip_imu_rad_s'])]
    # 离线逐样本驱动 G2,得到仅基于可观测量的力矩指令。
    g=G2(ac); commands=np.array([g.update(o) for o in obs])
    # 与实验中记录的请求力矩对比,证明 G2 可仅凭观测完全复现。
    recorded=np.column_stack([data['L_hip_request_next_Nm'],data['R_hip_request_next_Nm']])
    assert np.max(abs(commands-recorded))<1e-12
    # 离线重放是逐样本喂入的。篡改后半段观测,前半段输出必须保持不变(因果性)。
    k=len(obs)//2; g=G2(ac); changed=[]
    for i,o in enumerate(obs):
        if i>=k: o=SensorObservation(o.time_s,(-.4,.8),(2.,-2.))
        changed.append(g.update(o))
    changed=np.asarray(changed); assert np.array_equal(changed[:k],commands[:k])
    # 后半段输出必须明显改变,证明控制器确实对传感器数据有响应。
    assert np.max(abs(changed[k:]-commands[k:]))>.1
    # G1 是纯时间函数:即便把角度/速度篡改为极端值,输出也应完全一致。
    g1=G1(ac)
    for o in obs[::100]:
        assert np.array_equal(g1.update(o),g1.update(SensorObservation(o.time_s,(999.,-999.),(999.,-999.))))
    # 静态检查 assistance_controllers.py 的导入边界:只允许标准库与 numpy。
    tree=ast.parse((ROOT/'assistance_controllers.py').read_text())
    imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
    imports += [a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names]
    assert set(imports)<=set(['collections','dataclasses','math','numpy'])
    # 禁止出现的标识符:确保控制器不直接访问仿真状态或真实相位(避免作弊)。
    forbidden={'mujoco','qpos','qvel','target','TruthObservation','truth_phase','HIP','KNEE'}
    assert not ({n.id for n in ast.walk(tree) if isinstance(n,ast.Name)} & forbidden)
    return {'observable_only_replay_max_error_Nm':float(np.max(abs(commands-recorded))),
            'future_suffix_independence':True,'responds_to_changed_sensor_data':True,'G1_sensor_independence':True,'controller_import_boundary':imports}


def main():
    # 解析输出目录参数,默认 assistance_output;vout 为验证报告子目录。
    parser=argparse.ArgumentParser(); parser.add_argument('--out',type=Path,default=ROOT/'assistance_output'); args=parser.parse_args()
    out=safe_output(args.out); vout=safe_output(out/'validation'); before=fingerprints()
    # 加载基础配置 c 与助力配置 ac。
    c=json.loads((ROOT/'config.json').read_text()); ac=json.loads((ROOT/'assistance_config.json').read_text())
    # 先执行电机单元测试与力矩方向/绑带传递测试。
    report={'motor':motor_checks(ac),'signed_torque_and_strap_transfer':direction_checks(c)}
    # 读取各模式的 manifest,断言所有模式共享相同的人体/参考轨迹哈希、初始状态与模型哈希。
    manifests=[json.loads((out/m/'manifest.json').read_text()) for m in MODES]
    for key in ['human_and_reference_hash','initial_qpos','initial_qvel','model_sha256']:
        assert all(x[key]==manifests[0][key] for x in manifests)
    # 断言各模式配置与当前配置完全一致,且未启用理想噪声。
    for x in manifests: assert x['base_config']==c and x['assistance_config']==ac and x['ideal_noise'] is False
    report['identical_parameters_reference_and_initial_state']=True
    # 加载各模式 100Hz 传感器数据。
    data={m:load_csv(out/m/'sensors_100hz.csv') for m in MODES}
    for mode,a in data.items():
        # 断言:采样间隔严格为 0.01s(100Hz)。
        assert np.allclose(np.diff(a['time_s']),.01,atol=1e-12,rtol=0)
        # 断言:膝关节外骨骼力矩恒为 0(助力仅作用于髋关节)。
        assert np.max(abs(a['L_knee_exo_torque_Nm']))==np.max(abs(a['R_knee_exo_torque_Nm']))==0
        # 断言:所有模式的目标关节角度轨迹与 G0(无助力)完全相同。
        for j in legacy.JOINTS:
            assert np.array_equal(a[j+'_target_rad'],data['G0'][j+'_target_rad'])
        # 断言:metrics 中无任何警告。
        metrics=json.loads((out/mode/'metrics.json').read_text()); assert sum(metrics['warning_counts'])==0
    report['knees_zero_sampling_100hz_no_warnings']=True
    # 在新目录中运行原始模拟器(不污染 legacy output),将结果与 G0 逐列比对。
    with contextlib.redirect_stdout(io.StringIO()): legacy.run(copy.deepcopy(c),vout/'baseline_replay')
    old=load_csv(vout/'baseline_replay'/'sensors_100hz.csv'); g0=data['G0']
    differences={k:float(np.max(abs(old[k]-g0[k]))) for k in old}
    assert max(differences.values())<1e-10,differences
    report['G0_original_all_columns_max_abs_difference']=max(differences.values())
    # 保留原始 validate.py 不变,仅通过模块变量重定向其输出目录后执行其 main()。
    original_root=legacy_validation.ROOT; legacy_validation.ROOT=vout/'original_validation'
    legacy_validation.ROOT.mkdir(parents=True,exist_ok=True)
    (legacy_validation.ROOT/'config.json').write_text(json.dumps(c))
    try:
        with contextlib.redirect_stdout(io.StringIO()): legacy_validation.main()
    finally: legacy_validation.ROOT=original_root
    report['existing_validation_passed']=True
    # 验证 G2 控制器的因果性与隔离性。
    report['G2_causality_and_isolation']=controller_isolation(ac,data['G2'])
    # 验证双侧交替(反相):稳态段(t>=6s)左右髋指令应反相。
    anti={}
    for mode,a in data.items():
        steady=a['time_s']>=6
        if mode=='G1':
            # G1 为开环:左右请求力矩之和应严格为 0。
            error=float(np.max(abs(a['L_hip_request_next_Nm']+a['R_hip_request_next_Nm'])))
            assert error<1e-10
            anti[mode]={'exact_antiphase_request_max_sum_Nm':error}
        elif mode in ['G2','Oracle']:
            # G2 用 imu 相位,Oracle 用真实相位;相位差应接近 π。
            suffix='imu' if mode=='G2' else 'true'
            delta=legacy.wrap(a[f'R_phase_{suffix}_rad'][steady]-a[f'L_phase_{suffix}_rad'][steady]-np.pi)
            err=rms(np.rad2deg(delta)); assert err<45
            # 左右力矩指令相关系数应强负相关(反相)。
            corr=float(np.corrcoef(a['L_hip_request_next_Nm'][steady],a['R_hip_request_next_Nm'][steady])[0,1]); assert corr<-.8
            anti[mode]={'antiphase_rmse_deg':err,'left_right_command_correlation':corr}
    report['bilateral_alternation']=anti
    # 步长减半收敛性测试:四种模式、相同种子噪声、对齐的 100Hz 采样、完整 24s。
    convergence={}
    for mode in MODES:
        half=copy.deepcopy(c); half['timestep_s']/=2
        hm=run_experiment(half,copy.deepcopy(ac),mode,vout/'half_dt'/mode,plots=False)
        b=load_csv(vout/'half_dt'/mode/'sensors_100hz.csv'); a=data[mode]; steady=a['time_s']>=6
        # 时间轴必须完全一致,且减半步长后无警告。
        assert np.array_equal(a['time_s'],b['time_s']) and sum(hm['warning_counts'])==0
        # 稳态段各关节真实角度 RMS 误差应 < 0.25 度。
        angle_errors={j:rms(np.rad2deg(a[j+'_true_rad'][steady]-b[j+'_true_rad'][steady])) for j in legacy.JOINTS}
        assert max(angle_errors.values())<.25
        # 关键代理指标(四关节 RMS 力矩、人体正功)相对变化应 < 2%。
        original=json.loads((out/mode/'metrics.json').read_text())
        rel={key:abs(hm['aggregate'][key]/original['aggregate'][key]-1) for key in ['human_four_joint_rms_Nm','human_positive_work_J']}
        assert max(rel.values())<.02
        convergence[mode]={'joint_rmse_deg':angle_errors,'proxy_relative_changes':rel}
    report['step_halving']=convergence
    # 最终校验:所有遗留文件与输出的哈希必须与开始时完全一致。
    assert fingerprints()==before
    report['legacy_files_and_outputs_unchanged']=True
    report['passed']=True
    (vout/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()