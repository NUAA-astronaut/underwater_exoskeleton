"""Physics, sensing, baseline reduction and reconstruction sensitivity checks."""
import json, copy, hashlib
import numpy as np
import mujoco
from assistance_sim import ROOT, IMUPipeline
from assistance_analysis import load_csv, rms
from assistance_controllers import G2, SensorObservation
from pelvis_support import build_moving, motion, PrescribedBase, PaperTarget
from pelvis_runner import run_pelvis

def main():
    # ---------- 加载配置与准备输出目录 ----------
    c=json.loads((ROOT/'config.json').read_text());ac=json.loads((ROOT/'assistance_config.json').read_text())
    out=ROOT/'pelvis_output'/'validation';out.mkdir(parents=True,exist_ok=True)
    # 记录现有产出文件的哈希, 用于末尾确认验证过程未改动既有结果
    protected=[p for folder in ['output','validation','assistance_output'] for p in (ROOT/folder).rglob('*') if p.is_file()]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    zero={'amplitude_deg':0,'frequency_hz':.67,'phase_deg':0}
    report={}

    # ---------- 检查 1: 零扰动基线与旧版结果一致 ----------
    # 零骨盆俯仰下用 legacy 参考重跑, 应与既有 output/sensors_100hz.csv 逐关节吻合
    run_pelvis(c,ac,'G0',out/'zero_legacy',zero,'legacy',plots=False)
    old=load_csv(ROOT/'output'/'sensors_100hz.csv');new=load_csv(out/'zero_legacy'/'sensors_100hz.csv')
    report['zero_pitch_legacy_max_joint_difference_rad']=max(float(np.max(abs(old[j+'_true_rad']-new[j+'_true_rad']))) for j in ['L_hip','L_knee','R_hip','R_knee'])
    assert report['zero_pitch_legacy_max_joint_difference_rad']<1e-5

    # ---------- 检查 2: 五 IMU 解析加速度/角速度一致性 ----------
    # Entire articulated system rotates, with relative joints zero: analytic IMU
    # acceleration includes tangential + centripetal terms at each installation.
    # 整机随基座转动而相对关节为零, 用解析式核对各安装点的切向+向心加速度项
    m=mujoco.MjModel.from_xml_string(build_moving(c)[0]);d=mujoco.MjData(m)
    m.opt.gravity[:]=[0,0,-9.81];max_acc=0.;max_gyro=0.;max_relative=0.
    from simulate import IMUS,rotation_y
    # 关闭噪声, 保证解析对比是确定性验证
    cfg=dict(c,gyro_noise_std_deg_s=0,gyro_bias_std_deg_s=0,accel_noise_std_m_s2=0)
    pipeline=IMUPipeline(cfg,True)
    for step in range(801):
        # 通过逆动力学注入已知运动(q/v/a), 让 MuJoCo 生成传感器读数
        t=step*.01;q,v,a=motion(t,{'amplitude_deg':5,'frequency_hz':1.34,'phase_deg':90})
        d.qpos[:]=0;d.qvel[:]=0;d.qacc[:]=0;d.qpos[0]=q;d.qvel[0]=v;d.qacc[0]=a
        mujoco.mj_inverse(m,d)
        acc=[];gyro=[]
        for n in IMUS:
            sid=m.site(n+'_imu').id;r=d.site_xpos[sid]-d.xpos[m.body('pelvis').id]
            # 期望值: 切向项 a×r + 向心项 ω×(ω×r) - 重力(比力约定)
            expected=np.cross([0,a,0],r)+np.cross([0,v,0],np.cross([0,v,0],r))-m.opt.gravity
            R=d.site_xmat[sid].reshape(3,3)
            measured=d.sensor(n+'_acc').data.copy();g=d.sensor(n+'_gyro').data.copy()
            # 测量值需旋转到传感器坐标系后再与期望比较
            max_acc=max(max_acc,float(np.max(abs(measured-R.T@expected))));max_gyro=max(max_gyro,float(np.max(abs(g-R.T@np.array([0,v,0])))))
            acc.append(measured);gyro.append(g)
        # 用估计管线复核: 纯基座运动时相对角应估计为 0(跳过前 6 s 收敛期)
        _,_,estimated,_=pipeline.update(t,np.array(acc),np.array(gyro))
        if t>=6:max_relative=max(max_relative,float(np.max(abs(np.rad2deg(estimated)))))
    assert max_acc<1e-7 and max_gyro<1e-10
    report['analytic_five_imu_acceleration_max_error']=max_acc;report['analytic_five_imu_gyro_max_error']=max_gyro
    report['base_only_relative_angle_max_estimation_error_deg']=max_relative

    # ---------- 检查 3: 时间步收敛性(半步长) ----------
    # Under dynamic acceleration complementary filter need not cancel base error.
    # 动态加速度下互补滤波无法完全抵消基座误差, 故用半步长对比检查数值收敛
    case={'amplitude_deg':5,'frequency_hz':1.34,'phase_deg':180};short=dict(c,duration_s=12)
    a=run_pelvis(short,ac,'G2',out/'dt',case,plots=False)
    b=run_pelvis(dict(short,timestep_s=.0005),ac,'G2',out/'half_dt',case,plots=False)
    x=load_csv(out/'dt'/'sensors_100hz.csv');y=load_csv(out/'half_dt'/'sensors_100hz.csv');mask=x['time_s']>=6
    # 半时间步与标准步长的四关节真值 RMSE 应很小(收敛)
    err={j:rms(np.rad2deg(x[j+'_true_rad'][mask]-y[j+'_true_rad'][mask])) for j in ['L_hip','L_knee','R_hip','R_knee']}
    assert max(err.values())<.25
    report['half_step_joint_rmse_deg']=err

    # ---------- 检查 4: 无警告与反力加速度残差 ----------
    assert sum(a['warning_counts'])==sum(b['warning_counts'])==0
    assert a['base']['max_acceleration_residual']<1e-5
    report['reaction_acceleration_max_residual']=a['base']['max_acceleration_residual']

    # ---------- 检查 5: G2 控制器可仅由传感器输入确定性复现 ----------
    # 重放记录的单侧髋角度/角速度, 输出力矩应与 CSV 中请求值完全一致
    ctrl=G2(ac);replay=[]
    for i,t in enumerate(x['time_s']):replay.append(ctrl.update(SensorObservation(float(t),(x['L_hip_imu_rad'][i],x['R_hip_imu_rad'][i]),(x['L_hip_imu_rad_s'][i],x['R_hip_imu_rad_s'][i]))))
    re=np.max(abs(np.asarray(replay)-np.column_stack([x['L_hip_request_next_Nm'],x['R_hip_request_next_Nm']])))
    assert re<1e-12;report['G2_sensor_only_replay_max_error']=float(re)

    # ---------- 检查 6: 参考轨迹重建敏感性 ----------
    # 换用 alternate 重建参考, 对比聚合指标以评估对重建方式的敏感度
    alt=run_pelvis(short,ac,'G2',out/'alternate_reconstruction',case,'alternate',plots=False)
    report['reconstruction_sensitivity']={'primary':a['aggregate'],'alternate':alt['aggregate']}

    # ---------- 收尾: 确认既有产出未被改动并写出报告 ----------
    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p)] for p in protected)
    report['old_outputs_unchanged']=True;report['passed']=True
    (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()