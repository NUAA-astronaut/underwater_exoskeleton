"""Analysis only. Never imported by the online controller module."""
import csv
import json
import numpy as np
from simulate import JOINTS, wrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def columns(rows):
    return {k:np.asarray([r[k] for r in rows],dtype=float) for k in rows[0]}


def load_csv(path):
    with path.open() as f: return columns(list(csv.DictReader(f)))


def rms(x): return float(np.sqrt(np.mean(np.asarray(x)**2)))


def summarize(rows, physics, c, ac):
    # s:100Hz 传感器数据列;p:全步频物理数据列。
    s=columns(rows); p=columns(physics)
    # 稳态评估起始时刻;mask 用于物理步频数据,sm 用于传感器步频数据。
    start=ac['evaluation']['steady_start_s']; mask=p['time_s']>=start; sm=s['time_s']>=start
    if mask.sum()<2: raise ValueError('Not enough steady-state data')
    dt=c['timestep_s']; duration=mask.sum()*dt
    # result 字典:记录评估窗口、积分方式说明,以及 joints/phase 两个子字典。
    result={'evaluation_start_s':start,'evaluation_duration_s':duration,'integration':'physics-rate left endpoint; positive work per joint before summation',
            'proxy_warning':'Net ideal joint-actuator burden, NOT metabolic cost or underwater propulsion efficiency', 'joints':{},'phase':{}}
    # 汇总用临时列表:正功、力矩、髋部正功、外骨骼净功、跟踪误差。
    work=[]; torque=[]; hip_work=[]; exo_work=[]; errors=[]
    for j in JOINTS:
        # 取该关节稳态段的人体力矩、功率、外骨骼功率及跟踪误差。
        tq=p[j+'_human_torque_Nm'][mask]; power=p[j+'_human_power_W'][mask]; ep=p[j+'_exo_power_W'][mask]
        err=np.rad2deg(p[j+'_true_rad'][mask]-p[j+'_target_rad'][mask])
        # 正功:仅对功率为正的部分积分(左端点 Riemann 和)。
        positive=float(np.maximum(power,0).sum()*dt)
        work.append(positive); torque.append(tq); errors.append(err); exo_work.append(float(ep.sum()*dt))
        if 'hip' in j: hip_work.append(positive)
        # 单关节指标:人体力矩 RMS、正/净功,外骨骼正/负/净功,跟踪 RMSE/最大误差,
        # 人体-外骨骼角度偏差,IMU 估计误差,人体力矩饱和占比,外骨骼最大实际力矩,
        # 以及人体力矩各分量(前馈/P/D/反馈/动力学/限幅)的 RMS。
        result['joints'][j]={'human_active_rms_Nm':rms(tq),'human_positive_work_J':positive,'human_net_work_J':float(power.sum()*dt),
            'exo_net_work_J':float(ep.sum()*dt),'exo_positive_work_J':float(np.maximum(ep,0).sum()*dt),'exo_negative_work_J':float(np.minimum(ep,0).sum()*dt),
            'tracking_rmse_deg':rms(err),'tracking_max_abs_deg':float(np.max(abs(err))),
            'human_exo_rmse_deg':rms(np.rad2deg(p[j+'_true_rad'][mask]-p[j+'_encoder_rad'][mask])),
            'imu_rmse_deg':rms(np.rad2deg(wrap(s[j+'_imu_rad'][sm]-s[j+'_true_rad'][sm]))),
            'human_saturation_fraction':float(np.mean(np.abs(p[j+'_human_clipping_Nm'][mask])>1e-12)),
            'max_exo_actual_Nm':float(np.max(abs(p[j+'_exo_torque_Nm']))),
            'human_component_rms_Nm':{k:rms(p[j+'_human_'+k+'_Nm'][mask]) for k in ['feedforward','p','d','feedback','dynamics','clipping']}}
        # 髋关节额外指标:幅值饱和占比、slew 限速占比、最大力矩变化率。
        if 'hip' in j:
            result['joints'][j].update(amplitude_saturation_fraction=float(np.mean(p[j+'_amplitude_clipped'][mask])),
                slew_limited_fraction=float(np.mean(p[j+'_slew_limited'][mask])),
                max_abs_torque_rate_Nm_s=float(np.max(abs(p[j+'_torque_rate_Nm_s']))))
    # 聚合指标:四关节力矩向量 RMS、髋关节力矩 RMS、人体总正功、髋部正功、
    # 外骨骼净功、综合跟踪 RMSE、最大绑带力与最大绑带位移。
    result['aggregate']={'human_four_joint_rms_Nm':rms(np.sqrt(np.sum(np.asarray(torque)**2,axis=0))),
                         'human_hip_rms_Nm':rms(np.sqrt(np.asarray(torque)[0]**2+np.asarray(torque)[2]**2)),
                         'human_positive_work_J':sum(work),'human_hip_positive_work_J':sum(hip_work),
                         'exo_net_work_J':sum(exo_work),'tracking_rmse_deg':rms(errors),
                         'max_strap_force_N':max(float(np.max(p[k])) for k in p if k.endswith('_force_norm_N')),
                         'max_strap_displacement_m':max(float(np.max(p[k])) for k in p if k.endswith('_displacement_norm_m'))}
    # 相位指标:每侧相位估计有效占比,以及 IMU 估计相位与真实相位的 RMSE(仅在有效样本上)。
    for side in ['L','R']:
        valid=sm & (s[side+'_phase_valid']>0)
        result['phase'][side]={'valid_fraction':float(np.mean(s[side+'_phase_valid'][sm])),
            'imu_state_phase_rmse_deg':rms(np.rad2deg(wrap(s[side+'_phase_imu_rad'][valid]-s[side+'_phase_true_rad'][valid]))) if valid.any() else None}
    # 双侧反相误差:右侧相位 - 左侧相位 与 π 的偏差的 RMS(仅在两侧均有效时)。
    valid=sm & (s['L_phase_valid']>0) & (s['R_phase_valid']>0)
    result['phase']['bilateral_antiphase_rmse_deg']=rms(np.rad2deg(wrap(s['R_phase_imu_rad'][valid]-s['L_phase_imu_rad'][valid]-np.pi))) if valid.any() else None
    return result


def compare(out):
    # 加载四种模式的聚合 metrics 与 100Hz 传感器数据。
    modes=['G0','G1','G2','Oracle']; metrics={n:json.loads((out/n/'metrics.json').read_text()) for n in modes}
    data={n:load_csv(out/n/'sensors_100hz.csv') for n in modes}
    # 以 G0(无助力)为基准,计算各模式关键代理指标的相对变化百分比。
    base=metrics['G0']['aggregate']; result={'groups':metrics,'relative_to_G0_percent':{}}
    for n in modes:
        result['relative_to_G0_percent'][n]={k:100*(metrics[n]['aggregate'][k]/base[k]-1) for k in ['human_four_joint_rms_Nm','human_hip_rms_Nm','human_positive_work_J','tracking_rmse_deg']}
    (out/'comparison_metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    # 图 1:2×2 柱状图,对比四模式的四项聚合指标(力矩 RMS、正功、跟踪 RMSE、外骨骼净功)。
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for ax,key,label in zip(axes.flat,['human_four_joint_rms_Nm','human_positive_work_J','tracking_rmse_deg','exo_net_work_J'],
                           ['Human torque-vector RMS (Nm)','Human positive mechanical work (J)','Tracking RMSE (deg)','Exo net mechanical work (J)']):
        values=[metrics[n]['aggregate'][key] for n in modes]; ax.bar(modes,values,color=['gray','#348abd','#e69f00','#009e73'])
        ax.set_title(label); ax.grid(axis='y',alpha=.25)
        # 在每根柱子顶部标注数值。
        for i,v in enumerate(values): ax.annotate(f'{v:.3f}',(i,v),ha='center',va='bottom',fontsize=9)
    fig.suptitle('Stage 1: model proxy metrics, not metabolic or propulsion efficiency'); fig.tight_layout(); fig.savefig(out/'comparison.png',dpi=150); plt.close(fig)
    # 图 2:4×1 时序图,截取 8~14s 稳态窗口,对比四模式的左髋外骨骼力矩、人体力矩、跟踪误差、IMU 相位误差。
    fig,axes=plt.subplots(4,1,figsize=(13,11),sharex=True)
    for n in modes:
        a=data[n]; t=a['time_s']; mask=(t>=8)&(t<=14)
        axes[0].plot(t[mask],a['L_hip_exo_torque_Nm'][mask],label=n)
        axes[1].plot(t[mask],a['L_hip_human_torque_Nm'][mask],label=n)
        axes[2].plot(t[mask],np.rad2deg(a['L_hip_true_rad'][mask]-a['L_hip_target_rad'][mask]),label=n)
        axes[3].plot(t[mask],np.rad2deg(wrap(a['L_phase_imu_rad'][mask]-a['L_phase_true_rad'][mask])),label=n)
    for ax,label in zip(axes,['Exo left hip (Nm)','Human left hip (Nm)','Tracking error (deg)','IMU state-phase error (deg)']):
        ax.set_ylabel(label); ax.grid(alpha=.25); ax.legend(ncol=4)
    axes[-1].set_xlabel('Time (s)'); fig.tight_layout(); fig.savefig(out/'control_comparison.png',dpi=150); plt.close(fig)