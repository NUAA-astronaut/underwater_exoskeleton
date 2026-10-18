"""Independent geometry/sensor and timestep convergence checks."""
import copy
import csv
import json
import contextlib
import io
import numpy as np
import mujoco
from simulate import ROOT, build, run, JOINTS, IMUS, rotation_y

def main():
    """
    独立验证 build() 模型 & run() 仿真管线的正确性。

    检查项:
      1. 腿部质量核算 —— 按 anthropometric 系数 × 人体质量 = 腿部 6 个刚体实际质量（去除脚蹼）
      2. 绑带 site 初始重合 —— 在零关节角和任意关节角下，人体侧与外骨骼侧 strap 都必须在 3D 空间重合
      3. 段 tilt → 关节角可恢复 —— 从 IMU site 的实际旋转矩阵（减去 mount_y 安装角）恢复关节角，
                                     与直接写入的 qpos 完全一致
      4. 陀螺轴变换正确性 —— 给关节速度 qvel 直接赋值，通过 gyro 传感器 + mount 矩阵反推关节速度应精确匹配
      5. dt 收敛性 —— dt=0.001 s 和 dt=0.0005 s 两次稳态运行的关节轨迹 RMSE < 0.25°
      6. 被动外骨骼零力矩 —— ideal 模式下外骨骼不输出力矩，max_exo_active_torque_Nm 必须为 0
      7. 无求解器警告 —— MuJoCo warning 计数和为 0
      8. 传感器采样率 —— CSV 中相邻时间戳差精确为 0.01 s (100 Hz)

    通过后写 validation/report.json，并打印摘要。
    """

    # ==================== 构建模型 & 预索引 ====================
    c=json.loads((ROOT/'config.json').read_text())
    m=mujoco.MjModel.from_xml_string(build(c)[0]); d=mujoco.MjData(m)
    h=np.array([m.joint(j).qposadr[0] for j in JOINTS])
    e=np.array([m.joint('exo_'+j).qposadr[0] for j in JOINTS])

    # ==================== 1. 腿部质量核算 ====================
    # 理论值：大腿 10% + 小腿 4.65% + 脚掌 1.45%，双侧合计；不包含脚蹼（脚蹼是附加 fin_mass_kg_each）
    expected_leg_mass=c['human_mass_kg']*(.1+.0465+.0145)*2
    # 实际值：求和 6 个 body 的质量，再减去 2 只脚蹼（脚蹼 geom 挂在 foot body 下，质量含在 foot body 里）
    actual_leg_mass=sum(m.body(n).mass[0] for n in ['L_thigh','L_shank','L_foot','R_thigh','R_shank','R_foot'])-2*c['fin_mass_kg_each']
    assert abs(actual_leg_mass-expected_leg_mass)<1e-10

    # ==================== 2+3. 绑带几何重合 & 段 tilt → 关节角可恢复 ====================
    # 在两组关节角下验证：零位 + 任意非零关节角
    for q in [np.zeros(4),np.array([.6,1.0,.2,.3])]:
        d.qpos[h]=q; d.qpos[e]=q; mujoco.mj_forward(m,d)

        # ---- 检查绑带 site 空间重合 ----
        # Initial equal-angle human/exo attachments coincide in 3D.
        for n in IMUS[1:]:    # 跳过 pelvis（没有腿 strap）
            assert np.linalg.norm(d.site_xpos[m.site(n+'_strap').id]-d.site_xpos[m.site('exo_'+n+'_strap').id])<1e-10

        # ---- 检查 tilt 递推关节角 ----
        # 思路：从 IMU site 的实际旋转矩阵 xmat 中消去 IMU 安装角 mount_y_deg，得到段坐标系相对于世界的旋转，
        # 取 R[0,0] 和 R[0,2] 算俯仰角 tilt；相邻段 tilt 之差应等于关节角。
        # 这里完全不依赖 run() 里的互补滤波，直接从 MuJoCo 几何真值恢复。
        tilt=[]
        for n,a in zip(IMUS,c['mount_y_deg']):
            segment_R=d.site_xmat[m.site(n+'_imu').id].reshape(3,3)@rotation_y(-np.deg2rad(a))
            tilt.append(np.arctan2(segment_R[0,2],segment_R[0,0]))
        recovered=np.array([tilt[1]-tilt[0],tilt[1]-tilt[2],tilt[3]-tilt[0],tilt[3]-tilt[4]])
        assert np.max(np.abs(recovered-q))<1e-10

    # ==================== 4. 陀螺轴变换正确性 ====================
    # 直接给 qvel 赋值，让传感器读到精确的关节角速度；
    # 用 mount 矩阵旋转 gyro 原始读数到段坐标系，再取 y 轴分量做相邻段差分，应等于我们写入的 qvel。
    # Local gyro -> anatomical axes, known joint rates.
    d.qvel[h]=[.5,-.3,-.7,.2]; mujoco.mj_forward(m,d)
    gy=[(rotation_y(np.deg2rad(a))@d.sensor(n+'_gyro').data)[1] for n,a in zip(IMUS,c['mount_y_deg'])]
    recovered=np.array([gy[1]-gy[0],gy[1]-gy[2],gy[3]-gy[0],gy[3]-gy[4]])
    assert np.max(np.abs(recovered-d.qvel[h]))<1e-10

    # ==================== 5–8. 时间步收敛 + 零力矩 + 采样率 + 无警告 ====================
    results={}; arrays=[]
    for dt in [.001,.0005]:
        cfg=copy.deepcopy(c); cfg['timestep_s']=dt; cfg['duration_s']=12
        out=ROOT/'validation'/str(dt)
        # 静默 run() 内部的 metrics.json 打印，避免污染验证脚本输出
        with contextlib.redirect_stdout(io.StringIO()): metrics=run(cfg,out,ideal=True)

        assert metrics['max_exo_active_torque_Nm']==0          # 外骨骼力矩恒 0（被动）
        assert sum(metrics['warning_counts'])==0                # MuJoCo 无求解器警告

        # 读 CSV 验证时间戳精度 & 截取稳态轨迹
        with (out/'sensors_100hz.csv').open() as f: rows=list(csv.DictReader(f))
        t=np.array([float(r['time_s']) for r in rows]); assert np.max(np.abs(np.diff(t)-.01))<1e-10
        arrays.append(np.array([[float(r[j+'_true_rad']) for j in JOINTS] for r in rows])[t>=6])
        results[str(dt)]=metrics

    # ---- 两次 dt 运行的稳态轨迹 RMSE ----
    # Sample timestamps are rounded for steady selection at floating point boundaries.
    n=min(len(a) for a in arrays)
    if len(arrays[0])!=len(arrays[1]): raise AssertionError('Steady sample alignment mismatch')
    error=np.rad2deg(np.sqrt(np.mean((arrays[0][:n]-arrays[1][:n])**2,axis=0)))
    assert np.max(error)<.25, error          # 减半 dt 后轨迹变化应 < 0.25°，说明数值积分收敛

    # ==================== 写验证报告 ====================
    report={'passed':True,'geometry_and_gyro_checks':True,'zero_exo_torque':True,'sampling_100hz':True,'no_solver_warnings':True,'dt_halving_joint_rmse_deg':dict(zip(JOINTS,error.tolist())),'runs':results}
    (ROOT/'validation'/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='runs'},indent=2))

if __name__=='__main__': main()