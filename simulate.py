"""Planar flutter-kick sensing prototype. SI units internally; no hardware I/O."""
from pathlib import Path
import argparse
import csv
import json
import time
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from scipy.interpolate import CubicSpline
from scipy.signal import periodogram
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
JOINTS = ['L_hip', 'L_knee', 'R_hip', 'R_knee']
IMUS = ['pelvis', 'L_thigh', 'L_shank', 'R_thigh', 'R_shank']

def elem(parent, tag, **attrs):
    return ET.SubElement(parent, tag, {k: str(v) for k, v in attrs.items()})

def build(c):
    """
    构建 MuJoCo 仿真模型的 XML 字符串，用于模拟蝶泳打腿 (flutter kick) 运动。

    模型由两部分组成：
    1. 人体骨骼（蓝色半透明）：简化的三刚体链 —— 固定骨盆 + 大腿 + 小腿 + 脚掌/脚蹼
    2. 外骨骼（橙色不透明）：紧贴人体外侧的辅助机构，包含髋/膝电机

    两者之间通过虚拟弹簧-阻尼绑带 (strap) 在大腿/小腿中段连接。

    参数:
        c (dict): 配置字典，包含人体尺寸、质量、外骨骼参数、传感器参数等。

    返回:
        tuple: (xml_string, strap_names)
            - xml_string: 完整的 MuJoCo XML 模型字符串
            - strap_names: 人体侧所有绑带 site 名称的列表，用于后续配对弹簧力计算
    """

    # ==================== 根节点 & 全局配置 ====================
    root = ET.Element('mujoco', model='175cm_70kg_flutter_kick')
    elem(root, 'compiler', angle='radian')                           # 角度单位：弧度
    elem(root, 'option', timestep=c['timestep_s'], gravity='0 0 -9.81', integrator='implicitfast')
    visual=elem(root,'visual'); elem(visual,'global',offwidth=1080,offheight=720)  # 离屏渲染分辨率

    # ==================== 默认属性 ====================
    # 所有 geom 默认关闭碰撞检测 (contype=0, conaffinity=0)，因为本模型不依赖接触求解
    default = elem(root, 'default')
    elem(default, 'geom', contype=0, conaffinity=0)
    elem(default, 'joint', limited='true', damping=0.1)              # 关节带限位和微小阻尼
    elem(default, 'site', size=0.012, rgba='1 0.65 0 1')            # site 默认显示为橙色小球

    # ==================== 世界 & 固定体 ====================
    world = elem(root, 'worldbody')
    elem(world, 'light', pos='0 -2 3', dir='0 0 -1')                 # 场景光源（观察者侧向下照射）
    elem(world, 'geom', type='plane', size='3 3 .1', rgba='.92 .94 .96 1')  # 地面（仅视觉，无接触）

    # 骨盆：简化的躯干，人体全部非腿部质量集中于此，固定在世界上不参与动力学
    pelvis = elem(world, 'body', name='pelvis', pos='0 0 1.2')
    # Lump all non-leg human mass into fixed pelvis/trunk: total human mass = 70 kg.
    mass = c['human_mass_kg']; lt = .245*c['height_m']; ls = .246*c['height_m']  # lt=大腿长, ls=小腿长
    elem(pelvis, 'geom', type='box', pos='-.16 0 0', size='.20 .17 .09', mass=mass*.678, rgba='.25 .55 .75 1')

    bodies = {'pelvis': pelvis}   # 用于后续按名字查找 body 的字典
    straps = []                    # 收集人体侧绑带 site 名，供 run() 配对弹簧力

    # ==================== 双侧腿 + 外骨骼 循环 ====================
    # 外层循环：左右腿（L 在 y=+0.115, R 在 y=-0.115）
    # 内层循环：人体段 (exo=False) + 外骨骼段 (exo=True)
    for side, y in [('L', .115), ('R', -.115)]:
        for exo in [False, True]:
            prefix = 'exo_' if exo else ''                                    # 外骨骼命名前缀
            color = '.85 .35 .12 1' if exo else '.25 .55 .75 .75'            # 外骨骼橙色不透明 / 人体蓝色半透明
            yy = y + (np.sign(y)*.105 if exo else 0)                         # 外骨骼比人体偏外侧 10.5 cm（绑带跨距）

            # ---- 大腿 ----
            thigh = elem(pelvis, 'body', name=prefix+side+'_thigh', pos=f'0 {yy} 0')
            # 外骨骼关节显式设置阻尼、摩擦、转动惯量；人体关节仅用 default
            kwargs = dict(damping=c['exo_joint_damping_Nms_rad'], frictionloss=c['exo_joint_friction_Nm'], armature=c['exo_armature_kgm2']) if exo else {}
            elem(thigh, 'joint', name=prefix+side+'_hip', axis='0 1 0', range='-.35 1.5', **kwargs)  # 髋关节：绕 y 轴（冠状面内）
            elem(thigh, 'geom', type='capsule', fromto=f'0 0 0 {lt} 0 0', size=.018 if exo else .065,
                 mass=c['thigh_rail_mass_kg'] if exo else mass*.100, rgba=color)  # 外骨骼细 rail / 人体粗大腿段

            # ---- 小腿 ----
            shank = elem(thigh, 'body', name=prefix+side+'_shank', pos=f'{lt} 0 0')
            elem(shank, 'joint', name=prefix+side+'_knee', axis='0 -1 0', range='-.15 1.65', **kwargs)  # 膝关节：绕 -y 轴（屈膝方向符号相反）
            elem(shank, 'geom', type='capsule', fromto=f'0 0 0 {ls} 0 0', size=.016 if exo else .045,
                 mass=c['shank_rail_mass_kg'] if exo else mass*.0465, rgba=color)

            if exo:
                # 外骨骼电机 housing：髋关节电机固定在骨盆上，膝关节电机固定在大腿上（随大腿运动）
                # Hip housing belongs to fixed support; knee housing belongs to thigh.
                elem(pelvis, 'geom', type='sphere', pos=f'0 {yy} 0', size=.038, mass=c['motor_mass_kg'], rgba='.2 .2 .2 1')
                elem(thigh, 'geom', type='sphere', pos=f'{lt} 0 0', size=.038, mass=c['motor_mass_kg'], rgba='.2 .2 .2 1')
            else:
                # 人体侧：追加脚掌 + 脚蹼（无踝关节 DOF，脚固定在小腿前方）
                foot = elem(shank, 'body', name=side+'_foot', pos=f'{ls} 0 0')
                # Fixed streamlined foot orientation; no ankle DOF, not anatomical angle calibration.
                elem(foot, 'geom', type='box', pos='.09 0 0', size='.09 .045 .025', mass=mass*.0145, rgba=color)
                elem(foot, 'geom', type='box', pos='.27 0 0', size='.14 .09 .009', mass=c['fin_mass_kg_each'], rgba='.15 .2 .3 1')
                bodies[side+'_thigh'] = thigh; bodies[side+'_shank'] = shank  # 记录 body 引用供后续 IMU site 使用

            # ---- 绑带 attachment site ----
            # 在大腿、小腿各段的 strap_fraction 位置各放一个 site；外骨骼侧有横向 10.5 cm 偏移
            for segment, body, length in [('thigh', thigh, lt), ('shank', shank, ls)]:
                # Attachment points coincide initially, including lateral cuff reach.
                offset = -np.sign(y)*.105 if exo else 0
                elem(body, 'site', name=prefix+side+'_'+segment+'_strap', pos=f'{length*c["strap_fraction"]} {offset} 0')
                if not exo: straps.append(side+'_'+segment)   # 仅记录人体侧名字

    # ==================== 执行器 ====================
    # 为人体和外骨骼的四个关节各创建一个位置/力矩电机，gear=1 表示控制量直接作为关节力矩
    actuator = elem(root, 'actuator')
    for prefix in ['', 'exo_']:
        for j in JOINTS: elem(actuator, 'motor', name=prefix+j+'_motor', joint=prefix+j, gear=1)

    # ==================== 传感器 ====================
    sensors = elem(root, 'sensor')

    # ---- IMU ----
    # 每个 IMU 先在对应的 body 上创建一个 site（带 euler 安装角），然后在 site 上挂加速度计 + 陀螺仪
    for name, angle in zip(IMUS, c['mount_y_deg']):
        pos = '0 0 .10' if name == 'pelvis' else f'{(lt if "thigh" in name else ls)*.45} 0 -.06'  # 骨盆上偏前上方，腿段在 45% 长度处偏前下方
        elem(bodies[name], 'site', name=name+'_imu', pos=pos, euler=f'0 {np.deg2rad(angle)} 0')
        elem(sensors, 'accelerometer', name=name+'_acc', site=name+'_imu')
        elem(sensors, 'gyro', name=name+'_gyro', site=name+'_imu')

    # ---- 外骨骼关节编码器 ----
    for j in JOINTS:
        elem(sensors, 'jointpos', name=j+'_encoder', joint='exo_'+j)          # 外骨骼侧关节位置
        elem(sensors, 'jointvel', name=j+'_encoder_velocity', joint='exo_'+j) # 外骨骼侧关节速度

    # ==================== 收尾 ====================
    ET.indent(root)
    return ET.tostring(root, encoding='unicode'), straps

# Smooth heuristic waveform informed by Yang table 2, NOT digitized subject data.
HIP = CubicSpline([0,.227,.519,.69,1], np.deg2rad([-1.16,20,50.19,30,-1.16]), bc_type='periodic')
KNEE = CubicSpline([0,.126,.22,.375,.52,.666,.76,.807,1], np.deg2rad([0,30,60,67.51,65,30,0,-.75,0]), bc_type='periodic')

def target(t, freq):
    if t <= 2: return np.zeros(4), np.zeros(4)
    u = min((t-2)/2, 1); envelope = 10*u**3-15*u**4+6*u**5
    de = (30*u**2-60*u**3+30*u**4)/2 if u < 1 else 0
    p = (t-2)*freq
    q=[]; v=[]
    for phase in [p, p+.5]:
        for spline in [HIP,KNEE]:
            value=spline(phase%1); q.append(envelope*value)
            v.append(de*value+envelope*freq*spline(phase%1,1))
    return np.array(q), np.array(v)

def wrap(a): return (a+np.pi)%(2*np.pi)-np.pi

def rotation_y(a):
    return np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]])

def run(c, out, viewer=False, ideal=False):
    """
    运行完整的蝶泳打腿仿真，持续 duration_s 秒。

    流程概览：
      1) 用 build() 生成 XML → 构建 MuJoCo 模型 MjModel / 数据 MjData
      2) 预先查好关节位置/速度/执行器/绑带 site 在 qpos / ctrl / sensor 里的索引
      3) 主循环：每个 dt 步
           a. 用 mj_forward 刷新运动学
           b. 计算人体-外骨骼绑带的弹簧-阻尼内力，以 qfrc_applied 形式注入
           c. 计算目标轨迹 target() → 人采用 PD+重力补偿控制，外骨骼力矩为 0（被动跟随）
           d. 每 every 个物理步采集一次 IMU（加性噪声 + 偏置），跑互补滤波更新 segment tilt
      4) 写 CSV、调 analyze() 画图、汇总 metrics.json、渲染 model_preview.png

    参数:
        c (dict): 仿真配置，含时间步长、人体/外骨骼参数、传感器参数、控制器增益等。
        out (Path): 输出目录（自动创建）。
        viewer (bool): True 时启动 MuJoCo 被动实时 viewer（依赖 OpenGL）。
        ideal (bool): True 时 IMU 无噪声、陀螺无偏置，作为对照实验。

    返回:
        dict: analyze() 产出的所有统计指标 + 若干运行时峰值和硬件参数。
    """

    # ==================== 初始化与索引预计算 ====================
    out.mkdir(parents=True, exist_ok=True)
    xml, names = build(c); (out/'model.xml').write_text(xml, encoding='utf-8')   # 把生成的模型落盘便于调试
    (out/'config_used.json').write_text(json.dumps(c,indent=2), encoding='utf-8')
    m=mujoco.MjModel.from_xml_string(xml); d=mujoco.MjData(m)

    # 预索引：qpos/qvel 中人体关节(h) 和外骨骼关节(e) 的起始地址；共 4 个关节 × 每关节 1 DOF
    h=np.array([m.joint(j).qposadr[0] for j in JOINTS]); e=np.array([m.joint('exo_'+j).qposadr[0] for j in JOINTS])
    # 执行器 id：人体侧 (ha) 与外骨骼侧 (ea)
    ha=np.array([m.actuator(j+'_motor').id for j in JOINTS]); ea=np.array([m.actuator('exo_'+j+'_motor').id for j in JOINTS])
    # 绑带 site 配对：(人体侧 site id, 外骨骼侧 site id)
    pairs=[(m.site(n+'_strap').id,m.site('exo_'+n+'_strap').id) for n in names]

    # 传感器下采样比：every 个物理步 = 1 个传感器样本；要求 every·dt 恰好等于 1/sensor_hz
    dt=c['timestep_s']; every=round(1/c['sensor_hz']/dt)
    if every < 1 or not np.isclose(every*dt,1/c['sensor_hz']): raise ValueError('Sensor period must be integer multiple of physics dt')

    # 随机数源 + 陀螺偏置：ideal 模式下零偏置；否则从正态分布采样固定偏置
    rng=np.random.default_rng(c['seed']); bias=np.zeros((5,3)) if ideal else rng.normal(0,np.deg2rad(c['gyro_bias_std_deg_s']),(5,3))
    # IMU 安装矩阵：把"传感器坐标系"的原始测量转到"段坐标系"
    mount=[rotation_y(np.deg2rad(a)) for a in c['mount_y_deg']]

    # 互补滤波状态量
    angles=np.zeros(5); bias_est=np.zeros((5,3)); calib=[]; previous_gyro=None
    # 记录容器 & 运行峰值监控
    records=[]; max_exo=0.; max_force=0.; max_stretch=0.; max_human=0.
    # 预分配雅可比缓存：jp/jr/jq 形状 (3, m.nv)，反复给 mj_jacSite 复用避免分配
    jp=np.zeros((3,m.nv)); jr=np.zeros((3,m.nv)); jq=np.zeros((3,m.nv))
    handle=None

    # ==================== 可选 viewer ====================
    if viewer:
        from mujoco import viewer as mjviewer
        handle=mjviewer.launch_passive(m,d)
        handle.cam.lookat[:]=[.3,0,1.1]; handle.cam.distance=2.7; handle.cam.azimuth=100; handle.cam.elevation=-25

    # ==================== 主仿真循环 ====================
    for step in range(round(c['duration_s']/dt)+1):
        started=time.perf_counter()

        # ---- 绑带弹簧-阻尼力 ----
        # 先 mj_forward 保证 site_xpos/site_xmat 更新；每步清空 qfrc_applied 再重新叠加
        mujoco.mj_forward(m,d)
        d.qfrc_applied[:]=0
        for si,sj in pairs:
            mujoco.mj_jacSite(m,d,jp,jr,si); mujoco.mj_jacSite(m,d,jq,jr,sj)   # jp=人体 site 雅可比, jq=外骨骼 site 雅可比
            delta=d.site_xpos[sj]-d.site_xpos[si]                              # 外骨骼相对于人体的伸展向量
            force=c['strap_stiffness_N_m']*delta+c['strap_damping_Ns_m']*((jq-jp)@d.qvel)  # F = kΔx + cΔv
            d.qfrc_applied[:]+=(jp-jq).T@force                                  # 广义力 = JᵀF；作用在两侧符号相反
            max_force=max(max_force,float(np.linalg.norm(force))); max_stretch=max(max_stretch,float(np.linalg.norm(delta)))

        # ---- 人体 PD + 重力补偿控制 ----
        qref,vref=target(d.time,c['frequency_hz'])       # 带起包络的周期目标轨迹
        # Bias compensation belongs to HUMAN actuation only. No exoskeleton compensation.
        tau=c['human_kp_Nm_rad']*(qref-d.qpos[h])+c['human_kd_Nms_rad']*(vref-d.qvel[h])+d.qfrc_bias[h]
        d.ctrl[ha]=np.clip(tau,-c['human_torque_limit_Nm'],c['human_torque_limit_Nm']); d.ctrl[ea]=0   # 外骨骼力矩恒 0，被动被绑带驱动

        # ---- 第二次 mj_forward：控制输入生效后刷新传感器 ----
        mujoco.mj_forward(m,d)
        max_exo=max(max_exo,float(np.max(np.abs(d.actuator_force[ea]))))
        max_human=max(max_human,float(np.max(np.abs(d.ctrl[ha]))))
        if not np.all(np.isfinite(d.qpos)): raise RuntimeError('Nonfinite state')

        # ---- 传感器采样（每 every 个物理步一次） ----
        if step%every==0:
            acc=np.array([d.sensor(n+'_acc').data.copy() for n in IMUS])
            # MuJoCo reports zero cacc on this welded world/static body. A physical
            # supported accelerometer measures specific force -R^T g, not zero.
            static_R=d.site_xmat[m.site('pelvis_imu').id].reshape(3,3)    # 骨盆 IMU 的旋转矩阵（骨盆固定，所以 cacc 恒为 0）
            acc[0]=static_R.T @ (-m.opt.gravity)                             # 用重力反推骨盆 IMU 的静态加速度计读数
            gyro=np.array([d.sensor(n+'_gyro').data.copy() for n in IMUS])+bias  # 注入固定偏置

            if not ideal:
                acc+=rng.normal(0,c['accel_noise_std_m_s2'],acc.shape)          # 加性高斯噪声
                gyro+=rng.normal(0,np.deg2rad(c['gyro_noise_std_deg_s']),gyro.shape)

            # Known sensor-to-segment installation rotation, applied to raw local measurements.
            ab=np.array([r@a for r,a in zip(mount,acc)])     # 转到段坐标系的加速度
            gb=np.array([r@g for r,g in zip(mount,gyro)])    # 转到段坐标系的角速度
            tilt=np.arctan2(-ab[:,0],ab[:,2])                # 用重力在段坐标系的投影估计段俯仰角（前提：静止或低速）

            # ---- 阶段 1（t<1.0s）---- 静止保持，tilt 直接作为角度初值
            # ---- 阶段 2（1.0s≤t<1.8s）---- 陀螺偏置在线标定：持续采样取均值
            # ---- 阶段 3（t≥1.8s）---- 互补滤波：陀螺仪积分预测 + 加速度计 tilt 修正
            if d.time < 1.8:
                if d.time >= 1.0:
                    calib.append(gb.copy()); bias_est=np.mean(calib,axis=0)
                angles=tilt.copy()
            else:
                gy=gb[:,1]-bias_est[:,1]                                    # y 轴角速度（俯仰）减估计偏置
                if previous_gyro is None: previous_gyro=gy
                pred=angles+.5*(previous_gyro+gy)/c['sensor_hz']            # 梯形法则积分陀螺得角度预测
                trust=np.clip(1-np.abs(np.linalg.norm(ab,axis=1)-9.81)/2,0,1)  # 加速度模偏离 g 越多，信任度越低
                angles=pred+c['tilt_correction_gain_s_inv']/c['sensor_hz']*trust*wrap(tilt-pred)  # 加权修正

            previous_gyro=gb[:,1]-bias_est[:,1]

            # 由段间角度差得到关节估计值：L_hip = L_thigh - pelvis, L_knee = L_thigh - L_shank 等
            est=np.array([angles[1]-angles[0],angles[1]-angles[2],angles[3]-angles[0],angles[3]-angles[4]])
            av=gb[:,1]-bias_est[:,1]
            vest=np.array([av[1]-av[0],av[1]-av[2],av[3]-av[0],av[3]-av[4]])

            # ---- 构建一行 CSV 记录 ----
            row={'time_s':round(step*dt,10)}
            for i,j in enumerate(JOINTS):
                for suffix,val in [('true_rad',d.qpos[h[i]]),('target_rad',qref[i]),('imu_rad',est[i]),('encoder_rad',d.sensor(j+'_encoder').data[0]),('true_rad_s',d.qvel[h[i]]),('imu_rad_s',vest[i]),('encoder_rad_s',d.sensor(j+'_encoder_velocity').data[0]),('human_torque_Nm',d.ctrl[ha[i]]),('exo_torque_Nm',d.actuator_force[ea[i]])]: row[j+'_'+suffix]=float(val)
            # IMU 段角度真值 —— pelvis 为 0，thigh/knee 由人体关节角递推
            truths=np.array([0,d.qpos[h[0]],d.qpos[h[0]]-d.qpos[h[1]],d.qpos[h[2]],d.qpos[h[2]]-d.qpos[h[3]]])
            for i,n in enumerate(IMUS):
                row[n+'_tilt_est_rad']=float(angles[i]); row[n+'_tilt_true_rad']=float(truths[i])
                for k,axis in enumerate('xyz'):
                    row[n+'_acc_'+axis+'_m_s2']=float(acc[i,k]); row[n+'_gyro_'+axis+'_rad_s']=float(gyro[i,k])
            records.append(row)

        # ---- viewer 同步 & 实时节流 ----
        if handle:
            if not handle.is_running(): break
            if step%20==0: handle.sync()
            time.sleep(max(0,dt-(time.perf_counter()-started)))

        # ---- 物理步进（最后一次不 step，保证步数对齐持续时间） ----
        if step < round(c['duration_s']/dt): mujoco.mj_step(m,d)

    # ==================== 收尾：CSV / metrics / 渲染 ====================
    if handle: handle.close()
    with (out/'sensors_100hz.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
    metrics=analyze(records,c,out)
    metrics.update(max_exo_active_torque_Nm=max_exo,max_strap_force_N=max_force,max_strap_displacement_m=max_stretch,max_human_torque_Nm=max_human,
                   ideal_noise=ideal,mujoco_version=mujoco.__version__,physics_steps=step,sensor_samples=len(records),warning_counts=[int(w.number) for w in d.warning])
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    print(json.dumps(metrics,indent=2))

    # 原生 MuJoCo 离屏渲染；若 OpenGL 不可用则写一个 render_note.txt 说明原因
    try:
        with mujoco.Renderer(m,height=720,width=1080) as renderer:
            camera=mujoco.MjvCamera(); camera.lookat[:]=[.25,0,1.05]; camera.distance=2.7; camera.azimuth=100; camera.elevation=-22
            renderer.update_scene(d,camera=camera); plt.imsave(out/'model_preview.png',renderer.render())
        if (out/'render_note.txt').exists(): (out/'render_note.txt').unlink()
    except Exception as exc:
        (out/'render_note.txt').write_text(str(exc),encoding='utf-8')
    return metrics

def analyze(rows,c,out):
    """
    分析 run() 生成的传感器记录，计算指标并输出三张分析图。

    输出文件:
        joint_analysis.png   —— 4 行 × 2 列，左列关节角度时间曲线（真值/目标/IMU/编码器），
                                 右列误差曲线（IMU 误差、编码器-人体差）
        joint_velocity.png   —— 4 行 × 1 列，各关节角速度对比（真值 / IMU 陀螺差分 / 编码器）
        imu_signals.png      —— 5 行 × 2 列，左列各 IMU 三轴加速度原始信号，右列三轴角速度原始信号

    参数:
        rows (list[dict]): run() 中累积的传感器记录列表。
        c (dict): 原始仿真配置（用于 sensor_hz 等）。
        out (Path): 图片输出目录。

    返回:
        dict: metrics 字典，包含:
            - steady_start_s: 稳态分析起点秒数（固定 6 s）
            - joints[j]: 每个关节的 RMSE/ROM/主频等
            - imus[n]: 每个 IMU 的 tilt RMSE 和陀螺 y 轴主频
    """

    # ---- 把 list[dict] 转成 dict[np.ndarray]，便于批量向量化 ----
    # 进入稳态的判断：t >= 6 s（前 6 s 包含起包络阶段，不稳定）
    a={k:np.array([r[k] for r in rows]) for k in rows[0]}; t=a['time_s']; steady=t>=6
    if steady.sum()<10: raise ValueError('Run at least 7 seconds for analysis')
    metrics={'steady_start_s':6,'joints':{},'imus':{}}

    # ==================== 图 1：关节角度 + 误差 ====================
    fig,axes=plt.subplots(4,2,figsize=(14,12),sharex=True)
    for i,j in enumerate(JOINTS):
        truth=a[j+'_true_rad']; estimate=a[j+'_imu_rad']; encoder=a[j+'_encoder_rad']; ref=a[j+'_target_rad']

        # ---- 指标计算（仅稳态区间） ----
        err=np.rad2deg(wrap(estimate-truth))[steady]                                 # IMU 关节角估计误差（wrap 处理 ±π 跳变）
        velerr=np.rad2deg(a[j+'_imu_rad_s']-a[j+'_true_rad_s'])[steady]             # IMU 角速度估计误差
        f,p=periodogram(truth[steady],fs=c['sensor_hz'])                            # Welch 功率谱估计主频
        metrics['joints'][j]={'imu_rmse_deg':float(np.sqrt(np.mean(err**2))), 'imu_max_abs_deg':float(np.max(np.abs(err))),
            'gyro_joint_velocity_rmse_deg_s':float(np.sqrt(np.mean(velerr**2))),
            'tracking_rmse_deg':float(np.rad2deg(np.sqrt(np.mean((truth[steady]-ref[steady])**2)))),  # 人体跟踪目标的 RMSE
            'human_exo_rmse_deg':float(np.rad2deg(np.sqrt(np.mean((truth[steady]-encoder[steady])**2)))),  # 人体-外骨骼跟随误差
            'dominant_frequency_hz':float(f[np.argmax(p)]),                           # 真值主频，应接近 frequency_hz
            'ROM_deg':float(np.rad2deg(np.ptp(truth[steady])))}                       # 关节活动范围

        # ---- 绘图 ----
        # 左列：四条时间曲线叠加在同一张图
        for data,label in [(truth,'Human truth'),(ref,'Target'),(estimate,'IMU estimate'),(encoder,'Exo encoder')]: axes[i,0].plot(t,np.rad2deg(data),label=label,lw=1)
        axes[i,0].set_ylabel(j+' (deg)')
        # 右列：两类误差
        axes[i,1].plot(t,np.rad2deg(wrap(estimate-truth)),label='IMU error')
        axes[i,1].plot(t,np.rad2deg(encoder-truth),label='Exo - human')
        axes[i,1].set_ylabel('Error (deg)')
    for ax in axes.flat: ax.grid(alpha=.25); ax.legend(fontsize=8)
    axes[-1,0].set_xlabel('Time (s)'); axes[-1,1].set_xlabel('Time (s)'); fig.tight_layout(); fig.savefig(out/'joint_analysis.png',dpi=150); plt.close(fig)

    # ==================== 图 2：关节角速度 ====================
    fig,axes=plt.subplots(4,1,figsize=(12,10),sharex=True)
    for ax,j in zip(axes,JOINTS):
        # 三种角速度：MuJoCo 真值 qvel / IMU 段间陀螺差分 / 外骨骼编码器差分
        for suffix,label in [('true_rad_s','Human truth'),('imu_rad_s','IMU relative gyro'),('encoder_rad_s','Exo encoder')]:
            ax.plot(t,np.rad2deg(a[j+'_'+suffix]),label=label,lw=.8)
        ax.set_ylabel(j+' (deg/s)'); ax.grid(alpha=.25); ax.legend(fontsize=8)
    axes[-1].set_xlabel('Time (s)'); fig.tight_layout(); fig.savefig(out/'joint_velocity.png',dpi=150); plt.close(fig)

    # ==================== 图 3：IMU 原始信号 ====================
    fig,axes=plt.subplots(5,2,figsize=(14,13),sharex=True)
    for i,n in enumerate(IMUS):
        # 段 tilt 估计误差（互补滤波输出 vs 递推真值）
        err=wrap(a[n+'_tilt_est_rad']-a[n+'_tilt_true_rad'])[steady]
        metrics['imus'][n]={'tilt_rmse_deg':float(np.rad2deg(np.sqrt(np.mean(err**2))))}
        # 陀螺 y 轴功率谱 —— pelvis 静止所以主频设为 None
        f,p=periodogram(a[n+'_gyro_y_rad_s'][steady],fs=c['sensor_hz'])
        metrics['imus'][n]['dominant_gyro_frequency_hz']=None if n=='pelvis' else float(f[np.argmax(p)])

        # 左列：三轴加速度；右列：三轴角速度（°/s 方便读）
        for axis in 'xyz':
            axes[i,0].plot(t,a[n+'_acc_'+axis+'_m_s2'],label=axis,lw=.7)
            axes[i,1].plot(t,np.rad2deg(a[n+'_gyro_'+axis+'_rad_s']),label=axis,lw=.7)
        axes[i,0].set_ylabel(n+' acc (m/s2)'); axes[i,1].set_ylabel('Gyro (deg/s)')
    for ax in axes.flat: ax.grid(alpha=.25); ax.legend(fontsize=8)
    axes[-1,0].set_xlabel('Time (s)'); axes[-1,1].set_xlabel('Time (s)'); fig.tight_layout(); fig.savefig(out/'imu_signals.png',dpi=150); plt.close(fig)
    return metrics

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,default=ROOT/'config.json'); p.add_argument('--out',type=Path,default=ROOT/'output'); p.add_argument('--viewer',action='store_true'); p.add_argument('--ideal',action='store_true'); p.add_argument('--duration',type=float); p.add_argument('--dt',type=float)
    args=p.parse_args(); config=json.loads(args.config.read_text(encoding='utf-8'))
    if args.duration: config['duration_s']=args.duration
    if args.dt: config['timestep_s']=args.dt
    run(config,args.out,args.viewer,args.ideal)