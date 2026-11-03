"""Separate prescribed-pitch experiment; legacy run is unchanged.
Loop follows assistance_sim, using its sensor/control components.
"""
from assistance_sim import *
from pelvis_support import build_moving, PrescribedBase, PaperTarget, old_target, enrich

def run_pelvis(c, ac, mode, out, disturbance, reference="paper", ideal=False, viewer=False, plots=True):
    # 输入校验:模式合法、时长足以覆盖稳态评估、助力爬升时长为正。
    if mode not in MODES: raise ValueError(mode)
    if c['duration_s']<=ac['evaluation']['steady_start_s']+.1: raise ValueError('Insufficient evaluation duration')
    if ac['assistance']['ramp_s']<=0: raise ValueError('Positive ramp duration required')
    out=safe_output(out)
    # 构建 MuJoCo XML 模型并写入输出目录。
    xml,names=build_moving(c)
    (out/'model.xml').write_text(xml,encoding='utf-8')
    m=mujoco.MjModel.from_xml_string(xml); d=mujoco.MjData(m)
    assert m.nv==9
    base=PrescribedBase(m,disturbance)
    trajectory=old_target if reference=="legacy" else PaperTarget(reference=="alternate")
    # h/e:人体与外骨骼各关节在 qpos 中的地址;ha/ea:对应执行器 id。
    h=np.array([m.joint(j).qposadr[0] for j in JOINTS]); e=np.array([m.joint('exo_'+j).qposadr[0] for j in JOINTS])
    ha=np.array([m.actuator(j+'_motor').id for j in JOINTS]); ea=np.array([m.actuator('exo_'+j+'_motor').id for j in JOINTS])
    # One-DOF hinges: qpos and DOF addresses coincide, verified rather than assumed.
    assert all(m.joint(j).qposadr[0]==m.joint(j).dofadr[0] for j in JOINTS+['exo_'+j for j in JOINTS])
    # 记录实验元数据:模式、配置、初始状态、模型哈希及 legacy 源文件哈希,用于可复现性与完整性校验。
    manifest={'disturbance':disturbance,'reference':reference,'reference_label':'digitized visible line plus reconstructed gaps; NOT original data','mode':mode,'base_config':c,'assistance_config':ac,'ideal_noise':ideal,
              'human_and_reference_hash':digest({'base':c,'human':ac['human']}),
              'initial_qpos':d.qpos.tolist(),'initial_qvel':d.qvel.tolist(),'model_sha256':hashlib.sha256(xml.encode()).hexdigest(),
              'legacy_sources_sha256':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['simulate.py','config.json','validate.py']}}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    # 绑带连接点对(人体侧 site,外骨骼侧 site),用于计算绑带力。
    pairs=[(m.site(n+'_strap').id,m.site('exo_'+n+'_strap').id) for n in names]
    # 预分配雅可比矩阵(jp:人体侧,jq:外骨骼侧,jr:旋转雅可比,此处未用)。
    jp=np.zeros((3,m.nv)); jr=np.zeros((3,m.nv)); jq=np.zeros((3,m.nv))
    # 构建各子系统:IMU 管线、在线相位估计器、电机响应模型、人体 PD 控制器、助力控制器。
    imu=IMUPipeline(c,ideal); monitor=OnlinePhase(ac['phase']); motor=MotorResponse(ac['motor']); human=HumanPD(c,ac['human'])
    controller={'G0':G0,'G1':G1,'G2':G2,'Oracle':Oracle}[mode](ac)
    # 传感器采样步长:every 个物理步对应一次传感器读数,必须为整数。
    dt=c['timestep_s']; every=round(1/c['sensor_hz']/dt)
    if every<1 or not np.isclose(every*dt,1/c['sensor_hz']): raise ValueError('Noninteger sensor stride')
    requested=np.zeros(2); records=[]; physics=[]; handle=None
    # 可选:启动 MuJoCo 被动查看器并设置相机视角。
    if viewer:
        from mujoco import viewer as mjviewer
        handle=mjviewer.launch_passive(m,d); handle.cam.lookat[:]=[.3,0,1.1]
        handle.cam.distance=2.7; handle.cam.azimuth=100; handle.cam.elevation=-25
    steps=round(c['duration_s']/dt)
    try:
        for step in range(steps+1):
            started=time.perf_counter(); t=round(step*dt,10)
            base.set_state(d,t)
            # 前向运动学:清零外加力,准备计算绑带力。
            mujoco.mj_forward(m,d); d.qfrc_applied[:]=0
            strap={}
            for name,(si,sj) in zip(names,pairs):
                # 计算人体侧与外骨骼侧绑带点的位置雅可比。
                mujoco.mj_jacSite(m,d,jp,jr,si); mujoco.mj_jacSite(m,d,jq,jr,sj)
                delta=d.site_xpos[sj]-d.site_xpos[si]
                # 绑带力 = 刚度×位移差 + 阻尼×相对速度。
                force=c['strap_stiffness_N_m']*delta+c['strap_damping_Ns_m']*((jq-jp)@d.qvel)
                # 通过雅可比转置将笛卡尔力投影为关节广义力并累加。
                d.qfrc_applied[:]+=(jp-jq).T@force
                # 记录各轴位移/力及模长,供 CSV 输出。
                for axis,x,f in zip('xyz',delta,force):
                    strap[name+'_displacement_'+axis+'_m']=float(x); strap[name+'_force_'+axis+'_N']=float(f)
                strap[name+'_displacement_norm_m']=float(np.linalg.norm(delta)); strap[name+'_force_norm_N']=float(np.linalg.norm(force))
            # 计算人体目标轨迹,PD 控制器输出人体关节力矩。
            qref,vref=trajectory(d.time,c['frequency_hz'])
            parts=human.compute(qref,vref,d.qpos[h],d.qvel[h],d.qfrc_bias[h])
            d.ctrl[ha]=parts['actual']
            # Only exoskeleton actuator controls receive assistance. No assistance
            # term is added to human controls or qfrc_applied (straps only).
            # 电机按请求力矩输出(含幅值与 slew 限制),仅驱动外骨骼髋部执行器。
            applied=motor.step(requested,dt)
            d.ctrl[ea]=[applied[0],0,applied[1],0]
            base_record=base.enforce(m,d)
            if not np.all(np.isfinite(d.qpos)) or not np.all(np.isfinite(d.qvel)): raise RuntimeError('Nonfinite state')
            # 用真实状态计算参考相位(仅 Oracle 可用,G0/G1/G2 不可见)。
            truth_phase=state_phase(d.qpos[h[[0,2]]],d.qvel[h[[0,2]]],np.deg2rad(ac['phase']['center_deg']),ac['phase']['scale_frequency_hz'])
            row={'time_s':t,**strap,**base_record}
            # 记录每个关节的真实/目标/编码器角度、速度、力矩、功率等。
            for i,j in enumerate(JOINTS):
                for suffix,val in [('true_rad',d.qpos[h[i]]),('target_rad',qref[i]),('target_rad_s',vref[i]),('encoder_rad',d.qpos[e[i]]),
                                   ('true_rad_s',d.qvel[h[i]]),('encoder_rad_s',d.qvel[e[i]]),('human_torque_Nm',d.ctrl[ha[i]]),('exo_torque_Nm',d.actuator_force[ea[i]])]:
                    row[j+'_'+suffix]=float(val)
                for key,value in parts.items(): row[j+'_human_'+key+'_Nm']=float(value[i])
                row[j+'_human_power_W']=float(d.ctrl[ha[i]]*d.qvel[h[i]])
                row[j+'_exo_power_W']=float(d.actuator_force[ea[i]]*d.qvel[e[i]])
            # 记录左右侧相位、参考相位、请求力矩及电机饱和/限速状态。
            for i,side in enumerate(['L','R']):
                row[side+'_phase_true_rad']=float(truth_phase[i])
                row[side+'_phase_reference_rad']=float((2*np.pi*((d.time-2)*c['frequency_hz']+i*.5))%(2*np.pi))
                row[side+'_hip_request_held_Nm']=float(requested[i])
                row[side+'_hip_amplitude_clipped']=int(motor.saturated[i])
                row[side+'_hip_slew_limited']=int(motor.slew_limited[i])
                row[side+'_hip_torque_rate_Nm_s']=float(motor.rate[i])
            # Physics CSV uses left endpoints of actual integration intervals.
            if step<steps: physics.append(row.copy())
            # 每隔 every 步执行传感器管线与控制器更新。
            if step%every==0:
                # 读取 IMU 加速度计与陀螺仪数据;骨盆 IMU 用姿态矩阵修正重力方向。
                acc=np.array([d.sensor(n+'_acc').data.copy() for n in IMUS])
                # All five IMUs now read physical moving-body sensors; no static override.
                gyro=np.array([d.sensor(n+'_gyro').data.copy() for n in IMUS])
                # IMU 管线输出滤波后的加速度、角速度及角度/角速度估计。
                acc,gyro,est,vest=imu.update(d.time,acc,gyro)
                observation=SensorObservation(t,tuple(est[[0,2]]),tuple(vest[[0,2]]))
                phase,valid=monitor.update(observation)
                # All modes run the same sensor pipeline. Privileged data enters
                # ONLY Oracle; G0/G1/G2 receive the frozen observable-only object.
                if mode=='Oracle': next_request=controller.update(TruthObservation(t,tuple(truth_phase)))
                else: next_request=controller.update(observation)
                # 记录 IMU 相位、有效性、下一步请求力矩及 IMU 估计角度/速度。
                for i,side in enumerate(['L','R']):
                    row[side+'_phase_imu_rad']=float(phase[i]); row[side+'_phase_valid']=int(valid[i])
                    row[side+'_hip_request_next_Nm']=float(next_request[i])
                for i,j in enumerate(JOINTS):
                    row[j+'_imu_rad']=float(est[i]); row[j+'_imu_rad_s']=float(vest[i])
                # 记录各 IMU 的倾斜估计值与真实值,以及原始加速度/角速度。
                truths=base.q+np.array([0,d.qpos[h[0]],d.qpos[h[0]]-d.qpos[h[1]],d.qpos[h[2]],d.qpos[h[2]]-d.qpos[h[3]]])
                for i,n in enumerate(IMUS):
                    row[n+'_tilt_est_rad']=float(imu.angles[i]); row[n+'_tilt_true_rad']=float(truths[i])
                    for k,axis in enumerate('xyz'):
                        row[n+'_acc_'+axis+'_m_s2']=float(acc[i,k]); row[n+'_gyro_'+axis+'_rad_s']=float(gyro[i,k])
                records.append(row.copy()); requested=np.asarray(next_request).copy()
            # 查看器模式:若窗口关闭则退出;按 20 步同步并限速至实时。
            if handle:
                if not handle.is_running(): break
                if step%20==0: handle.sync()
                time.sleep(max(0,dt-(time.perf_counter()-started)))
            # 执行一步物理积分(最后一步不积分,仅用于记录末态)。
            if step<steps: mujoco.mj_step(m,d)
    finally:
        if handle: handle.close()
    # 写出传感器(100Hz)与物理步(全步频)两个 CSV。
    write_csv(out/'sensors_100hz.csv',records); write_csv(out/'physics.csv',physics)
    from assistance_analysis import summarize
    metrics=summarize(records,physics,c,ac)
    enrich(metrics,records,physics,{'steady_start_s':ac['evaluation']['steady_start_s'],'dt':dt})
    # 补充元数据:模式、理想噪声标志、MuJoCo 警告计数、版本号、采样数。
    metrics.update(mode=mode,ideal_noise=ideal,warning_counts=[int(w.number) for w in d.warning],
                   mujoco_version=mujoco.__version__,sensor_samples=len(records),physics_intervals=len(physics))
    if plots: analyze(records,c,out)
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False),encoding='utf-8')
    return metrics

