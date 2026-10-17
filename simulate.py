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
    root = ET.Element('mujoco', model='175cm_70kg_flutter_kick')
    elem(root, 'compiler', angle='radian')
    elem(root, 'option', timestep=c['timestep_s'], gravity='0 0 -9.81', integrator='implicitfast')
    visual=elem(root,'visual'); elem(visual,'global',offwidth=1080,offheight=720)
    default = elem(root, 'default')
    elem(default, 'geom', contype=0, conaffinity=0)
    elem(default, 'joint', limited='true', damping=0.1)
    elem(default, 'site', size=0.012, rgba='1 0.65 0 1')
    world = elem(root, 'worldbody')
    elem(world, 'light', pos='0 -2 3', dir='0 0 -1')
    elem(world, 'geom', type='plane', size='3 3 .1', rgba='.92 .94 .96 1')
    pelvis = elem(world, 'body', name='pelvis', pos='0 0 1.2')
    # Lump all non-leg human mass into fixed pelvis/trunk: total human mass = 70 kg.
    mass = c['human_mass_kg']; lt = .245*c['height_m']; ls = .246*c['height_m']
    elem(pelvis, 'geom', type='box', pos='-.16 0 0', size='.20 .17 .09', mass=mass*.678, rgba='.25 .55 .75 1')
    bodies = {'pelvis': pelvis}
    straps = []
    for side, y in [('L', .115), ('R', -.115)]:
        for exo in [False, True]:
            prefix = 'exo_' if exo else ''
            color = '.85 .35 .12 1' if exo else '.25 .55 .75 .75'
            yy = y + (np.sign(y)*.105 if exo else 0)
            thigh = elem(pelvis, 'body', name=prefix+side+'_thigh', pos=f'0 {yy} 0')
            kwargs = dict(damping=c['exo_joint_damping_Nms_rad'], frictionloss=c['exo_joint_friction_Nm'], armature=c['exo_armature_kgm2']) if exo else {}
            elem(thigh, 'joint', name=prefix+side+'_hip', axis='0 1 0', range='-.35 1.5', **kwargs)
            elem(thigh, 'geom', type='capsule', fromto=f'0 0 0 {lt} 0 0', size=.018 if exo else .065,
                 mass=c['thigh_rail_mass_kg'] if exo else mass*.100, rgba=color)
            shank = elem(thigh, 'body', name=prefix+side+'_shank', pos=f'{lt} 0 0')
            elem(shank, 'joint', name=prefix+side+'_knee', axis='0 -1 0', range='-.15 1.65', **kwargs)
            elem(shank, 'geom', type='capsule', fromto=f'0 0 0 {ls} 0 0', size=.016 if exo else .045,
                 mass=c['shank_rail_mass_kg'] if exo else mass*.0465, rgba=color)
            if exo:
                # Hip housing belongs to fixed support; knee housing belongs to thigh.
                elem(pelvis, 'geom', type='sphere', pos=f'0 {yy} 0', size=.038, mass=c['motor_mass_kg'], rgba='.2 .2 .2 1')
                elem(thigh, 'geom', type='sphere', pos=f'{lt} 0 0', size=.038, mass=c['motor_mass_kg'], rgba='.2 .2 .2 1')
            else:
                foot = elem(shank, 'body', name=side+'_foot', pos=f'{ls} 0 0')
                # Fixed streamlined foot orientation; no ankle DOF, not anatomical angle calibration.
                elem(foot, 'geom', type='box', pos='.09 0 0', size='.09 .045 .025', mass=mass*.0145, rgba=color)
                elem(foot, 'geom', type='box', pos='.27 0 0', size='.14 .09 .009', mass=c['fin_mass_kg_each'], rgba='.15 .2 .3 1')
                bodies[side+'_thigh'] = thigh; bodies[side+'_shank'] = shank
            for segment, body, length in [('thigh', thigh, lt), ('shank', shank, ls)]:
                # Attachment points coincide initially, including lateral cuff reach.
                offset = -np.sign(y)*.105 if exo else 0
                elem(body, 'site', name=prefix+side+'_'+segment+'_strap', pos=f'{length*c["strap_fraction"]} {offset} 0')
                if not exo: straps.append(side+'_'+segment)
    actuator = elem(root, 'actuator')
    for prefix in ['', 'exo_']:
        for j in JOINTS: elem(actuator, 'motor', name=prefix+j+'_motor', joint=prefix+j, gear=1)
    sensors = elem(root, 'sensor')
    for name, angle in zip(IMUS, c['mount_y_deg']):
        pos = '0 0 .10' if name == 'pelvis' else f'{(lt if "thigh" in name else ls)*.45} 0 -.06'
        elem(bodies[name], 'site', name=name+'_imu', pos=pos, euler=f'0 {np.deg2rad(angle)} 0')
        elem(sensors, 'accelerometer', name=name+'_acc', site=name+'_imu')
        elem(sensors, 'gyro', name=name+'_gyro', site=name+'_imu')
    for j in JOINTS:
        elem(sensors, 'jointpos', name=j+'_encoder', joint='exo_'+j)
        elem(sensors, 'jointvel', name=j+'_encoder_velocity', joint='exo_'+j)
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
    out.mkdir(parents=True, exist_ok=True)
    xml, names = build(c); (out/'model.xml').write_text(xml, encoding='utf-8')
    (out/'config_used.json').write_text(json.dumps(c,indent=2), encoding='utf-8')
    m=mujoco.MjModel.from_xml_string(xml); d=mujoco.MjData(m)
    h=np.array([m.joint(j).qposadr[0] for j in JOINTS]); e=np.array([m.joint('exo_'+j).qposadr[0] for j in JOINTS])
    ha=np.array([m.actuator(j+'_motor').id for j in JOINTS]); ea=np.array([m.actuator('exo_'+j+'_motor').id for j in JOINTS])
    pairs=[(m.site(n+'_strap').id,m.site('exo_'+n+'_strap').id) for n in names]
    dt=c['timestep_s']; every=round(1/c['sensor_hz']/dt)
    if every < 1 or not np.isclose(every*dt,1/c['sensor_hz']): raise ValueError('Sensor period must be integer multiple of physics dt')
    rng=np.random.default_rng(c['seed']); bias=np.zeros((5,3)) if ideal else rng.normal(0,np.deg2rad(c['gyro_bias_std_deg_s']),(5,3))
    mount=[rotation_y(np.deg2rad(a)) for a in c['mount_y_deg']]
    angles=np.zeros(5); bias_est=np.zeros((5,3)); calib=[]; previous_gyro=None
    records=[]; max_exo=0.; max_force=0.; max_stretch=0.; max_human=0.
    jp=np.zeros((3,m.nv)); jr=np.zeros((3,m.nv)); jq=np.zeros((3,m.nv))
    handle=None
    if viewer:
        from mujoco import viewer as mjviewer
        handle=mjviewer.launch_passive(m,d)
        handle.cam.lookat[:]=[.3,0,1.1]; handle.cam.distance=2.7; handle.cam.azimuth=100; handle.cam.elevation=-25
    for step in range(round(c['duration_s']/dt)+1):
        started=time.perf_counter()
        # Refresh positions before external forces; recompute all sensors at SAME timestamp after controls.
        mujoco.mj_forward(m,d)
        d.qfrc_applied[:]=0
        for si,sj in pairs:
            mujoco.mj_jacSite(m,d,jp,jr,si); mujoco.mj_jacSite(m,d,jq,jr,sj)
            delta=d.site_xpos[sj]-d.site_xpos[si]
            force=c['strap_stiffness_N_m']*delta+c['strap_damping_Ns_m']*((jq-jp)@d.qvel)
            d.qfrc_applied[:]+=(jp-jq).T@force
            max_force=max(max_force,float(np.linalg.norm(force))); max_stretch=max(max_stretch,float(np.linalg.norm(delta)))
        qref,vref=target(d.time,c['frequency_hz'])
        # Bias compensation belongs to HUMAN actuation only. No exoskeleton compensation.
        tau=c['human_kp_Nm_rad']*(qref-d.qpos[h])+c['human_kd_Nms_rad']*(vref-d.qvel[h])+d.qfrc_bias[h]
        d.ctrl[ha]=np.clip(tau,-c['human_torque_limit_Nm'],c['human_torque_limit_Nm']); d.ctrl[ea]=0
        mujoco.mj_forward(m,d)
        max_exo=max(max_exo,float(np.max(np.abs(d.actuator_force[ea]))))
        max_human=max(max_human,float(np.max(np.abs(d.ctrl[ha]))))
        if not np.all(np.isfinite(d.qpos)): raise RuntimeError('Nonfinite state')
        if step%every==0:
            acc=np.array([d.sensor(n+'_acc').data.copy() for n in IMUS])
            # MuJoCo reports zero cacc on this welded world/static body. A physical
            # supported accelerometer measures specific force -R^T g, not zero.
            static_R=d.site_xmat[m.site('pelvis_imu').id].reshape(3,3)
            acc[0]=static_R.T @ (-m.opt.gravity)
            gyro=np.array([d.sensor(n+'_gyro').data.copy() for n in IMUS])+bias
            if not ideal:
                acc+=rng.normal(0,c['accel_noise_std_m_s2'],acc.shape)
                gyro+=rng.normal(0,np.deg2rad(c['gyro_noise_std_deg_s']),gyro.shape)
            # Known sensor-to-segment installation rotation, applied to raw local measurements.
            ab=np.array([r@a for r,a in zip(mount,acc)])
            gb=np.array([r@g for r,g in zip(mount,gyro)])
            tilt=np.arctan2(-ab[:,0],ab[:,2])
            if d.time < 1.8:
                if d.time >= 1.0:
                    calib.append(gb.copy()); bias_est=np.mean(calib,axis=0)
                angles=tilt.copy()
            else:
                gy=gb[:,1]-bias_est[:,1]
                if previous_gyro is None: previous_gyro=gy
                pred=angles+.5*(previous_gyro+gy)/c['sensor_hz']
                trust=np.clip(1-np.abs(np.linalg.norm(ab,axis=1)-9.81)/2,0,1)
                angles=pred+c['tilt_correction_gain_s_inv']/c['sensor_hz']*trust*wrap(tilt-pred)
            previous_gyro=gb[:,1]-bias_est[:,1]
            est=np.array([angles[1]-angles[0],angles[1]-angles[2],angles[3]-angles[0],angles[3]-angles[4]])
            av=gb[:,1]-bias_est[:,1]
            vest=np.array([av[1]-av[0],av[1]-av[2],av[3]-av[0],av[3]-av[4]])
            row={'time_s':round(step*dt,10)}
            for i,j in enumerate(JOINTS):
                for suffix,val in [('true_rad',d.qpos[h[i]]),('target_rad',qref[i]),('imu_rad',est[i]),('encoder_rad',d.sensor(j+'_encoder').data[0]),('true_rad_s',d.qvel[h[i]]),('imu_rad_s',vest[i]),('encoder_rad_s',d.sensor(j+'_encoder_velocity').data[0]),('human_torque_Nm',d.ctrl[ha[i]]),('exo_torque_Nm',d.actuator_force[ea[i]])]: row[j+'_'+suffix]=float(val)
            truths=np.array([0,d.qpos[h[0]],d.qpos[h[0]]-d.qpos[h[1]],d.qpos[h[2]],d.qpos[h[2]]-d.qpos[h[3]]])
            for i,n in enumerate(IMUS):
                row[n+'_tilt_est_rad']=float(angles[i]); row[n+'_tilt_true_rad']=float(truths[i])
                for k,axis in enumerate('xyz'):
                    row[n+'_acc_'+axis+'_m_s2']=float(acc[i,k]); row[n+'_gyro_'+axis+'_rad_s']=float(gyro[i,k])
            records.append(row)
        if handle:
            if not handle.is_running(): break
            if step%20==0: handle.sync()
            time.sleep(max(0,dt-(time.perf_counter()-started)))
        if step < round(c['duration_s']/dt): mujoco.mj_step(m,d)
    if handle: handle.close()
    with (out/'sensors_100hz.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
    metrics=analyze(records,c,out)
    metrics.update(max_exo_active_torque_Nm=max_exo,max_strap_force_N=max_force,max_strap_displacement_m=max_stretch,max_human_torque_Nm=max_human,
                   ideal_noise=ideal,mujoco_version=mujoco.__version__,physics_steps=step,sensor_samples=len(records),warning_counts=[int(w.number) for w in d.warning])
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    print(json.dumps(metrics,indent=2))
    # Native MuJoCo render; viewer remains an alternative if OpenGL unavailable.
    try:
        with mujoco.Renderer(m,height=720,width=1080) as renderer:
            camera=mujoco.MjvCamera(); camera.lookat[:]=[.25,0,1.05]; camera.distance=2.7; camera.azimuth=100; camera.elevation=-22
            renderer.update_scene(d,camera=camera); plt.imsave(out/'model_preview.png',renderer.render())
        if (out/'render_note.txt').exists(): (out/'render_note.txt').unlink()
    except Exception as exc:
        (out/'render_note.txt').write_text(str(exc),encoding='utf-8')
    return metrics

def analyze(rows,c,out):
    a={k:np.array([r[k] for r in rows]) for k in rows[0]}; t=a['time_s']; steady=t>=6
    if steady.sum()<10: raise ValueError('Run at least 7 seconds for analysis')
    metrics={'steady_start_s':6,'joints':{},'imus':{}}
    fig,axes=plt.subplots(4,2,figsize=(14,12),sharex=True)
    for i,j in enumerate(JOINTS):
        truth=a[j+'_true_rad']; estimate=a[j+'_imu_rad']; encoder=a[j+'_encoder_rad']; ref=a[j+'_target_rad']
        err=np.rad2deg(wrap(estimate-truth))[steady]
        velerr=np.rad2deg(a[j+'_imu_rad_s']-a[j+'_true_rad_s'])[steady]
        f,p=periodogram(truth[steady],fs=c['sensor_hz'])
        metrics['joints'][j]={'imu_rmse_deg':float(np.sqrt(np.mean(err**2))), 'imu_max_abs_deg':float(np.max(np.abs(err))),
            'gyro_joint_velocity_rmse_deg_s':float(np.sqrt(np.mean(velerr**2))),
            'tracking_rmse_deg':float(np.rad2deg(np.sqrt(np.mean((truth[steady]-ref[steady])**2)))),
            'human_exo_rmse_deg':float(np.rad2deg(np.sqrt(np.mean((truth[steady]-encoder[steady])**2)))),
            'dominant_frequency_hz':float(f[np.argmax(p)]), 'ROM_deg':float(np.rad2deg(np.ptp(truth[steady])))}
        for data,label in [(truth,'Human truth'),(ref,'Target'),(estimate,'IMU estimate'),(encoder,'Exo encoder')]: axes[i,0].plot(t,np.rad2deg(data),label=label,lw=1)
        axes[i,0].set_ylabel(j+' (deg)'); axes[i,1].plot(t,np.rad2deg(wrap(estimate-truth)),label='IMU error'); axes[i,1].plot(t,np.rad2deg(encoder-truth),label='Exo - human'); axes[i,1].set_ylabel('Error (deg)')
    for ax in axes.flat: ax.grid(alpha=.25); ax.legend(fontsize=8)
    axes[-1,0].set_xlabel('Time (s)'); axes[-1,1].set_xlabel('Time (s)'); fig.tight_layout(); fig.savefig(out/'joint_analysis.png',dpi=150); plt.close(fig)
    fig,axes=plt.subplots(4,1,figsize=(12,10),sharex=True)
    for ax,j in zip(axes,JOINTS):
        for suffix,label in [('true_rad_s','Human truth'),('imu_rad_s','IMU relative gyro'),('encoder_rad_s','Exo encoder')]:
            ax.plot(t,np.rad2deg(a[j+'_'+suffix]),label=label,lw=.8)
        ax.set_ylabel(j+' (deg/s)'); ax.grid(alpha=.25); ax.legend(fontsize=8)
    axes[-1].set_xlabel('Time (s)'); fig.tight_layout(); fig.savefig(out/'joint_velocity.png',dpi=150); plt.close(fig)
    fig,axes=plt.subplots(5,2,figsize=(14,13),sharex=True)
    for i,n in enumerate(IMUS):
        err=wrap(a[n+'_tilt_est_rad']-a[n+'_tilt_true_rad'])[steady]
        metrics['imus'][n]={'tilt_rmse_deg':float(np.rad2deg(np.sqrt(np.mean(err**2))))}
        f,p=periodogram(a[n+'_gyro_y_rad_s'][steady],fs=c['sensor_hz'])
        metrics['imus'][n]['dominant_gyro_frequency_hz']=None if n=='pelvis' else float(f[np.argmax(p)])
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
