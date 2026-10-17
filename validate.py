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
    c=json.loads((ROOT/'config.json').read_text())
    m=mujoco.MjModel.from_xml_string(build(c)[0]); d=mujoco.MjData(m)
    h=np.array([m.joint(j).qposadr[0] for j in JOINTS])
    e=np.array([m.joint('exo_'+j).qposadr[0] for j in JOINTS])
    # Exact human mass accounting, excluding fins and exoskeleton.
    expected_leg_mass=c['human_mass_kg']*(.1+.0465+.0145)*2
    actual_leg_mass=sum(m.body(n).mass[0] for n in ['L_thigh','L_shank','L_foot','R_thigh','R_shank','R_foot'])-2*c['fin_mass_kg_each']
    assert abs(actual_leg_mass-expected_leg_mass)<1e-10
    for q in [np.zeros(4),np.array([.6,1.0,.2,.3])]:
        d.qpos[h]=q; d.qpos[e]=q; mujoco.mj_forward(m,d)
        # Initial equal-angle human/exo attachments coincide in 3D.
        for n in IMUS[1:]:
            assert np.linalg.norm(d.site_xpos[m.site(n+'_strap').id]-d.site_xpos[m.site('exo_'+n+'_strap').id])<1e-10
        # Relative orientation signs independent of estimator output.
        tilt=[]
        for n,a in zip(IMUS,c['mount_y_deg']):
            segment_R=d.site_xmat[m.site(n+'_imu').id].reshape(3,3)@rotation_y(-np.deg2rad(a))
            tilt.append(np.arctan2(segment_R[0,2],segment_R[0,0]))
        recovered=np.array([tilt[1]-tilt[0],tilt[1]-tilt[2],tilt[3]-tilt[0],tilt[3]-tilt[4]])
        assert np.max(np.abs(recovered-q))<1e-10
    # Local gyro -> anatomical axes, known joint rates.
    d.qvel[h]=[.5,-.3,-.7,.2]; mujoco.mj_forward(m,d)
    gy=[(rotation_y(np.deg2rad(a))@d.sensor(n+'_gyro').data)[1] for n,a in zip(IMUS,c['mount_y_deg'])]
    recovered=np.array([gy[1]-gy[0],gy[1]-gy[2],gy[3]-gy[0],gy[3]-gy[4]])
    assert np.max(np.abs(recovered-d.qvel[h]))<1e-10
    results={}; arrays=[]
    for dt in [.001,.0005]:
        cfg=copy.deepcopy(c); cfg['timestep_s']=dt; cfg['duration_s']=12
        out=ROOT/'validation'/str(dt)
        with contextlib.redirect_stdout(io.StringIO()): metrics=run(cfg,out,ideal=True)
        assert metrics['max_exo_active_torque_Nm']==0
        assert sum(metrics['warning_counts'])==0
        with (out/'sensors_100hz.csv').open() as f: rows=list(csv.DictReader(f))
        t=np.array([float(r['time_s']) for r in rows]); assert np.max(np.abs(np.diff(t)-.01))<1e-10
        arrays.append(np.array([[float(r[j+'_true_rad']) for j in JOINTS] for r in rows])[t>=6])
        results[str(dt)]=metrics
    # Sample timestamps are rounded for steady selection at floating point boundaries.
    n=min(len(a) for a in arrays)
    if len(arrays[0])!=len(arrays[1]): raise AssertionError('Steady sample alignment mismatch')
    error=np.rad2deg(np.sqrt(np.mean((arrays[0][:n]-arrays[1][:n])**2,axis=0)))
    assert np.max(error)<.25, error
    report={'passed':True,'geometry_and_gyro_checks':True,'zero_exo_torque':True,'sampling_100hz':True,'no_solver_warnings':True,'dt_halving_joint_rmse_deg':dict(zip(JOINTS,error.tolist())),'runs':results}
    (ROOT/'validation'/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='runs'},indent=2))

if __name__=='__main__': main()
