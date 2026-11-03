"""Audit completed pilot group alignment, sampling and warnings without rerunning."""
import json
import numpy as np
from assistance_sim import ROOT,MODES
from assistance_analysis import load_csv

out=ROOT/'pelvis_output'
suite=json.loads((out/'suite_metrics.json').read_text());report={}
for case,groups in suite.items():
    base=load_csv(out/case/'G0'/'sensors_100hz.csv')
    reference=json.loads((out/case/'G0'/'manifest.json').read_text())
    result={}
    for mode in MODES:
        a=load_csv(out/case/mode/'sensors_100hz.csv');manifest=json.loads((out/case/mode/'manifest.json').read_text())
        for key in ['base_config','assistance_config','disturbance','initial_qpos','initial_qvel','reference','model_sha256']:
            assert manifest[key]==reference[key],(case,mode,key)
        for key in ['time_s','base_pitch_rad','base_velocity_rad_s','base_acceleration_rad_s2']+[j+'_target_rad' for j in ['L_hip','L_knee','R_hip','R_knee']]:
            assert np.array_equal(a[key],base[key]),(case,mode,key)
        assert np.allclose(np.diff(a['time_s']),.01,atol=1e-12,rtol=0)
        assert sum(groups[mode]['warning_counts'])==0
        assert np.max(abs(a['L_knee_exo_torque_Nm']))==np.max(abs(a['R_knee_exo_torque_Nm']))==0
        result[mode]={'samples':len(a['time_s']),'warning_counts':groups[mode]['warning_counts'],'base_net_work_J':groups[mode]['base']['net_work_J']}
    report[case]=result
(out/'suite_validation.json').write_text(json.dumps({'passed':True,'identical_group_inputs':True,'cases':report},indent=2))
print('Pilot alignment, 100 Hz sampling, knee-zero torque, and warnings checks passed.')
