"""Prescribed sagittal base motion. Disturbance values are hypotheses, not Yang data."""
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from scipy.interpolate import CubicSpline
from simulate import ROOT, build, target as old_target

class PaperTarget:
    def __init__(self, alternate=False):
        a=np.genfromtxt(ROOT/'paper_reference'/'yang_normal_reconstructed.csv',delimiter=',',names=True)
        self.splines=[CubicSpline(a['cycle_fraction'],np.deg2rad(a[n+('_alt_deg' if alternate else '_deg')]),bc_type='periodic') for n in ['hip','knee']]
    def __call__(self,t,f):
        if t<=2:return np.zeros(4),np.zeros(4)
        u=min((t-2)/2,1);env=10*u**3-15*u**4+6*u**5;de=(30*u**2-60*u**3+30*u**4)/2 if u<1 else 0
        q=[];v=[]
        for shift in [0,.5]:
            p=((t-2)*f+shift)%1
            for s in self.splines:q.append(env*s(p));v.append(de*s(p)+env*f*s(p,1))
        return np.array(q),np.array(v)

def motion(t,c):
    if t<=2:return 0.,0.,0.
    u=min((t-2)/2,1);b=10*u**3-15*u**4+6*u**5
    bd=(30*u*u-60*u**3+30*u**4)/2 if u<1 else 0
    bdd=(60*u-180*u*u+120*u**3)/4 if u<1 else 0
    amp=np.deg2rad(c['amplitude_deg']);w=2*np.pi*c['frequency_hz'];p=w*(t-2)+np.deg2rad(c['phase_deg'])
    return amp*b*np.sin(p),amp*(bd*np.sin(p)+b*w*np.cos(p)),amp*(bdd*np.sin(p)+2*bd*w*np.cos(p)-b*w*w*np.sin(p))

def build_moving(c):
    xml,names=build(c);root=ET.fromstring(xml);pelvis=root.find(".//body[@name='pelvis']")
    root.find('option').set('tolerance','1e-12')
    pelvis.insert(0,ET.Element('joint',name='base_pitch',axis='0 1 0',limited='false',damping='0',frictionloss='0'))
    return ET.tostring(root,encoding='unicode'),names

class PrescribedBase:
    def __init__(self,m,c):
        self.c=c;self.index=m.joint('base_pitch').dofadr[0];self.max_residual=0
    def set_state(self,d,t):
        self.q,self.v,self.a=motion(t,self.c)
        d.qpos[self.index]=self.q;d.qvel[self.index]=self.v
    def enforce(self,m,d):
        # Solve scalar reaction torque with all other applied forces unchanged.
        # Numerical slope includes the MuJoCo constraint/friction solution.
        idx=self.index;original=d.qfrc_applied[idx];torque=0.
        for _ in range(20):
            d.qfrc_applied[idx]=original+torque;mujoco.mj_forward(m,d)
            error=self.a-d.qacc[idx]
            if abs(error)<1e-8:break
            current=d.qacc[idx]
            d.qfrc_applied[idx]=original+torque+1.;mujoco.mj_forward(m,d)
            slope=d.qacc[idx]-current
            if abs(slope)<1e-10:raise RuntimeError('Singular base reaction solve')
            torque+=error/slope
        d.qfrc_applied[idx]=original+torque;mujoco.mj_forward(m,d)
        residual=abs(d.qacc[idx]-self.a);self.max_residual=max(residual,self.max_residual)
        if residual>1e-5:raise RuntimeError(f'Prescribed acceleration residual {residual}')
        return {'base_pitch_rad':self.q,'base_velocity_rad_s':self.v,'base_acceleration_rad_s2':self.a,
                'base_reaction_torque_Nm':float(torque),'base_power_W':float(torque*self.v),'base_acceleration_residual':float(residual)}

def enrich(metrics,records,physics,c):
    s=c['steady_start_s'];steady=[r for r in physics if r['time_s']>=s];dt=c['dt']
    power=np.array([r['base_power_W'] for r in steady])
    metrics['base']={'motion_is_hypothetical':True,'net_work_J':float(power.sum()*dt),'positive_work_J':float(np.maximum(power,0).sum()*dt),
                     'max_reaction_torque_Nm':max(abs(r['base_reaction_torque_Nm']) for r in physics),
                     'max_acceleration_residual':max(r['base_acceleration_residual'] for r in physics)}
    ss=[r for r in records if r['time_s']>=s]
    metrics['segment_tilt_rmse_deg']={n:float(np.rad2deg(np.sqrt(np.mean([(r[n+'_tilt_est_rad']-r[n+'_tilt_true_rad'])**2 for r in ss])))) for n in ['pelvis','L_thigh','L_shank','R_thigh','R_shank']}
