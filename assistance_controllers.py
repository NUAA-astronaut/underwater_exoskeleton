"""No MuJoCo, reference trajectory or simulator access in deployable controllers."""
from collections import deque
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class SensorObservation:
    time_s: float
    hip_angle_rad: tuple[float, float]
    hip_velocity_rad_s: tuple[float, float]


def state_phase(angle, velocity, center_rad, scale_frequency_hz):
    """Operational phase portrait: min angle=0, flexion=positive sin(phase).

    Not reference-generator phase; non-sinusoidal motion need not advance uniformly.
    """
    return np.mod(np.arctan2(np.asarray(velocity)/(2*math.pi*scale_frequency_hz),
                            center_rad-np.asarray(angle)), 2*math.pi)


class OnlinePhase:
    def __init__(self, config):
        self.c = dict(config)
        self.history = deque()
        self.last_time = None
        self.phase = np.zeros(2)
        self.valid = np.zeros(2, dtype=bool)

    def update(self, observation: SensorObservation):
        # 读取当前观测时间戳，用于滑动窗口管理。
        t = observation.time_s
        # 强制观测严格按时间递增，防止乱序数据破坏历史窗口与相位连续性。
        if self.last_time is not None and t <= self.last_time:
            raise ValueError('Sensor observations must be strictly chronological')
        self.last_time = t
        # 将当前(时间, 两侧髋角)样本压入历史队列。
        self.history.append((t, observation.hip_angle_rad))
        # 滑动窗口：移除早于 t-history_s 的旧样本，保持队列只覆盖最近 history_s 秒。
        while self.history and self.history[0][0] < t-self.c['history_s']:
            self.history.popleft()
        # 提取窗口内所有髋角样本，形状为 (N, 2)。
        q = np.asarray([x[1] for x in self.history])
        # 相位有效判定（两侧独立）：窗口内髋角峰峰值 ≥ min_motion_range_deg，且样本数 ≥ min_history_samples。
        # 静止或历史不足时相位无周期意义，标记为无效，请求力矩应为零。
        self.valid = ((np.ptp(q, axis=0) >= math.radians(self.c['min_motion_range_deg'])) &
                      (len(q) >= self.c['min_history_samples']))
        # 用当前髋角/角速度计算状态相位：atan2(v/(2πf), center-q)，中心角与归一化频率为先验参数。
        self.phase = state_phase(observation.hip_angle_rad, observation.hip_velocity_rad_s,
                                 math.radians(self.c['center_deg']), self.c['scale_frequency_hz'])
        # 返回相位副本与有效标志副本，避免外部修改内部状态。
        return self.phase.copy(), self.valid.copy()


class AssistanceController:
    def __init__(self, config):
        self.c = dict(config)

    def waveform(self, time_s, phase):
        c = self.c['assistance']
        u = np.clip((time_s-c['start_s'])/c['ramp_s'], 0, 1)
        envelope = u*u*(3-2*u)
        return c['amplitude_Nm']*envelope*np.sin(np.asarray(phase)+c['phase_offset_rad'])


class G0(AssistanceController):
    def update(self, observation):
        return np.zeros(2)


class G1(AssistanceController):
    """Preset oscillator; clock and explicit presets ONLY. No target() calls."""
    def update(self, observation):
        c = self.c['g1']
        # 纯开环相位:由系统时钟直接推算,不依赖任何传感器观测。
        phase = 2*math.pi*c['frequency_hz']*(observation.time_s-c['phase_epoch_s'])
        # 左右侧分别加上预设的固定相位偏置,送入共用的波形函数(含爬升包络)。
        return self.waveform(observation.time_s, phase+np.array([c['left_phase_rad'], c['right_phase_rad']]))


class G2(AssistanceController):
    """Causal sensor phase portrait; no clock-based phase propagation."""
    def __init__(self, config):
        # 调用基类构造函数,保存配置副本。
        super().__init__(config)
        # 在线相位估计器:基于传感器观测(髋角度/速度)的相平面法估计步态相位,仅依赖可观测量。
        self.estimator = OnlinePhase(config['phase'])

    def update(self, observation):
        phase, valid = self.estimator.update(observation)
        return self.waveform(observation.time_s, phase)*valid


class MotorResponse:
    """Per-physics-step amplitude clipping -> optional lag -> hard slew limit.

    All parameters are engineering assumptions, not manufacturer specifications.
    Knee command is structurally absent (only two hips enter this interface).
    """
    def __init__(self, config):
        # 保存电机配置副本（幅值限幅、变化率限幅、一阶响应时间常数）。
        self.c = dict(config)
        # 参数合法性校验：限幅必须为正，响应时间常数不能为负（0 表示无延迟）。
        if config['torque_limit_Nm'] <= 0 or config['slew_limit_Nm_s'] <= 0 or config['response_time_s'] < 0:
            raise ValueError('Invalid motor bounds')
        # 上一步实际输出力矩（左右髋），初始为零。
        self.actual = np.zeros(2)
        # 幅值饱和标志：请求力矩是否超出 ±torque_limit_Nm。
        self.saturated = np.zeros(2, dtype=bool)
        # 变化率受限标志：力矩增量是否触及变化率限制。
        self.slew_limited = np.zeros(2, dtype=bool)
        # 本步实际力矩变化率（Nm/s），用于监控与记录。
        self.rate = np.zeros(2)

    def step(self, requested, dt):
        # 输入校验:步长必须为正,请求力矩必须全部有限。
        if dt <= 0 or not np.all(np.isfinite(requested)):
            raise ValueError('Invalid motor input')
        # 第一步:幅值限幅,将请求力矩裁剪到 ±torque_limit 范围内。
        limit = self.c['torque_limit_Nm']
        clipped = np.clip(requested, -limit, limit)
        # 记录哪些通道触发了幅值饱和。
        self.saturated = np.abs(requested) > limit
        # 第二步:一阶滞后(响应延迟)。tau=0 时 gain=1,即无延迟直接跟随。
        tau = self.c['response_time_s']
        gain = -math.expm1(-dt/tau) if tau else 1.0
        # 按一阶系统离散更新律计算本步期望力矩变化量。
        delta = gain*(clipped-self.actual)
        # 第三步:硬 slew 限速。单步最大变化量 = slew_limit × dt。
        max_change = self.c['slew_limit_Nm_s']*dt
        # 记录哪些通道触发了 slew 限速。
        self.slew_limited = np.abs(delta) > max_change
        delta = np.clip(delta, -max_change, max_change)
        # 更新实际输出力矩,并记录力矩变化率(供分析/限速状态输出)。
        self.actual += delta
        self.rate = delta/dt
        return self.actual.copy()


class HumanPD:
    """No group/mode argument; future adaptation is explicit and currently rejected."""
    def __init__(self, base, human):
        # stage 1 暂不支持人体自适应,若配置非 'none' 则直接报错。
        if human['adaptation'] != 'none':
            raise ValueError('Human adaptation is not implemented in stage 1')
        # PD 增益:比例系数 kp 与微分系数 kd。
        self.kp = base['human_kp_Nm_rad']
        self.kd = base['human_kd_Nms_rad']
        # 人体关节力矩幅值上限。
        self.limit = base['human_torque_limit_Nm']
        # 前馈力矩向量(四个关节),形状必须为 (4,)。
        self.ff = np.array(human['feedforward_Nm'], dtype=float)
        if self.ff.shape != (4,): raise ValueError('Four feedforward torques required')
        # 动力学补偿缩放系数:用于部分抵消科氏力/重力等偏置项。
        self.compensation_scale = human['dynamics_compensation_scale']

    def compute(self, qref, vref, q, v, bias):
        # 比例项：关节角误差的比例反馈，Kp*(参考角-实际角)。
        p = self.kp*(qref-q)
        # 微分项：关节角速度误差的微分反馈，Kd*(参考角速度-实际角速度)。
        d = self.kd*(vref-v)
        # 动力学补偿：按缩放系数乘以 MuJoCo 提供的偏置力矩（含重力+科氏/离心项）。
        dynamics = self.compensation_scale*bias
        # 限幅前总力矩 = 比例 + 微分 + 动力学补偿 + 前馈。
        raw = p+d+dynamics+self.ff
        # 关节力矩幅值硬限幅到 ±human_torque_limit_Nm。
        actual = np.clip(raw, -self.limit, self.limit)
        # 返回各分量与最终力矩，供记录与分析：
        # p/d 为单分量，feedback 为 PD 反馈和，feedforward 为前馈，dynamics 为动力学补偿，
        # requested 为限幅前总力矩，actual 为限幅后实际力矩，clipping 为限幅修正量(actual-requested)。
        return dict(p=p, d=d, feedback=p+d, feedforward=self.ff.copy(), dynamics=dynamics,
                    requested=raw, actual=actual, clipping=actual-raw)