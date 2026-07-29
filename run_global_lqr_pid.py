"""全局路径纵横向控制学习模板：速度 PID + 动态自行车模型 LQR。

这个文件是可直接运行的独立入口，通讯流程与 ``run_hmxzw.py`` 一致：

1. 使用 config-center/field/interface 创建 multicast channels；
2. 接收 ActorPrepare，从 brief_data 的地图、起点和终点生成 XODR 全局路径；
3. 在发送 ActorPrepareResult 之前先发布一帧与初始速度一致的中性控制；
4. 接收 Notify 和 INS，发布 VehicleControl；
5. 不订阅 NPC，不考虑旁车或障碍物。

图片中的车辆参数
==================

以下数值来自参数1.png、参数2.png；``*2`` 已换算为整轴刚度。

================  ===========  =============  ==========================
符号              数值         单位           含义
================  ===========  =============  ==========================
m                 1134         kg             整车质量
ms                1008.1       kg             簧上质量
muf               71.4         kg             前轴簧下质量
mur               54.5         kg             后轴簧下质量
Iz                1343.1       kg·m²          横摆转动惯量
Ix                440.6        kg·m²          侧倾转动惯量
lf                1.04         m              质心到前轴
lr                1.56         m              质心到后轴
L                 2.60         m              轴距
B                 1.48         m              轮距
hcg               0.54         m              质心高度
hr                0.165        m              侧倾中心高度
kphif             22002        N·m/rad        前悬架侧倾刚度
kphir             14381        N·m/rad        后悬架侧倾刚度
bphif             2000         N·m·s/rad      前悬架侧倾阻尼
bphir             2000         N·m·s/rad      后悬架侧倾阻尼
f                 0.008        -              滚动阻力系数
CD                0.3          -              空气阻力系数
rho               1.206        kg/m³          空气密度
A                 1.6          m²             正面迎风面积
Iw                0.8          kg·m²          车轮转动惯量
Rw                0.287        m              车轮滚动半径
g                 9.8          m/s²           重力加速度
Cf                86320        N/rad          前轴侧偏刚度（43160×2）
Cr                58420        N/rad          后轴侧偏刚度（29210×2）
Cxf               179040       N              前轴纵向刚度（89520×2）
Cxr               121480       N              后轴纵向刚度（60740×2）
================  ===========  =============  ==========================

方向盘传动比没有出现在三张图片中，因此它不是“实测图片参数”。代码把
``steering_ratio`` 单独作为必须标定的执行器参数，默认 16.0。

四状态侧向-横摆 LQR 实际使用 m、Iz、lf、lr、Cf、Cr；纵向阻力前馈使用
m、f、rho、CD、A、g。图片中的侧倾参数被完整记录但没有硬塞进状态矩阵，
因为当前 INS 通讯没有提供可确认的侧倾角/侧倾角速度。把不可测状态固定为
零反而不是严格模型；若平台补充 roll/roll_rate，应另行扩展六状态模型和
状态观测器，而不能只在 A 矩阵中增加两行。

参数3.png 表明轮胎在大侧偏角/大滑移率下会饱和。LQR 的 Cf/Cr 取图中
线性段斜率，因此速度规划用 ``v²|kappa| <= max_lateral_accel`` 把正常
跟踪限制在线性工作区；大曲率和调头会先降速，而不是让线性 LQR 在轮胎
饱和后继续错误地增加转角。

动态 LQR 理论模型
================

轮胎在线性区使用

    F_yf = Cf * (delta - (v_y + lf*r)/v_x)
    F_yr = Cr * (      - (v_y - lr*r)/v_x)

误差状态严格选为

    x = [e_y, dot(e_y), e_psi, dot(e_psi)]^T

其中 ``dot(e_psi)=r-v_x*kappa_ref``。连续系统为

    dot(x) = A(v_x) x + B delta + E(v_x) kappa_ref

    A = [[0, 1, 0, 0],
         [0, -(Cf+Cr)/(m v), (Cf+Cr)/m,
             -(lf Cf-lr Cr)/(m v)],
         [0, 0, 0, 1],
         [0, -(lf Cf-lr Cr)/(Iz v),
             (lf Cf-lr Cr)/Iz,
             -(lf² Cf+lr² Cr)/(Iz v)]]

    B = [0, Cf/m, 0, lf Cf/Iz]^T

    E = [0,
         -v²-(lf Cf-lr Cr)/m,
         0,
         -(lf² Cf+lr² Cr)/Iz]^T

代码没有使用 ``Ad≈I+A*dt`` 的欧拉近似，而是对增广矩阵做矩阵指数：

    exp([[A, B, E], [0, 0, 0], [0, 0, 0]] dt)
      -> Ad, Bd, Ed

离散最优控制通过完整 DARE 求解：

    P = Ad'PAd - Ad'PBd(R+Bd'PBd)^-1 Bd'PAd + Q
    K = (R+Bd'PBd)^-1 Bd'PAd

曲率不是用一个拍脑袋的固定前馈系数。对常值扰动
``d=Ed*kappa_ref``，求无限时域二次型的仿射最优项：

    Acl = Ad-Bd K
    s = (I-Acl')^-1 Acl' P d
    delta_ff = -(R+Bd'PBd)^-1 Bd'(P d+s)
    delta = delta_ff-Kx

低于动态轮胎模型有效速度时，使用同样经过矩阵指数、DARE 和仿射扰动
求解的二状态运动学 LQR，而不是把 ``v`` 粗暴钳到某个非零值。

PID 调参学习顺序
================

1. 令 Ki=Kd=0，逐步增加 Kp，直到速度响应足够快但不持续振荡；
2. 增加 Ki 消除坡度、风阻造成的稳态误差，观察 integral 和饱和时间；
3. 少量增加 Kd 抑制超调；本实现对测量值微分并带一阶低通，避免目标跳变；
4. 先固定纵向参数再调 LQR。LQR 中增大 Q[0] 强调横向位置，
   增大 Q[2] 强调航向；增大 R 会减小转向动作；
5. 查看 CSV 中的 P/I/D、LQR K、前馈、反馈、饱和量，不要只看最终转角。

运行 ``python run_global_lqr_pid.py --self-test`` 可离线检查矩阵离散化、
DARE 残差、直线稳定性、左右弯前馈符号和 PID 抗积分饱和。
"""

import argparse
import csv
import importlib
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
ONSITE_ROOT = SCRIPT_DIR.parent
TASK_TIMEOUT_EXIT_CODE = 75
INS_STALL_EXIT_CODE = 76
for _root in (ONSITE_ROOT, SCRIPT_DIR):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))


def clip(value, lower, upper):
    return max(lower, min(upper, value))


def wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def finite_float(value, default=0.0):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


@dataclass(frozen=True)
class VehicleParameters:
    mass: float = 1134.0
    sprung_mass: float = 1008.1
    front_unsprung_mass: float = 71.4
    rear_unsprung_mass: float = 54.5
    yaw_inertia: float = 1343.1
    roll_inertia: float = 440.6
    lf: float = 1.04
    lr: float = 1.56
    wheelbase: float = 2.60
    track_width: float = 1.48
    cg_height: float = 0.54
    roll_center_height: float = 0.165
    front_roll_stiffness: float = 22002.0
    rear_roll_stiffness: float = 14381.0
    front_roll_damping: float = 2000.0
    rear_roll_damping: float = 2000.0
    rolling_resistance: float = 0.008
    drag_coefficient: float = 0.3
    air_density: float = 1.206
    frontal_area: float = 1.6
    wheel_inertia: float = 0.8
    wheel_radius: float = 0.287
    gravity: float = 9.8
    front_cornering_stiffness: float = 43160.0 * 2.0
    rear_cornering_stiffness: float = 29210.0 * 2.0
    front_longitudinal_stiffness: float = 89520.0 * 2.0
    rear_longitudinal_stiffness: float = 60740.0 * 2.0

    def validate(self):
        if abs((self.lf + self.lr) - self.wheelbase) > 1e-9:
            raise ValueError("lf + lr must equal wheelbase")
        positive = (
            self.mass,
            self.yaw_inertia,
            self.lf,
            self.lr,
            self.front_cornering_stiffness,
            self.rear_cornering_stiffness,
        )
        if not all(value > 0.0 for value in positive):
            raise ValueError("dynamic bicycle parameters must be positive")


VEHICLE = VehicleParameters()
VEHICLE.validate()


def matrix_exponential_pade13(matrix):
    """Higham scaling-and-squaring Padé [13/13] matrix exponential."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix exponential requires a square matrix")
    size = matrix.shape[0]
    if size == 0:
        return matrix.copy()
    norm_1 = float(np.linalg.norm(matrix, 1))
    theta_13 = 5.371920351148152
    scale_power = (
        0
        if norm_1 <= theta_13
        else int(math.ceil(math.log(norm_1 / theta_13, 2.0)))
    )
    a = matrix / (2.0 ** scale_power)
    identity = np.eye(size)
    a2 = a @ a
    a4 = a2 @ a2
    a6 = a4 @ a2
    b = (
        64764752532480000.0,
        32382376266240000.0,
        7771770303897600.0,
        1187353796428800.0,
        129060195264000.0,
        10559470521600.0,
        670442572800.0,
        33522128640.0,
        1323241920.0,
        40840800.0,
        960960.0,
        16380.0,
        182.0,
        1.0,
    )
    u = a @ (
        a6 @ (b[13] * a6 + b[11] * a4 + b[9] * a2)
        + b[7] * a6
        + b[5] * a4
        + b[3] * a2
        + b[1] * identity
    )
    v = (
        a6 @ (b[12] * a6 + b[10] * a4 + b[8] * a2)
        + b[6] * a6
        + b[4] * a4
        + b[2] * a2
        + b[0] * identity
    )
    result = np.linalg.solve(v - u, v + u)
    for _ in range(scale_power):
        result = result @ result
    return result


def exact_zero_order_hold(a_continuous, *input_matrices, dt):
    """Exact ZOH discretization for A and any number of input matrices."""
    a_continuous = np.asarray(a_continuous, dtype=float)
    state_size = a_continuous.shape[0]
    matrices = [
        np.asarray(item, dtype=float).reshape(state_size, -1)
        for item in input_matrices
    ]
    input_size = sum(item.shape[1] for item in matrices)
    augmented = np.zeros(
        (state_size + input_size, state_size + input_size),
        dtype=float,
    )
    augmented[:state_size, :state_size] = a_continuous
    if matrices:
        augmented[:state_size, state_size:] = np.hstack(matrices)
    exponential = matrix_exponential_pade13(augmented * float(dt))
    ad = exponential[:state_size, :state_size]
    discrete_inputs = []
    column = state_size
    for matrix in matrices:
        width = matrix.shape[1]
        discrete_inputs.append(
            exponential[:state_size, column:column + width]
        )
        column += width
    return (ad, *discrete_inputs)


def solve_discrete_are_iterative(a, b, q, r, tolerance=1e-11):
    """Solve DARE by value iteration and verify the algebraic residual."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    q = np.asarray(q, dtype=float)
    r = np.asarray(r, dtype=float)
    p = q.copy()
    converged = False
    for iteration in range(1, 20001):
        control_hessian = r + b.T @ p @ b
        gain_term = np.linalg.solve(
            control_hessian, b.T @ p @ a
        )
        next_p = (
            a.T @ p @ a
            - a.T @ p @ b @ gain_term
            + q
        )
        next_p = 0.5 * (next_p + next_p.T)
        scale = max(1.0, float(np.linalg.norm(next_p, ord="fro")))
        if np.linalg.norm(next_p - p, ord="fro") <= tolerance * scale:
            p = next_p
            converged = True
            break
        p = next_p
    if not converged:
        raise RuntimeError("DARE value iteration did not converge")
    hessian = r + b.T @ p @ b
    residual = (
        a.T @ p @ a
        - p
        - a.T @ p @ b
        @ np.linalg.solve(hessian, b.T @ p @ a)
        + q
    )
    residual_norm = float(np.linalg.norm(residual, ord="fro"))
    if residual_norm > 1e-6 * max(
        1.0, float(np.linalg.norm(p, ord="fro"))
    ):
        raise RuntimeError(
            f"DARE residual too large: {residual_norm:.3e}"
        )
    return p, iteration, residual_norm


def solve_lqr_base(a, b, q, r):
    """Solve DARE once for one gain-scheduling operating point."""
    p, iterations, residual = solve_discrete_are_iterative(
        a, b, q, r
    )
    hessian = r + b.T @ p @ b
    gain = np.linalg.solve(hessian, b.T @ p @ a)
    closed_loop = a - b @ gain
    spectral_radius = float(
        np.max(np.abs(np.linalg.eigvals(closed_loop)))
    )
    return {
        "p": p,
        "k": gain,
        "hessian": hessian,
        "closed_loop": closed_loop,
        "spectral_radius": spectral_radius,
        "dare_iterations": iterations,
        "dare_residual": residual,
    }


def constant_disturbance_feedforward(base, b, disturbance):
    """Exact affine optimal action for a constant discrete disturbance."""
    p = base["p"]
    closed_loop = base["closed_loop"]
    identity = np.eye(closed_loop.shape[0])
    costate = np.linalg.solve(
        identity - closed_loop.T,
        closed_loop.T @ p @ disturbance,
    )
    feedforward = -np.linalg.solve(
        base["hessian"],
        b.T @ (p @ disturbance + costate),
    )
    return float(feedforward[0, 0])


def affine_lqr(a, b, disturbance, q, r):
    """Convenience wrapper used by self-tests and one-shot calculations."""
    result = solve_lqr_base(a, b, q, r)
    result["feedforward"] = constant_disturbance_feedforward(
        result, b, disturbance
    )
    return result


@dataclass
class LqrWeights:
    lateral_error: float = 8.0
    lateral_error_rate: float = 1.2
    heading_error: float = 12.0
    heading_error_rate: float = 1.5
    steering: float = 2.0
    kinematic_lateral_error: float = 8.0
    kinematic_heading_error: float = 12.0
    kinematic_steering: float = 2.0


class DynamicBicycleLQR:
    """Gain-scheduled, exact-ZOH four-state dynamic bicycle LQR."""

    def __init__(
        self,
        vehicle,
        weights,
        dt,
        dynamic_speed_threshold=3.0,
        recompute_speed_delta=0.20,
    ):
        self.vehicle = vehicle
        self.weights = weights
        self.dt = float(dt)
        self.dynamic_speed_threshold = float(dynamic_speed_threshold)
        self.recompute_speed_delta = max(
            0.0, float(recompute_speed_delta)
        )
        self.last_model_speed = None
        self.last_dynamic_solution = None
        self.last_kinematic_speed = None
        self.last_kinematic_solution = None

    def dynamic_continuous_matrices(self, speed):
        p = self.vehicle
        v = float(speed)
        if v <= 0.0:
            raise ValueError("dynamic model speed must be positive")
        cf = p.front_cornering_stiffness
        cr = p.rear_cornering_stiffness
        moment_delta = p.lf * cf - p.lr * cr
        moment_square = p.lf ** 2 * cf + p.lr ** 2 * cr
        a = np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [
                    0.0,
                    -(cf + cr) / (p.mass * v),
                    (cf + cr) / p.mass,
                    -moment_delta / (p.mass * v),
                ],
                [0.0, 0.0, 0.0, 1.0],
                [
                    0.0,
                    -moment_delta / (p.yaw_inertia * v),
                    moment_delta / p.yaw_inertia,
                    -moment_square / (p.yaw_inertia * v),
                ],
            ],
            dtype=float,
        )
        b = np.array(
            [
                [0.0],
                [cf / p.mass],
                [0.0],
                [p.lf * cf / p.yaw_inertia],
            ],
            dtype=float,
        )
        e = np.array(
            [
                [0.0],
                [-v * v - moment_delta / p.mass],
                [0.0],
                [-moment_square / p.yaw_inertia],
            ],
            dtype=float,
        )
        return a, b, e

    def _dynamic_solution(self, speed, curvature):
        recalculate = (
            self.last_dynamic_solution is None
            or self.last_model_speed is None
            or abs(speed - self.last_model_speed)
            >= self.recompute_speed_delta
        )
        if recalculate:
            a, b, e = self.dynamic_continuous_matrices(speed)
            ad, bd, ed = exact_zero_order_hold(
                a, b, e, dt=self.dt
            )
            q = np.diag(
                [
                    self.weights.lateral_error,
                    self.weights.lateral_error_rate,
                    self.weights.heading_error,
                    self.weights.heading_error_rate,
                ]
            )
            r = np.array([[self.weights.steering]], dtype=float)
            base = solve_lqr_base(ad, bd, q, r)
            self.last_dynamic_solution = {
                "ad": ad,
                "bd": bd,
                "ed": ed,
                "q": q,
                "r": r,
                **base,
            }
            self.last_model_speed = float(speed)
        cached = self.last_dynamic_solution
        disturbance = cached["ed"] * float(curvature)
        result = dict(cached)
        result["feedforward"] = constant_disturbance_feedforward(
            cached,
            cached["bd"],
            disturbance,
        )
        result["model"] = "dynamic"
        result["model_speed"] = self.last_model_speed
        return result

    def _kinematic_solution(self, speed, curvature):
        v = max(0.05, float(speed))
        recalculate = (
            self.last_kinematic_solution is None
            or self.last_kinematic_speed is None
            or abs(v - self.last_kinematic_speed)
            >= self.recompute_speed_delta
        )
        if recalculate:
            a = np.array(
                [[0.0, v], [0.0, 0.0]], dtype=float
            )
            b = np.array(
                [[0.0], [v / self.vehicle.wheelbase]],
                dtype=float,
            )
            e = np.array([[0.0], [-v]], dtype=float)
            ad, bd, ed = exact_zero_order_hold(
                a, b, e, dt=self.dt
            )
            q = np.diag(
                [
                    self.weights.kinematic_lateral_error,
                    self.weights.kinematic_heading_error,
                ]
            )
            r = np.array(
                [[self.weights.kinematic_steering]], dtype=float
            )
            base = solve_lqr_base(ad, bd, q, r)
            self.last_kinematic_solution = {
                "ad": ad,
                "bd": bd,
                "ed": ed,
                "q": q,
                "r": r,
                **base,
            }
            self.last_kinematic_speed = v
        cached = self.last_kinematic_solution
        result = dict(cached)
        result["feedforward"] = constant_disturbance_feedforward(
            cached,
            cached["bd"],
            cached["ed"] * float(curvature),
        )
        result["model"] = "kinematic"
        result["model_speed"] = self.last_kinematic_speed
        return result

    def control(
        self,
        lateral_error,
        lateral_error_rate,
        heading_error,
        heading_error_rate,
        speed,
        curvature,
    ):
        if abs(speed) >= self.dynamic_speed_threshold:
            state = np.array(
                [
                    [lateral_error],
                    [lateral_error_rate],
                    [heading_error],
                    [heading_error_rate],
                ],
                dtype=float,
            )
            solution = self._dynamic_solution(
                abs(speed), curvature
            )
        else:
            state = np.array(
                [[lateral_error], [heading_error]], dtype=float
            )
            solution = self._kinematic_solution(
                abs(speed), curvature
            )
        feedback = float(-(solution["k"] @ state)[0, 0])
        road_wheel_angle = solution["feedforward"] + feedback
        solution.update(
            {
                "state": state.reshape(-1),
                "feedback": feedback,
                "road_wheel_angle_raw": road_wheel_angle,
            }
        )
        return road_wheel_angle, solution


@dataclass
class PidTerms:
    error: float
    proportional: float
    integral: float
    derivative: float
    feedforward: float
    unsaturated: float
    output: float
    saturated: bool


class LongitudinalPID:
    """PID with derivative-on-measurement, LPF, and back-calculation."""

    def __init__(
        self,
        kp=1.0,
        ki=0.18,
        kd=0.08,
        derivative_cutoff_hz=4.0,
        anti_windup_gain=1.0,
        output_min=-6.0,
        output_max=3.0,
        integral_limit=10.0,
    ):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.derivative_cutoff_hz = max(
            0.0, float(derivative_cutoff_hz)
        )
        self.anti_windup_gain = max(
            0.0, float(anti_windup_gain)
        )
        self.output_min = float(output_min)
        self.output_max = float(output_max)
        self.integral_limit = abs(float(integral_limit))
        self.integral_state = 0.0
        self.previous_measurement = None
        self.filtered_measurement_rate = 0.0

    def reset(self, measurement=None):
        self.integral_state = 0.0
        self.previous_measurement = (
            None
            if measurement is None
            else finite_float(measurement)
        )
        self.filtered_measurement_rate = 0.0

    def step(self, target, measurement, dt, feedforward=0.0):
        dt = clip(float(dt), 1e-4, 0.5)
        target = float(target)
        measurement = float(measurement)
        error = target - measurement
        if self.previous_measurement is None:
            raw_measurement_rate = 0.0
        else:
            raw_measurement_rate = (
                measurement - self.previous_measurement
            ) / dt
        self.previous_measurement = measurement

        if self.derivative_cutoff_hz <= 0.0:
            alpha = 1.0
        else:
            tau = 1.0 / (
                2.0 * math.pi * self.derivative_cutoff_hz
            )
            alpha = dt / (tau + dt)
        self.filtered_measurement_rate += alpha * (
            raw_measurement_rate
            - self.filtered_measurement_rate
        )

        proportional = self.kp * error
        derivative = -self.kd * self.filtered_measurement_rate
        integral_before = self.integral_state
        integral_candidate = clip(
            integral_before + error * dt,
            -self.integral_limit,
            self.integral_limit,
        )
        unsaturated_candidate = (
            proportional
            + self.ki * integral_candidate
            + derivative
            + float(feedforward)
        )
        saturated_candidate = clip(
            unsaturated_candidate,
            self.output_min,
            self.output_max,
        )
        correction = (
            self.anti_windup_gain
            * (saturated_candidate - unsaturated_candidate)
            * dt
        )
        if abs(self.ki) > 1e-12:
            correction /= self.ki
        else:
            correction = 0.0
        self.integral_state = clip(
            integral_candidate + correction,
            -self.integral_limit,
            self.integral_limit,
        )
        integral = self.ki * self.integral_state
        unsaturated = (
            proportional + integral + derivative + feedforward
        )
        output = clip(
            unsaturated, self.output_min, self.output_max
        )
        return PidTerms(
            error=error,
            proportional=proportional,
            integral=integral,
            derivative=derivative,
            feedforward=float(feedforward),
            unsaturated=unsaturated,
            output=output,
            saturated=abs(output - unsaturated) > 1e-10,
        )


class NaturalCubicSpline1D:
    """Natural cubic spline with analytic first and second derivatives."""

    def __init__(self, station, values):
        self.station = np.asarray(station, dtype=float)
        self.values = np.asarray(values, dtype=float)
        if (
            self.station.ndim != 1
            or self.values.shape != self.station.shape
            or len(self.station) < 2
        ):
            raise ValueError("spline needs matching 1-D arrays")
        intervals = np.diff(self.station)
        if np.any(intervals <= 0.0):
            raise ValueError("spline station must be strictly increasing")
        self.second = self._solve_second_derivatives(intervals)

    def _solve_second_derivatives(self, intervals):
        count = len(self.station)
        if count <= 2:
            return np.zeros(count)
        lower = intervals[:-1].copy()
        diagonal = 2.0 * (
            intervals[:-1] + intervals[1:]
        )
        upper = intervals[1:].copy()
        rhs = 6.0 * (
            (self.values[2:] - self.values[1:-1])
            / intervals[1:]
            - (self.values[1:-1] - self.values[:-2])
            / intervals[:-1]
        )
        for index in range(1, len(diagonal)):
            factor = lower[index - 1] / diagonal[index - 1]
            diagonal[index] -= factor * upper[index - 1]
            rhs[index] -= factor * rhs[index - 1]
        interior = np.empty(count - 2)
        interior[-1] = rhs[-1] / diagonal[-1]
        for index in range(count - 4, -1, -1):
            interior[index] = (
                rhs[index] - upper[index] * interior[index + 1]
            ) / diagonal[index]
        second = np.zeros(count)
        second[1:-1] = interior
        return second

    def evaluate(self, query):
        scalar = np.isscalar(query)
        query = np.atleast_1d(np.asarray(query, dtype=float))
        query = np.clip(
            query, self.station[0], self.station[-1]
        )
        index = np.searchsorted(
            self.station, query, side="right"
        ) - 1
        index = np.clip(index, 0, len(self.station) - 2)
        left = self.station[index]
        right = self.station[index + 1]
        h = right - left
        a = (right - query) / h
        b = (query - left) / h
        value = (
            a * self.values[index]
            + b * self.values[index + 1]
            + (
                (a ** 3 - a) * self.second[index]
                + (b ** 3 - b) * self.second[index + 1]
            )
            * h ** 2
            / 6.0
        )
        first = (
            (self.values[index + 1] - self.values[index]) / h
            + h
            / 6.0
            * (
                -(3.0 * a ** 2 - 1.0) * self.second[index]
                + (3.0 * b ** 2 - 1.0)
                * self.second[index + 1]
            )
        )
        second = (
            a * self.second[index]
            + b * self.second[index + 1]
        )
        if scalar:
            return float(value[0]), float(first[0]), float(second[0])
        return value, first, second


class ReferencePath:
    """Arc-length spline, curvature, speed envelope, and progress projection."""

    def __init__(
        self,
        route,
        sample_step=0.20,
        max_speed=25.0,
        max_lateral_accel=3.0,
        max_accel=3.0,
        max_decel=6.0,
        stop_at_goal=False,
    ):
        raw_x = np.asarray(route.get("x", []), dtype=float)
        raw_y = np.asarray(route.get("y", []), dtype=float)
        if raw_x.shape != raw_y.shape or raw_x.size < 2:
            raise ValueError("global route requires matching x/y")
        finite = np.isfinite(raw_x) & np.isfinite(raw_y)
        points = np.column_stack((raw_x[finite], raw_y[finite]))
        keep = np.concatenate(
            (
                [True],
                np.linalg.norm(np.diff(points, axis=0), axis=1)
                > 1e-4,
            )
        )
        points = points[keep]
        if len(points) < 2:
            raise ValueError("global route has zero length")
        segment = np.linalg.norm(np.diff(points, axis=0), axis=1)
        raw_station = np.concatenate(([0.0], np.cumsum(segment)))
        self.spline_x = NaturalCubicSpline1D(
            raw_station, points[:, 0]
        )
        self.spline_y = NaturalCubicSpline1D(
            raw_station, points[:, 1]
        )
        self.length = float(raw_station[-1])
        sample_count = max(
            2, int(math.ceil(self.length / sample_step)) + 1
        )
        self.station = np.linspace(0.0, self.length, sample_count)
        (
            self.x,
            self.dx,
            self.ddx,
        ) = self.spline_x.evaluate(self.station)
        (
            self.y,
            self.dy,
            self.ddy,
        ) = self.spline_y.evaluate(self.station)
        tangent_norm = np.maximum(
            1e-9, np.hypot(self.dx, self.dy)
        )
        self.heading = np.unwrap(np.arctan2(self.dy, self.dx))
        self.curvature = (
            self.dx * self.ddy - self.dy * self.ddx
        ) / (tangent_norm ** 3)
        self.curvature = self._median_filter(
            self.curvature, radius=2
        )

        raw_speed = np.asarray(
            route.get("speed_limit", []), dtype=float
        )
        if raw_speed.size == raw_x.size:
            raw_speed = raw_speed[finite][keep]
            speed_cap = np.interp(
                self.station, raw_station, raw_speed
            )
        else:
            speed_cap = np.full_like(self.station, float(max_speed))
        speed_cap = np.minimum(
            np.maximum(0.0, speed_cap), float(max_speed)
        )
        curve_mask = np.abs(self.curvature) > 1e-6
        speed_cap[curve_mask] = np.minimum(
            speed_cap[curve_mask],
            np.sqrt(
                float(max_lateral_accel)
                / np.abs(self.curvature[curve_mask])
            ),
        )
        if stop_at_goal:
            speed_cap[-1] = 0.0
        ds = np.diff(self.station)
        for index in range(len(speed_cap) - 2, -1, -1):
            speed_cap[index] = min(
                speed_cap[index],
                math.sqrt(
                    max(
                        0.0,
                        speed_cap[index + 1] ** 2
                        + 2.0 * float(max_decel) * ds[index],
                    )
                ),
            )
        for index in range(1, len(speed_cap)):
            speed_cap[index] = min(
                speed_cap[index],
                math.sqrt(
                    max(
                        0.0,
                        speed_cap[index - 1] ** 2
                        + 2.0 * float(max_accel) * ds[index - 1],
                    )
                ),
            )
        self.speed = speed_cap
        self.last_station = None

    @staticmethod
    def _median_filter(values, radius):
        values = np.asarray(values, dtype=float)
        result = values.copy()
        for index in range(len(values)):
            lo = max(0, index - radius)
            hi = min(len(values), index + radius + 1)
            result[index] = np.median(values[lo:hi])
        return result

    def reset_progress(self):
        self.last_station = None

    def _candidate_indices(self):
        if self.last_station is None:
            return 0, len(self.station)
        lower_s = max(0.0, self.last_station - 4.0)
        upper_s = min(self.length, self.last_station + 100.0)
        lo = int(np.searchsorted(self.station, lower_s))
        hi = int(np.searchsorted(
            self.station, upper_s, side="right"
        ))
        return lo, max(lo + 1, hi)

    def project(self, x, y):
        lo, hi = self._candidate_indices()
        distance_sq = (
            (self.x[lo:hi] - x) ** 2
            + (self.y[lo:hi] - y) ** 2
        )
        index = lo + int(np.argmin(distance_sq))
        station = float(self.station[index])
        for _ in range(6):
            px, dx, ddx = self.spline_x.evaluate(station)
            py, dy, ddy = self.spline_y.evaluate(station)
            gradient = (px - x) * dx + (py - y) * dy
            hessian = (
                dx * dx
                + dy * dy
                + (px - x) * ddx
                + (py - y) * ddy
            )
            if abs(hessian) < 1e-10:
                break
            next_station = clip(
                station - gradient / hessian,
                max(0.0, self.station[max(0, index - 3)]),
                min(
                    self.length,
                    self.station[min(len(self.station) - 1, index + 3)],
                ),
            )
            if abs(next_station - station) < 1e-7:
                station = next_station
                break
            station = next_station
        px, dx, ddx = self.spline_x.evaluate(station)
        py, dy, ddy = self.spline_y.evaluate(station)
        heading = math.atan2(dy, dx)
        norm = max(1e-9, math.hypot(dx, dy))
        curvature = (dx * ddy - dy * ddx) / (norm ** 3)
        lateral_error = (
            -math.sin(heading) * (x - px)
            + math.cos(heading) * (y - py)
        )
        distance = math.hypot(x - px, y - py)
        if self.last_station is None or distance < 10.0:
            self.last_station = max(
                station,
                0.0
                if self.last_station is None
                else self.last_station - 1.0,
            )
        return {
            "station": station,
            "x": px,
            "y": py,
            "heading": heading,
            "curvature": curvature,
            "lateral_error": lateral_error,
            "distance": distance,
            "remaining": max(0.0, self.length - station),
            "target_speed": float(
                np.interp(station, self.station, self.speed)
            ),
        }


class LearningCsvLogger:
    FIELDNAMES = (
        "wall_time",
        "session_id",
        "sequence",
        "dt",
        "x",
        "y",
        "speed",
        "target_speed",
        "published_speed",
        "pid_error",
        "pid_p",
        "pid_i",
        "pid_d",
        "pid_ff",
        "acceleration",
        "station",
        "remaining",
        "path_distance",
        "lateral_error",
        "lateral_error_rate",
        "heading_error",
        "heading_error_rate",
        "curvature",
        "lqr_model",
        "lqr_model_speed",
        "lqr_k",
        "lqr_feedforward_deg",
        "lqr_feedback_deg",
        "road_wheel_deg",
        "steering_wheel_deg",
        "lqr_spectral_radius",
        "dare_residual",
    )

    def __init__(self, output_dir):
        output_dir = Path(output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.path = output_dir / (
            f"lqr_pid_{stamp}_{os.getpid()}.csv"
        )
        self.file = self.path.open(
            "w", newline="", encoding="utf-8"
        )
        self.writer = csv.DictWriter(
            self.file, fieldnames=self.FIELDNAMES
        )
        self.writer.writeheader()
        self.file.flush()

    def write(self, values):
        row = {name: values.get(name, "") for name in self.FIELDNAMES}
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        if not self.file.closed:
            self.file.close()


class GlobalPathLqrPidController:
    def __init__(self, args):
        self.args = args
        self.dt_nominal = 1.0 / max(1.0, args.control_hz)
        weights = LqrWeights(
            lateral_error=args.q_lateral_error,
            lateral_error_rate=args.q_lateral_error_rate,
            heading_error=args.q_heading_error,
            heading_error_rate=args.q_heading_error_rate,
            steering=args.r_steering,
            kinematic_lateral_error=args.q_lateral_error,
            kinematic_heading_error=args.q_heading_error,
            kinematic_steering=args.r_steering,
        )
        self.lqr = DynamicBicycleLQR(
            VEHICLE,
            weights,
            self.dt_nominal,
            dynamic_speed_threshold=args.dynamic_lqr_min_speed,
            recompute_speed_delta=args.lqr_speed_recompute_delta,
        )
        self.pid = LongitudinalPID(
            kp=args.speed_kp,
            ki=args.speed_ki,
            kd=args.speed_kd,
            derivative_cutoff_hz=args.speed_derivative_cutoff,
            anti_windup_gain=args.speed_anti_windup,
            output_min=-args.max_decel,
            output_max=args.max_accel,
            integral_limit=args.speed_integral_limit,
        )
        self.reference = None
        self.last_time = None
        self.last_road_wheel = 0.0
        self.published_speed = None

    def set_route(self, route, initial_speed):
        self.reference = ReferencePath(
            route,
            sample_step=self.args.path_sample_step,
            max_speed=self.args.max_speed,
            max_lateral_accel=self.args.max_lateral_accel,
            max_accel=self.args.max_accel,
            max_decel=self.args.max_decel,
            stop_at_goal=self.args.stop_at_goal,
        )
        self.pid.reset(initial_speed)
        self.last_time = None
        self.last_road_wheel = 0.0
        self.published_speed = max(0.0, float(initial_speed))

    def reset(self):
        self.reference = None
        self.pid.reset()
        self.last_time = None
        self.last_road_wheel = 0.0
        self.published_speed = None

    def control(self, ego, now=None):
        if self.reference is None:
            raise RuntimeError("controller has no global route")
        now = time.monotonic() if now is None else float(now)
        dt = (
            self.dt_nominal
            if self.last_time is None
            else clip(now - self.last_time, 0.005, 0.10)
        )
        self.last_time = now
        projection = self.reference.project(ego["x"], ego["y"])
        heading_error = wrap_angle(
            ego["heading"] - projection["heading"]
        )
        cos_h = math.cos(ego["heading"])
        sin_h = math.sin(ego["heading"])
        vx_body = cos_h * ego["vx_world"] + sin_h * ego["vy_world"]
        vy_body = -sin_h * ego["vx_world"] + cos_h * ego["vy_world"]
        model_speed = max(0.0, vx_body)
        lateral_error_rate = (
            vy_body + model_speed * math.sin(heading_error)
        )
        heading_error_rate = (
            ego["yaw_rate"]
            - model_speed * projection["curvature"]
        )
        road_wheel_raw, lqr = self.lqr.control(
            projection["lateral_error"],
            lateral_error_rate,
            heading_error,
            heading_error_rate,
            model_speed,
            projection["curvature"],
        )
        max_road_wheel = math.radians(
            self.args.max_road_wheel_deg
        )
        road_wheel = clip(
            road_wheel_raw, -max_road_wheel, max_road_wheel
        )
        max_step = math.radians(
            self.args.max_road_wheel_rate_deg
        ) * dt
        road_wheel = clip(
            road_wheel,
            self.last_road_wheel - max_step,
            self.last_road_wheel + max_step,
        )
        self.last_road_wheel = road_wheel
        steering_wheel_deg = (
            math.degrees(road_wheel)
            * self.args.steering_ratio
            * self.args.steering_sign
        )

        target_speed = projection["target_speed"]
        if (
            abs(heading_error)
            >= math.radians(self.args.large_heading_error_deg)
            or abs(projection["lateral_error"])
            >= self.args.large_lateral_error
        ):
            target_speed = min(
                target_speed, self.args.recovery_speed
            )
        if abs(heading_error) >= math.radians(75.0):
            target_speed = min(target_speed, 0.5)
        if (
            self.args.stop_at_goal
            and projection["remaining"] <= self.args.goal_tolerance
        ):
            target_speed = 0.0
        drag_acceleration = (
            VEHICLE.rolling_resistance * VEHICLE.gravity
            + 0.5
            * VEHICLE.air_density
            * VEHICLE.drag_coefficient
            * VEHICLE.frontal_area
            * ego["speed"] ** 2
            / VEHICLE.mass
        )
        pid = self.pid.step(
            target_speed,
            ego["speed"],
            dt,
            feedforward=drag_acceleration
            if target_speed > 0.1
            else 0.0,
        )
        if self.published_speed is None:
            self.published_speed = ego["speed"]
        desired_step = target_speed - self.published_speed
        published_step = clip(
            desired_step,
            -self.args.max_decel * dt,
            self.args.max_accel * dt,
        )
        self.published_speed = max(
            0.0, self.published_speed + published_step
        )
        return {
            "acceleration": pid.output,
            "speed": self.published_speed,
            "steering_wheel_deg": steering_wheel_deg,
            "road_wheel": road_wheel,
            "projection": projection,
            "pid": pid,
            "lqr": lqr,
            "dt": dt,
            "lateral_error_rate": lateral_error_rate,
            "heading_error": heading_error,
            "heading_error_rate": heading_error_rate,
            "target_speed": target_speed,
        }


SDK = {}


def load_multicast_sdk():
    if SDK:
        return SDK
    # run_hmxzw already contains the proven onsite protobuf-root discovery:
    # it searches beside this project, the current working directory, and
    # the native libMulticastNetwork module. Reuse those exact imported
    # classes/constants so this entrypoint cannot drift to another SDK copy.
    communication = importlib.import_module("run_hmxzw")
    SDK.update(
        {
            "network": communication.libMulticastNetwork,
            "VEHICLE_CONTROL": communication.VEHICLE_CONTROL,
            "VEHICLE_FEEDBACK": communication.VEHICLE_FEEDBACK,
            "VehicleControl": communication.VehicleControl,
            "VehicleFeedback": communication.VehicleFeedback,
            "ActorPrepare": communication.ActorPrepare,
            "ActorPrepareResult": communication.ActorPrepareResult,
            "Notify": communication.Notify,
            "get_ip_address": communication.get_ip_address,
            "MT_ACTOR_PREPARE": communication.MT_ACTOR_PREPARE,
            "MT_ACTOR_PREPARE_RESULT": (
                communication.MT_ACTOR_PREPARE_RESULT
            ),
            "MT_NOTIFY": communication.MT_NOTIFY,
            "NT_ABORT_TEST": communication.NT_ABORT_TEST,
            "NT_COLLIDE_ROLE": communication.NT_COLLIDE_ROLE,
            "NT_DESTROY_ROLE": communication.NT_DESTROY_ROLE,
            "NT_FINISH_TEST": communication.NT_FINISH_TEST,
            "NT_START_TEST": communication.NT_START_TEST,
        }
    )
    return SDK


def setup_global_route_planner(cache_dir):
    current = SCRIPT_DIR
    planner_root = None
    while True:
        candidate = current / "ros2_map" / "src" / "gloplan"
        if (
            candidate
            / "gloplan"
            / "global_route_planner.py"
        ).is_file():
            planner_root = candidate
            break
        if current.parent == current:
            break
        current = current.parent
    if planner_root is None:
        raise ImportError(
            "cannot locate ros2_map/src/gloplan pure-Python planner"
        )
    if str(planner_root) not in sys.path:
        sys.path.insert(0, str(planner_root))
    from opendrive_spiral_compat import install_spiral_support

    install_spiral_support()
    module = importlib.import_module(
        "gloplan.global_route_planner"
    )
    planner = module.DirectGlobalRoutePlanner(
        map_search_dirs=[str(SCRIPT_DIR / "maps")],
        persistent_cache_dir=str(cache_dir),
    )
    planner.PLANNER_CACHE_VERSION = (
        "direct-global-route-v2-spiral"
    )
    return planner


def session_order_key(value):
    import re

    match = re.search(
        r"_(\d{14})_(\d+)$", str(value or "").strip()
    )
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


class GlobalLqrPidRuntime:
    def __init__(self, args):
        self.args = args
        self.sdk = load_multicast_sdk()
        self.actor_id = str(args.actor_id)
        self.role_id = self.actor_id
        self.session_id = ""
        self.prepared = False
        self.started = False
        self.last_ins_sequence = None
        self.ego = None
        self.last_heading = None
        self.last_heading_time = None
        self.last_control_time = 0.0
        self.last_debug_time = 0.0
        self.test_started_monotonic = None
        self.last_ins_monotonic = None
        self.control_period = 1.0 / max(1.0, args.control_hz)
        self.controller = GlobalPathLqrPidController(args)
        self.route_planner = setup_global_route_planner(
            Path(args.route_cache_dir)
        )
        self.logger = LearningCsvLogger(args.log_dir)
        self.prepare_channel = None
        self.notify_channel = None
        self.ins_channel = None
        self.control_channel = None

    def create_channels(self):
        network = self.sdk["network"]
        param = network.CreateChannelsParam()
        param.config_center_addr = self.args.config_center
        param.local_ip = self.sdk["get_ip_address"](
            self.args.net_interface
        )
        param.net_interface_name = self.args.net_interface
        param.field_id = self.args.field_id
        param.log_level = 1
        param.client_name = self.actor_id
        param.recv_self_msg = False
        print(
            "[lqr-pid][network] "
            f"interface={param.net_interface_name} "
            f"local_ip={param.local_ip} "
            f"field={param.field_id}",
            flush=True,
        )
        channels = network.ChannelPtrVector()
        result = network.create_channels(param, channels)
        if result:
            raise RuntimeError(
                f"create channels failed, ret={result}"
            )
        channel_map = {channel.name(): channel for channel in channels}
        required = ("prepare", "notify", "ins", "vehiclecontrol")
        missing = [
            name for name in required if name not in channel_map
        ]
        if missing:
            raise RuntimeError(
                f"required channels missing: {missing}; "
                f"available={sorted(channel_map)}"
            )
        self.prepare_channel = channel_map["prepare"]
        self.notify_channel = channel_map["notify"]
        self.ins_channel = channel_map["ins"]
        self.control_channel = channel_map["vehiclecontrol"]
        print(
            "[lqr-pid] communication ready: "
            "prepare/notify/ins/vehiclecontrol",
            flush=True,
        )

    def send_control(self, acceleration, speed, steering_wheel_deg):
        message = self.sdk["VehicleControl"]()
        message.acceleration = float(acceleration)
        message.speed = float(speed)
        message.steering_control.target_steering_wheel_angle = (
            float(steering_wheel_deg)
        )
        payload = message.SerializeToString()
        result = self.control_channel.put(
            self.sdk["VEHICLE_CONTROL"], len(payload), payload
        )
        if result != 0:
            print(
                f"[lqr-pid][control][ERROR] put ret={result}",
                flush=True,
            )
        return result == 0

    def send_prepare_result(self, accepted):
        message = self.sdk["ActorPrepareResult"]()
        message.session_id = self.session_id
        message.actor_id = self.actor_id
        message.result = bool(accepted)
        payload = message.SerializeToString()
        result = self.prepare_channel.put(
            self.sdk["MT_ACTOR_PREPARE_RESULT"],
            len(payload),
            payload,
        )
        print(
            "[lqr-pid][prepare] result sent "
            f"session={self.session_id} accepted={int(accepted)} "
            f"ret={result}",
            flush=True,
        )
        return result == 0

    def reset_episode(self, keep_session=True):
        session = self.session_id
        self.prepared = False
        self.started = False
        self.role_id = self.actor_id
        self.last_ins_sequence = None
        self.ego = None
        self.last_heading = None
        self.last_heading_time = None
        self.test_started_monotonic = None
        self.last_ins_monotonic = None
        self.controller.reset()
        self.session_id = session if keep_session else ""

    def _build_route(self, brief_data):
        testees = brief_data.get("testees") or []
        if not testees:
            raise ValueError("brief_data has no testees")
        testee = testees[0]
        initial = testee["init_state"]
        target = testee["target_state"]
        map_ref = (
            brief_data.get("zjl_odv_file")
            or brief_data.get("map_name")
            or brief_data.get("map_id")
        )
        route = self.route_planner.plan(
            start_state=initial,
            goal_state=target,
            map_ref=map_ref,
        )
        return testee, initial, target, map_ref, route

    def handle_prepare(self, prepare):
        incoming = str(prepare.session_id or "").strip()
        if not incoming:
            print("[lqr-pid][prepare][WARN] empty session")
            return
        if incoming == self.session_id and self.prepared:
            return
        incoming_key = session_order_key(incoming)
        current_key = session_order_key(self.session_id)
        if (
            self.session_id
            and incoming_key is not None
            and current_key is not None
            and incoming_key <= current_key
        ):
            print(
                "[lqr-pid][prepare][WARN] stale session "
                f"received={incoming} current={self.session_id}",
                flush=True,
            )
            return
        self.reset_episode(keep_session=False)
        self.session_id = incoming
        try:
            brief_data = json.loads(
                prepare.archive_info.brief_data
            )
            (
                testee,
                initial,
                target,
                map_ref,
                route,
            ) = self._build_route(brief_data)
            self.role_id = str(
                testee.get("role_id") or self.actor_id
            )
            initial_speed = max(
                0.0, finite_float(initial.get("v", 0.0))
            )
            self.controller.set_route(route, initial_speed)
            print(
                "[lqr-pid][prepare] route ready "
                f"session={self.session_id} map={map_ref} "
                f"points={len(route['x'])} "
                f"length={self.controller.reference.length:.2f}m "
                f"initial_speed={initial_speed:.2f}m/s "
                f"goal=({float(target['x']):.2f},"
                f"{float(target['y']):.2f})",
                flush=True,
            )
            # 平台要求：准备结果之前先发布第一帧控制。使用初始速度，
            # 不制造目标速度跳变，也不提前转向。
            first_control_ok = self.send_control(
                0.0, initial_speed, 0.0
            )
            if not first_control_ok:
                raise RuntimeError(
                    "first neutral VehicleControl publication failed"
                )
            self.prepared = True
            self.send_prepare_result(True)
        except Exception as exc:
            print(
                "[lqr-pid][prepare][ERROR] "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            self.prepared = False
            self.send_prepare_result(False)

    def poll_prepare(self):
        for _ in range(16):
            result, message = self.prepare_channel.get()
            if message is None:
                return
            if (
                result < 0
                or message.type()
                != self.sdk["MT_ACTOR_PREPARE"]
            ):
                continue
            payload = self.sdk["network"].getMessageData(message)
            decoded = self.sdk["ActorPrepare"]()
            decoded.ParseFromString(payload)
            self.handle_prepare(decoded)

    def handle_notify(self, notify):
        incoming = str(notify.session_id or "").strip()
        session_sensitive = notify.type in (
            self.sdk["NT_START_TEST"],
            self.sdk["NT_FINISH_TEST"],
            self.sdk["NT_ABORT_TEST"],
        )
        if session_sensitive and self.session_id:
            if not incoming:
                print(
                    "[lqr-pid][notify][WARN] empty-session event "
                    f"ignored type={notify.type}",
                    flush=True,
                )
                return
            if incoming != self.session_id:
                print(
                    "[lqr-pid][notify][WARN] stale event ignored "
                    f"received={incoming} current={self.session_id}",
                    flush=True,
                )
                return
        if notify.type == self.sdk["NT_START_TEST"]:
            if self.prepared:
                self.started = True
                self.test_started_monotonic = time.monotonic()
                self.last_ins_monotonic = None
                print(
                    f"[lqr-pid][notify] START {self.session_id}",
                    flush=True,
                )
            return
        if notify.type in (
            self.sdk["NT_FINISH_TEST"],
            self.sdk["NT_ABORT_TEST"],
            self.sdk["NT_DESTROY_ROLE"],
        ):
            event = {
                self.sdk["NT_FINISH_TEST"]: "FINISH",
                self.sdk["NT_ABORT_TEST"]: "ABORT",
                self.sdk["NT_DESTROY_ROLE"]: "DESTROY",
            }[notify.type]
            print(
                f"[lqr-pid][notify] {event} {self.session_id}",
                flush=True,
            )
            self.send_control(0.0, 0.0, 0.0)
            self.reset_episode(keep_session=True)

    def poll_notify(self):
        for _ in range(32):
            result, message = self.notify_channel.get()
            if message is None:
                return
            if (
                result < 0
                or message.type() != self.sdk["MT_NOTIFY"]
            ):
                continue
            payload = self.sdk["network"].getMessageData(message)
            decoded = self.sdk["Notify"]()
            decoded.ParseFromString(payload)
            self.handle_notify(decoded)

    def poll_ins(self):
        ins = self.ins_channel.get_ins()
        sequence = int(getattr(ins, "sequence_num", 0))
        if sequence <= 1 or sequence > 1_000_000:
            return
        if sequence == self.last_ins_sequence:
            return
        now = time.monotonic()
        heading = finite_float(ins.heading)
        yaw_rate = finite_float(
            getattr(
                getattr(ins, "angular_velocity", None),
                "z",
                float("nan"),
            ),
            default=float("nan"),
        )
        if not math.isfinite(yaw_rate):
            if (
                self.last_heading is None
                or self.last_heading_time is None
            ):
                yaw_rate = 0.0
            else:
                yaw_rate = wrap_angle(
                    heading - self.last_heading
                ) / max(1e-3, now - self.last_heading_time)
        self.last_heading = heading
        self.last_heading_time = now
        vx_world = finite_float(ins.linear_velocity.x)
        vy_world = finite_float(ins.linear_velocity.y)
        values = (
            finite_float(ins.position.x, float("nan")),
            finite_float(ins.position.y, float("nan")),
            heading,
            vx_world,
            vy_world,
            yaw_rate,
        )
        if not all(math.isfinite(value) for value in values):
            return
        self.last_ins_sequence = sequence
        self.last_ins_monotonic = now
        self.ego = {
            "sequence": sequence,
            "x": values[0],
            "y": values[1],
            "heading": heading,
            "vx_world": vx_world,
            "vy_world": vy_world,
            "yaw_rate": yaw_rate,
            "speed": math.hypot(vx_world, vy_world),
        }

    def poll_feedback(self):
        for _ in range(8):
            result, message = self.control_channel.get()
            if message is None:
                return
            if (
                result < 0
                or message.type()
                != self.sdk["VEHICLE_FEEDBACK"]
            ):
                continue

    def publish_control(self):
        now = time.monotonic()
        if now - self.last_control_time < self.control_period:
            return
        self.last_control_time = now
        if not self.started:
            return
        if self.ego is None or self.controller.reference is None:
            self.send_control(0.0, 0.0, 0.0)
            return
        output = self.controller.control(self.ego, now)
        self.send_control(
            output["acceleration"],
            output["speed"],
            output["steering_wheel_deg"],
        )
        projection = output["projection"]
        pid = output["pid"]
        lqr = output["lqr"]
        row = {
            "wall_time": time.time(),
            "session_id": self.session_id,
            "sequence": self.ego["sequence"],
            "dt": output["dt"],
            "x": self.ego["x"],
            "y": self.ego["y"],
            "speed": self.ego["speed"],
            "target_speed": output["target_speed"],
            "published_speed": output["speed"],
            "pid_error": pid.error,
            "pid_p": pid.proportional,
            "pid_i": pid.integral,
            "pid_d": pid.derivative,
            "pid_ff": pid.feedforward,
            "acceleration": output["acceleration"],
            "station": projection["station"],
            "remaining": projection["remaining"],
            "path_distance": projection["distance"],
            "lateral_error": projection["lateral_error"],
            "lateral_error_rate": output["lateral_error_rate"],
            "heading_error": output["heading_error"],
            "heading_error_rate": output["heading_error_rate"],
            "curvature": projection["curvature"],
            "lqr_model": lqr["model"],
            "lqr_model_speed": lqr["model_speed"],
            "lqr_k": json.dumps(
                lqr["k"].reshape(-1).tolist()
            ),
            "lqr_feedforward_deg": math.degrees(
                lqr["feedforward"]
            ),
            "lqr_feedback_deg": math.degrees(lqr["feedback"]),
            "road_wheel_deg": math.degrees(
                output["road_wheel"]
            ),
            "steering_wheel_deg": output[
                "steering_wheel_deg"
            ],
            "lqr_spectral_radius": lqr["spectral_radius"],
            "dare_residual": lqr["dare_residual"],
        }
        self.logger.write(row)
        if now - self.last_debug_time >= self.args.debug_period:
            self.last_debug_time = now
            print(
                "[lqr-pid][control] "
                f"s={projection['station']:.1f}/"
                f"{self.controller.reference.length:.1f}m "
                f"v={self.ego['speed']:.2f}->"
                f"{output['target_speed']:.2f}m/s "
                f"ey={projection['lateral_error']:.3f}m "
                f"epsi={math.degrees(output['heading_error']):.2f}deg "
                f"kappa={projection['curvature']:.4f} "
                f"model={lqr['model']} "
                f"delta_ff={math.degrees(lqr['feedforward']):.2f}deg "
                f"delta_fb={math.degrees(lqr['feedback']):.2f}deg "
                f"steer={output['steering_wheel_deg']:.1f}deg "
                f"PID=[{pid.proportional:.2f},"
                f"{pid.integral:.2f},{pid.derivative:.2f}] "
                f"acc={pid.output:.2f}",
                flush=True,
            )

    def enforce_watchdogs(self):
        if (
            not self.started
            or self.test_started_monotonic is None
        ):
            return
        now = time.monotonic()
        elapsed = now - self.test_started_monotonic
        waiting_first = self.last_ins_monotonic is None
        silence = now - (
            self.test_started_monotonic
            if waiting_first
            else self.last_ins_monotonic
        )
        ins_limit = (
            self.args.first_ins_timeout
            if waiting_first
            else self.args.ins_stall_timeout
        )
        if ins_limit > 0.0 and silence >= ins_limit:
            print(
                "[lqr-pid][WATCHDOG] "
                f"{'first INS timeout' if waiting_first else 'INS stalled'} "
                f"silence={silence:.2f}s limit={ins_limit:.2f}s",
                flush=True,
            )
            self.send_control(0.0, 0.0, 0.0)
            os._exit(INS_STALL_EXIT_CODE)
        if (
            self.args.task_timeout > 0.0
            and elapsed >= self.args.task_timeout
        ):
            print(
                "[lqr-pid][WATCHDOG] task timeout "
                f"elapsed={elapsed:.2f}s "
                f"limit={self.args.task_timeout:.2f}s",
                flush=True,
            )
            self.send_control(0.0, 0.0, 0.0)
            os._exit(TASK_TIMEOUT_EXIT_CODE)

    def run(self):
        self.create_channels()
        print(
            "[lqr-pid] controller ready "
            f"vehicle={asdict(VEHICLE)} "
            f"csv={self.logger.path}",
            flush=True,
        )
        try:
            while True:
                self.poll_prepare()
                self.poll_notify()
                self.poll_ins()
                self.poll_feedback()
                self.publish_control()
                self.enforce_watchdogs()
                time.sleep(0.002)
        finally:
            self.logger.close()


def run_self_test():
    vehicle = VEHICLE
    weights = LqrWeights()
    controller = DynamicBicycleLQR(
        vehicle, weights, dt=0.02
    )
    a, b, e = controller.dynamic_continuous_matrices(20.0)
    ad, bd, ed = exact_zero_order_hold(a, b, e, dt=0.02)
    if np.linalg.norm(ad - np.eye(4)) <= 0.0:
        raise AssertionError("discretization produced identity")
    straight = affine_lqr(
        ad,
        bd,
        ed * 0.0,
        np.diag([8.0, 1.2, 12.0, 1.5]),
        np.array([[2.0]]),
    )
    if straight["spectral_radius"] >= 1.0:
        raise AssertionError("dynamic closed loop is unstable")
    left = affine_lqr(
        ad,
        bd,
        ed * 0.02,
        np.diag([8.0, 1.2, 12.0, 1.5]),
        np.array([[2.0]]),
    )
    right = affine_lqr(
        ad,
        bd,
        ed * -0.02,
        np.diag([8.0, 1.2, 12.0, 1.5]),
        np.array([[2.0]]),
    )
    if not (
        left["feedforward"] > 0.0
        and right["feedforward"] < 0.0
    ):
        raise AssertionError("curvature feedforward sign is wrong")
    if abs(left["feedforward"] + right["feedforward"]) > 1e-9:
        raise AssertionError("affine feedforward is not symmetric")
    state = np.array([[0.5], [0.0], [0.1], [0.0]])
    for _ in range(1000):
        road_angle, solution = controller.control(
            state[0, 0],
            state[1, 0],
            state[2, 0],
            state[3, 0],
            20.0,
            0.02,
        )
        state = (
            ad @ state + bd * road_angle + ed * 0.02
        )
    if abs(state[0, 0]) > 1e-5:
        raise AssertionError(
            "constant-curvature lateral error did not converge"
        )

    pid = LongitudinalPID(
        kp=1.0, ki=0.2, kd=0.1, output_min=-2.0, output_max=2.0
    )
    for _ in range(500):
        terms = pid.step(30.0, 0.0, 0.02)
    if not (-2.0 <= terms.output <= 2.0):
        raise AssertionError("PID saturation failed")
    if abs(pid.integral_state) > pid.integral_limit + 1e-9:
        raise AssertionError("PID anti-windup failed")

    route = {
        "x": np.linspace(0.0, 100.0, 101).tolist(),
        "y": np.zeros(101).tolist(),
        "speed_limit": np.full(101, 25.0).tolist(),
    }
    reference = ReferencePath(route)
    projection = reference.project(10.0, 1.0)
    if abs(projection["lateral_error"] - 1.0) > 1e-3:
        raise AssertionError("path projection sign/value failed")
    if abs(projection["curvature"]) > 1e-8:
        raise AssertionError("straight path curvature failed")
    if abs(reference.speed[-1] - 25.0) > 1e-9:
        raise AssertionError(
            "pass-through route must not force terminal speed to zero"
        )
    angle = np.linspace(-math.pi / 2.0, math.pi / 2.0, 101)
    radius = 8.0
    uturn = ReferencePath(
        {
            "x": (radius * np.cos(angle)).tolist(),
            "y": (radius * np.sin(angle)).tolist(),
            "speed_limit": np.full(101, 25.0).tolist(),
        }
    )
    center = len(uturn.station) // 2
    if abs(uturn.curvature[center] - 1.0 / radius) > 2e-3:
        raise AssertionError("U-turn spline curvature failed")
    if uturn.speed[center] > math.sqrt(3.0 * radius) + 1e-3:
        raise AssertionError("U-turn lateral acceleration cap failed")

    print("[self-test] vehicle parameters:", asdict(vehicle))
    print(
        "[self-test] PASS "
        f"DARE_residual={straight['dare_residual']:.3e} "
        f"spectral_radius={straight['spectral_radius']:.6f} "
        f"left_ff={math.degrees(left['feedforward']):.3f}deg"
    )


def supervise_runtime(args):
    """Use run_hmxzw's read-only DriverSim monitor for this controller."""
    monitor = importlib.import_module("run_hmxzw")
    supervisor_lock = monitor.acquire_supervisor_lock()
    if supervisor_lock is False:
        return 2
    child_argv = [
        sys.executable,
        str(Path(__file__).resolve()),
        *sys.argv[1:],
        "--runtime-child",
    ]
    restart_count = 0
    start_failures = 0
    simulator_launcher = None
    runtime = None
    while True:
        try:
            if not args.no_manage_simulator:
                driver_pids = monitor._find_driver_sim_pids(
                    args.simulator_dir
                )
                if driver_pids:
                    simulator_launcher = None
                    print(
                        "[lqr-pid][supervisor] attach DriverSim "
                        f"pids={driver_pids}",
                        flush=True,
                    )
                else:
                    simulator_launcher = (
                        monitor._start_managed_simulator(args)
                    )
                    driver_pids = monitor._wait_for_driver_sim(
                        args, simulator_launcher
                    )
                if not driver_pids:
                    start_failures += 1
                    monitor._archive_simulator_diagnostics(
                        args,
                        "lqr_pid_startup_process_missing",
                        simulator_launcher,
                    )
                    monitor._terminate_managed_process(
                        simulator_launcher,
                        "DriverSim launcher",
                    )
                    monitor._kill_simulator_residuals(
                        args.simulator_dir,
                        "lqr_pid_startup_process_missing",
                    )
                    failure_limit = int(
                        args.max_simulator_start_failures
                    )
                    if (
                        failure_limit > 0
                        and start_failures >= failure_limit
                    ):
                        print(
                            "[lqr-pid][supervisor][FATAL] "
                            f"DriverSim startup failed "
                            f"{start_failures} times",
                            flush=True,
                        )
                        return 1
                    time.sleep(max(0.0, args.restart_delay))
                    continue
                start_failures = 0

            print(
                "[lqr-pid][supervisor] start controller "
                f"attempt={restart_count + 1}",
                flush=True,
            )
            runtime = subprocess.Popen(
                child_argv, start_new_session=True
            )
            restart_reason = None
            return_code = None
            while restart_reason is None:
                return_code = runtime.poll()
                if return_code is not None:
                    if return_code == TASK_TIMEOUT_EXIT_CODE:
                        restart_reason = "lqr_pid_task_timeout"
                    elif return_code == INS_STALL_EXIT_CODE:
                        restart_reason = "lqr_pid_ins_stall"
                    else:
                        break
                if (
                    not args.no_manage_simulator
                    and not monitor._find_driver_sim_pids(
                        args.simulator_dir
                    )
                ):
                    restart_reason = (
                        "lqr_pid_simulator_process_missing"
                    )
                if restart_reason is None:
                    time.sleep(0.2)
        except KeyboardInterrupt:
            print(
                "[lqr-pid][supervisor] interrupted; cleanup",
                flush=True,
            )
            monitor._terminate_managed_process(
                runtime, "LQR PID runtime"
            )
            monitor._terminate_managed_process(
                simulator_launcher, "DriverSim launcher"
            )
            if not args.no_manage_simulator:
                monitor._kill_simulator_residuals(
                    args.simulator_dir,
                    "lqr_pid_supervisor_interrupted",
                )
            return 130

        if restart_reason is None:
            monitor._terminate_managed_process(
                simulator_launcher, "DriverSim launcher"
            )
            if not args.no_manage_simulator:
                monitor._kill_simulator_residuals(
                    args.simulator_dir,
                    "lqr_pid_runtime_exit",
                )
            print(
                "[lqr-pid][supervisor] controller exited "
                f"code={return_code}",
                flush=True,
            )
            return int(return_code or 0)

        if not args.no_manage_simulator:
            if "simulator_process_missing" in restart_reason:
                time.sleep(1.0)
            monitor._archive_simulator_diagnostics(
                args, restart_reason, simulator_launcher
            )
        monitor._terminate_managed_process(
            runtime, "LQR PID runtime"
        )
        monitor._terminate_managed_process(
            simulator_launcher, "DriverSim launcher"
        )
        if not args.no_manage_simulator:
            monitor._kill_simulator_residuals(
                args.simulator_dir, restart_reason
            )
        runtime = None
        simulator_launcher = None
        restart_count += 1
        print(
            "[lqr-pid][supervisor] restart complete stack "
            f"reason={restart_reason} count={restart_count}",
            flush=True,
        )
        time.sleep(max(0.0, args.restart_delay))


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Global XODR path tracking with longitudinal PID and "
            "full-theory dynamic/kinematic LQR"
        )
    )
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--runtime-child",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--config_center", default="47.110.233.70:52009"
    )
    parser.add_argument(
        "--field_id",
        default="field-zd-test1-22-0331134113-888",
    )
    parser.add_argument("--net_interface", default="usb0")
    parser.add_argument("--actor-id", default="apollo_testee")
    parser.add_argument("--control-hz", type=float, default=50.0)
    parser.add_argument("--max-speed", type=float, default=25.0)
    parser.add_argument("--max-accel", type=float, default=3.0)
    parser.add_argument("--max-decel", type=float, default=6.0)
    parser.add_argument(
        "--max-lateral-accel", type=float, default=3.0
    )
    parser.add_argument(
        "--goal-tolerance", type=float, default=1.5
    )
    parser.add_argument(
        "--stop-at-goal",
        action="store_true",
        help=(
            "brake to zero at the route endpoint; disabled by default "
            "so the vehicle passes through the goal"
        ),
    )
    parser.add_argument(
        "--recovery-speed", type=float, default=4.0
    )
    parser.add_argument(
        "--large-heading-error-deg",
        type=float,
        default=25.0,
    )
    parser.add_argument(
        "--large-lateral-error", type=float, default=1.5
    )
    parser.add_argument(
        "--path-sample-step", type=float, default=0.20
    )

    parser.add_argument("--speed-kp", type=float, default=1.0)
    parser.add_argument("--speed-ki", type=float, default=0.18)
    parser.add_argument("--speed-kd", type=float, default=0.08)
    parser.add_argument(
        "--speed-derivative-cutoff", type=float, default=4.0
    )
    parser.add_argument(
        "--speed-anti-windup", type=float, default=1.0
    )
    parser.add_argument(
        "--speed-integral-limit", type=float, default=10.0
    )

    parser.add_argument(
        "--q-lateral-error", type=float, default=8.0
    )
    parser.add_argument(
        "--q-lateral-error-rate", type=float, default=1.2
    )
    parser.add_argument(
        "--q-heading-error", type=float, default=12.0
    )
    parser.add_argument(
        "--q-heading-error-rate", type=float, default=1.5
    )
    parser.add_argument("--r-steering", type=float, default=2.0)
    parser.add_argument(
        "--dynamic-lqr-min-speed", type=float, default=3.0
    )
    parser.add_argument(
        "--lqr-speed-recompute-delta", type=float, default=0.20
    )

    parser.add_argument(
        "--steering-ratio",
        type=float,
        default=16.0,
        help=(
            "steering-wheel / road-wheel ratio; not provided by "
            "the parameter images and must be calibrated"
        ),
    )
    parser.add_argument(
        "--steering-sign", type=float, default=1.0
    )
    parser.add_argument(
        "--max-road-wheel-deg", type=float, default=35.0
    )
    parser.add_argument(
        "--max-road-wheel-rate-deg",
        type=float,
        default=120.0,
    )
    parser.add_argument(
        "--route-cache-dir",
        default=str(SCRIPT_DIR / "route_cache"),
    )
    parser.add_argument(
        "--log-dir",
        default=str(SCRIPT_DIR / "debug_logs" / "lqr_pid"),
    )
    parser.add_argument(
        "--debug-period", type=float, default=0.5
    )
    parser.add_argument(
        "--task-timeout", type=float, default=20.0
    )
    parser.add_argument(
        "--first-ins-timeout", type=float, default=15.0
    )
    parser.add_argument(
        "--ins-stall-timeout", type=float, default=4.0
    )
    parser.add_argument(
        "--restart-delay", type=float, default=3.0
    )
    parser.add_argument(
        "--simulator-dir",
        default=(
            "/media/pc/FanXiang2T/Onsite_FirstWithForth/"
            "LinuxNoEditor416"
        ),
    )
    parser.add_argument(
        "--simulator-ready-delay", type=float, default=2.0
    )
    parser.add_argument(
        "--simulator-start-timeout", type=float, default=30.0
    )
    parser.add_argument(
        "--max-simulator-start-failures",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--simulator-log-dir",
        default=str(SCRIPT_DIR / "debug_logs" / "simulator"),
    )
    parser.add_argument(
        "--no-manage-simulator", action="store_true"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="accepted for consistency; learning logs are always enabled",
    )
    return parser.parse_args()


def validate_args(args):
    positive = {
        "control_hz": args.control_hz,
        "max_speed": args.max_speed,
        "max_accel": args.max_accel,
        "max_decel": args.max_decel,
        "max_lateral_accel": args.max_lateral_accel,
        "path_sample_step": args.path_sample_step,
        "steering_ratio": args.steering_ratio,
        "max_road_wheel_deg": args.max_road_wheel_deg,
        "max_road_wheel_rate_deg": args.max_road_wheel_rate_deg,
        "r_steering": args.r_steering,
    }
    invalid = [
        name for name, value in positive.items()
        if not math.isfinite(value) or value <= 0.0
    ]
    q_values = (
        args.q_lateral_error,
        args.q_lateral_error_rate,
        args.q_heading_error,
        args.q_heading_error_rate,
    )
    if invalid:
        raise ValueError(
            f"arguments must be positive: {invalid}"
        )
    if any(
        not math.isfinite(value) or value < 0.0
        for value in q_values
    ) or not any(value > 0.0 for value in q_values):
        raise ValueError(
            "LQR Q weights must be nonnegative and not all zero"
        )


def main():
    args = parse_args()
    validate_args(args)
    if args.self_test:
        run_self_test()
        return
    if not args.runtime_child:
        raise SystemExit(supervise_runtime(args))
    runtime = GlobalLqrPidRuntime(args)
    try:
        runtime.run()
    except KeyboardInterrupt:
        print("[lqr-pid] stopped by user", flush=True)


if __name__ == "__main__":
    main()
