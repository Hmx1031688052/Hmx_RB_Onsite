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

LQR 输出是前轮转角，DriverSim 的控制字段名为方向盘转角，因此发送前仍需
经过执行器映射。仓库已有 INS/横摆响应标定表明 Cam6 的有效比例约为 1.65，
且正命令产生正横摆，所以默认 ``steering_ratio=1.65``、
``steering_sign=+1``；它不是普通乘用车约 16:1 的机械方向盘比例。

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
import multiprocessing
import os
import queue
import signal
import subprocess
import sys
import time
import types
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

from speed_limits import (
    initial_state_speed_mps,
    resolve_expected_speed,
)


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


# Competition-oriented presets.  Every value can still be overridden from
# the command line, so one executable can be used for repeatable parameter
# sweeps without editing source code between runs.
SCORE_PROFILES = {
    "comfort": {
        "max_speed": 40.0,
        "max_accel": 2.50,
        "max_decel": 2.50,
        "max_lateral_accel": 0.42,
        "max_longitudinal_jerk": 4.50,
        "max_lateral_jerk": 0.80,
        "max_yaw_rate": 0.42,
        "speed_limit_ratio": 1.00,
        "recovery_speed": 4.0,
        "speed_kp": 0.95,
        "speed_ki": 0.12,
        "speed_kd": 0.10,
        "max_road_wheel_rate_deg": 35.0,
        "max_lqr_feedback_road_wheel_deg": 8.0,
        "normal_feedback_speed_product_deg_mps": 500.0,
        "high_speed_damping_blend": 0.0,
        "recovery_entry_corridor_ratio": 0.80,
        "max_recovery_feedback_road_wheel_deg": 10.0,
        "curvature_preview_time": 0.30,
        "geometric_feedforward_blend": 0.0,
    },
    "balanced": {
        "max_speed": 45.0,
        "max_accel": 2.70,
        "max_decel": 2.60,
        "max_lateral_accel": 0.43,
        "max_longitudinal_jerk": 4.80,
        "max_lateral_jerk": 0.80,
        "max_yaw_rate": 0.43,
        "speed_limit_ratio": 1.15,
        "recovery_speed": 5.0,
        "speed_kp": 1.05,
        "speed_ki": 0.12,
        "speed_kd": 0.10,
        "max_road_wheel_rate_deg": 50.0,
        "max_lqr_feedback_road_wheel_deg": 8.0,
        "normal_feedback_speed_product_deg_mps": 500.0,
        "high_speed_damping_blend": 0.0,
        "recovery_entry_corridor_ratio": 0.80,
        "max_recovery_feedback_road_wheel_deg": 10.0,
        "curvature_preview_time": 0.35,
        "geometric_feedforward_blend": 0.0,
    },
    "efficiency": {
        "max_speed": 55.0,
        "max_accel": 2.85,
        "max_decel": 2.75,
        "max_lateral_accel": 0.46,
        "max_longitudinal_jerk": 5.40,
        "max_lateral_jerk": 0.88,
        "max_yaw_rate": 0.46,
        "speed_limit_ratio": 1.17,
        "recovery_speed": 6.0,
        "speed_kp": 1.25,
        "speed_ki": 0.10,
        "speed_kd": 0.06,
        "max_road_wheel_rate_deg": 60.0,
        "max_lqr_feedback_road_wheel_deg": 7.0,
        "normal_feedback_speed_product_deg_mps": 500.0,
        "high_speed_damping_blend": 0.0,
        "recovery_entry_corridor_ratio": 0.80,
        "max_recovery_feedback_road_wheel_deg": 10.0,
        "curvature_preview_time": 0.40,
        "geometric_feedforward_blend": 0.0,
    },
    "attack": {
        "max_speed": 60.0,
        "max_accel": 2.85,
        "max_decel": 2.85,
        "max_lateral_accel": 3.50,
        "max_longitudinal_jerk": 5.50,
        "max_lateral_jerk": 8.00,
        "max_yaw_rate": 0.85,
        "speed_limit_ratio": 1.17,
        "recovery_speed": 8.0,
        "speed_kp": 1.25,
        "speed_ki": 0.10,
        "speed_kd": 0.06,
        "max_road_wheel_rate_deg": 90.0,
        "max_lqr_feedback_road_wheel_deg": 2.5,
        "normal_feedback_speed_product_deg_mps": 15.0,
        "high_speed_damping_blend": 1.0,
        "recovery_entry_corridor_ratio": 0.90,
        "max_recovery_feedback_road_wheel_deg": 10.0,
        "curvature_preview_time": 0.45,
        "geometric_feedforward_blend": 1.0,
    },
}

COMFORT_THRESHOLDS = {
    "longitudinal_accel": 3.0,
    "longitudinal_jerk": 6.0,
    "lateral_accel": 0.5,
    "lateral_jerk": 1.0,
    "yaw_rate": 0.5,
}


def road_wheel_curvature(road_wheel_angle, wheelbase):
    """Return kinematic curvature for a road-wheel angle in radians."""
    return math.tan(float(road_wheel_angle)) / max(
        1e-6, float(wheelbase)
    )


def comfort_curvature_limit(speed, max_lateral_accel, max_yaw_rate):
    """Maximum curvature allowed by lateral acceleration and yaw rate."""
    speed = abs(float(speed))
    if speed <= 0.10:
        return float("inf")
    lateral_limit = float(max_lateral_accel) / max(
        speed * speed, 1e-6
    )
    yaw_limit = float(max_yaw_rate) / max(speed, 1e-6)
    return max(0.0, min(lateral_limit, yaw_limit))


def comfort_road_wheel_rate_limit(
    speed,
    longitudinal_accel,
    road_wheel_angle,
    wheelbase,
    max_lateral_jerk,
):
    """Road-wheel rate that keeps ``d(v^2*kappa)/dt`` in budget.

    The bound covers both longitudinal acceleration in a bend and steering
    transients.  At walking speed the score is insensitive to steering-rate
    induced lateral jerk, so callers may retain their ordinary actuator cap.
    """
    speed = abs(float(speed))
    if speed <= 0.50:
        return float("inf")
    wheelbase = max(1e-6, float(wheelbase))
    angle = float(road_wheel_angle)
    curvature = road_wheel_curvature(angle, wheelbase)
    acceleration_term = abs(
        2.0 * speed * float(longitudinal_accel) * curvature
    )
    remaining = max(
        0.0, float(max_lateral_jerk) - acceleration_term
    )
    return (
        remaining
        * wheelbase
        * math.cos(angle) ** 2
        / max(speed * speed, 1e-6)
    )


def resolve_score_profile(args):
    """Fill only unspecified tuning values from the selected preset."""
    defaults = SCORE_PROFILES[args.score_profile]
    for name, value in defaults.items():
        if getattr(args, name, None) is None:
            setattr(args, name, value)
    return args


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
        v = max(0.10, float(speed))
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
        force_kinematic=False,
    ):
        if (
            abs(speed) >= self.dynamic_speed_threshold
            and not force_kinematic
        ):
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
        speed_limit_ratio=1.0,
        max_yaw_rate=0.5,
        max_lateral_jerk=1.0,
        max_longitudinal_jerk=6.0,
        legal_speed_mps=None,
        expected_speed_mps=None,
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

        # ``DirectGlobalRoutePlanner.speed_limit`` is a geometry/design-speed
        # profile.  It contains conservative curve and lane-change caps and
        # must not be used as either the legal limit or the evaluator's
        # expected speed.  Those two values come from ActorPrepare/XODR via
        # ``resolve_expected_speed`` and remain separate from geometry.
        resolved_legal_speed = finite_float(
            legal_speed_mps,
            float(max_speed) / max(1.0, float(speed_limit_ratio)),
        )
        resolved_expected_speed = finite_float(
            expected_speed_mps, resolved_legal_speed
        )
        self.legal_speed_mps = max(0.1, resolved_legal_speed)
        self.expected_speed_mps = max(
            0.1, resolved_expected_speed
        )
        self.legal_speed = np.full_like(
            self.station, self.legal_speed_mps
        )
        speed_cap = np.minimum(
            self.legal_speed * float(speed_limit_ratio),
            float(max_speed),
        )

        # Three independent curve caps correspond directly to the three
        # lateral comfort indicators used by the competition.
        abs_curvature = np.abs(self.curvature)
        curve_mask = abs_curvature > 1e-6
        speed_cap[curve_mask] = np.minimum(
            speed_cap[curve_mask],
            np.sqrt(
                float(max_lateral_accel)
                / abs_curvature[curve_mask]
            ),
        )
        speed_cap[curve_mask] = np.minimum(
            speed_cap[curve_mask],
            float(max_yaw_rate) / abs_curvature[curve_mask],
        )
        self.curvature_gradient = np.gradient(
            self.curvature, self.station, edge_order=1
        )
        gradient_mask = np.abs(self.curvature_gradient) > 1e-8
        speed_cap[gradient_mask] = np.minimum(
            speed_cap[gradient_mask],
            np.cbrt(
                float(max_lateral_jerk)
                / np.abs(self.curvature_gradient[gradient_mask])
            ),
        )
        if stop_at_goal:
            speed_cap[-1] = 0.0
        ds = np.diff(self.station)

        # Reserve distance for the command-jerk ramp instead of assuming that
        # full acceleration/deceleration appears instantaneously.
        jerk = max(0.1, float(max_longitudinal_jerk))
        accel_ramp_time = float(max_accel) / jerk
        decel_ramp_time = float(max_decel) / jerk
        profile_accel = float(max_accel) / (
            1.0 + 0.5 * accel_ramp_time
        )
        profile_decel = float(max_decel) / (
            1.0 + 0.75 * decel_ramp_time
        )

        def propagate_reachable_limits():
            for index in range(len(speed_cap) - 2, -1, -1):
                speed_cap[index] = min(
                    speed_cap[index],
                    math.sqrt(
                        max(
                            0.0,
                            speed_cap[index + 1] ** 2
                            + 2.0 * profile_decel * ds[index],
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
                            + 2.0 * profile_accel * ds[index - 1],
                        )
                    ),
                )

        propagate_reachable_limits()
        # Include the usually omitted 2*v*a*kappa term in lateral jerk.
        # Three fixed-point passes are enough because every pass only lowers
        # the envelope and is followed by longitudinal reachability.
        for _ in range(3):
            longitudinal_accel_profile = 0.5 * np.gradient(
                speed_cap * speed_cap,
                self.station,
                edge_order=1,
            )
            for index in range(len(speed_cap)):
                curvature = abs(float(self.curvature[index]))
                curvature_rate = abs(
                    float(self.curvature_gradient[index])
                )
                acceleration = abs(
                    float(longitudinal_accel_profile[index])
                )
                if (
                    curvature <= 1e-9
                    and curvature_rate <= 1e-9
                ):
                    continue
                upper = float(speed_cap[index])
                if (
                    2.0 * upper * acceleration * curvature
                    + upper ** 3 * curvature_rate
                    <= float(max_lateral_jerk)
                ):
                    continue
                lower = 0.0
                for _iteration in range(36):
                    candidate = 0.5 * (lower + upper)
                    lateral_jerk = (
                        2.0
                        * candidate
                        * acceleration
                        * curvature
                        + candidate ** 3 * curvature_rate
                    )
                    if lateral_jerk <= float(max_lateral_jerk):
                        lower = candidate
                    else:
                        upper = candidate
                speed_cap[index] = lower
            propagate_reachable_limits()
        self.speed = speed_cap
        self.expected_time = self.length / self.expected_speed_mps
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

    def preview_curvature(self, station, distance):
        preview_station = clip(
            float(station) + max(0.0, float(distance)),
            0.0,
            self.length,
        )
        return (
            float(
                np.interp(
                    preview_station,
                    self.station,
                    self.curvature,
                )
            ),
            preview_station,
        )

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
            "curvature_gradient": float(
                np.interp(station, self.station, self.curvature_gradient)
            ),
            "legal_speed": float(
                np.interp(station, self.station, self.legal_speed)
            ),
            "lateral_error": lateral_error,
            "distance": distance,
            "remaining": max(0.0, self.length - station),
            "target_speed": float(
                np.interp(station, self.station, self.speed)
            ),
        }


class ScoreProxyTracker:
    """Online proxy using wall time and actual ego travel distance."""

    def __init__(self):
        self.reset(0.0, 0.0)

    def reset(
        self,
        expected_speed,
        route_expected_time,
        lane_half_width=1.55,
        lane_margin=0.15,
    ):
        self.expected_speed = max(0.1, float(expected_speed))
        self.route_expected_time = max(
            0.0, float(route_expected_time)
        )
        self.lane_half_width = max(0.5, float(lane_half_width))
        self.lane_margin = max(0.0, float(lane_margin))
        self.started_monotonic = None
        self.last_sample_monotonic = None
        self.elapsed = 0.0
        self.distance_travelled = 0.0
        self.last_xy = None
        self.last_longitudinal_speed = None
        self.last_longitudinal_accel = None
        self.last_lateral_accel = None
        self.max_speed_ratio = 0.0
        self.route_violation_time = 0.0
        self.wrong_way_time = 0.0
        self.route_violation_detected = False
        self.wrong_way_detected = False
        self.violation_time = {
            name: 0.0 for name in COMFORT_THRESHOLDS
        }
        self.latest = {
            "measured_longitudinal_accel": 0.0,
            "measured_longitudinal_jerk": 0.0,
            "estimated_lateral_accel": 0.0,
            "estimated_lateral_jerk": 0.0,
            "measured_yaw_rate": 0.0,
            "speed_limit_ratio_actual": 0.0,
            "comfort_score_proxy": 10.0,
            "rule_score_proxy": 10.0,
        }

    def start(self, now=None):
        now = time.monotonic() if now is None else float(now)
        self.started_monotonic = now
        self.last_sample_monotonic = None
        self.elapsed = 0.0

    def update(self, ego, projection, now=None):
        now = time.monotonic() if now is None else float(now)
        if self.started_monotonic is None:
            self.start(now)
        if self.last_sample_monotonic is None:
            dt = 0.0
        else:
            dt = clip(
                now - self.last_sample_monotonic, 1e-4, 0.5
            )
        self.last_sample_monotonic = now
        self.elapsed = max(
            0.0, now - self.started_monotonic
        )

        speed = max(0.0, finite_float(ego.get("speed", 0.0)))
        longitudinal_speed = finite_float(
            ego.get("longitudinal_speed", speed), speed
        )
        yaw_rate = finite_float(ego.get("yaw_rate", 0.0))
        legal_speed = max(0.1, finite_float(
            projection.get("legal_speed", 0.1), 0.1
        ))
        speed_ratio = speed / legal_speed
        self.max_speed_ratio = max(self.max_speed_ratio, speed_ratio)

        longitudinal_accel = 0.0
        longitudinal_jerk = 0.0
        if self.last_longitudinal_speed is not None and dt > 0.0:
            longitudinal_accel = (
                longitudinal_speed - self.last_longitudinal_speed
            ) / dt
            if self.last_longitudinal_accel is not None:
                longitudinal_jerk = (
                    longitudinal_accel - self.last_longitudinal_accel
                ) / dt
        lateral_accel = longitudinal_speed * yaw_rate
        lateral_jerk = 0.0
        if self.last_lateral_accel is not None and dt > 0.0:
            lateral_jerk = (
                lateral_accel - self.last_lateral_accel
            ) / dt

        x = finite_float(ego.get("x", float("nan")), float("nan"))
        y = finite_float(ego.get("y", float("nan")), float("nan"))
        if math.isfinite(x) and math.isfinite(y):
            if self.last_xy is not None:
                step = math.hypot(
                    x - self.last_xy[0], y - self.last_xy[1]
                )
                # Reject stale-session jumps without discarding legitimate
                # high-speed travel.
                if step <= max(5.0, 2.5 * speed * max(dt, 0.02)):
                    self.distance_travelled += step
            self.last_xy = (x, y)

        values = {
            "longitudinal_accel": longitudinal_accel,
            "longitudinal_jerk": longitudinal_jerk,
            "lateral_accel": lateral_accel,
            "lateral_jerk": lateral_jerk,
            "yaw_rate": yaw_rate,
        }
        # Do not count derivative artefacts on the first real sample.
        if self.last_longitudinal_speed is not None and dt > 0.0:
            for name, value in values.items():
                if abs(value) > COMFORT_THRESHOLDS[name]:
                    self.violation_time[name] += dt

        lateral_error = abs(
            finite_float(projection.get("lateral_error", 0.0))
        )
        heading_error = abs(
            wrap_angle(
                finite_float(ego.get("heading", 0.0))
                - finite_float(projection.get("heading", 0.0))
            )
        )
        if dt > 0.0:
            if lateral_error > (
                self.lane_half_width - self.lane_margin
            ):
                self.route_violation_time += dt
            else:
                self.route_violation_time = max(
                    0.0, self.route_violation_time - 0.5 * dt
                )
            if (
                longitudinal_speed < -0.20
                or (
                    speed > 0.50
                    and heading_error > math.radians(100.0)
                )
            ):
                self.wrong_way_time += dt
            else:
                self.wrong_way_time = max(
                    0.0, self.wrong_way_time - 0.5 * dt
                )
        self.route_violation_detected = (
            self.route_violation_detected
            or self.route_violation_time >= 0.20
        )
        self.wrong_way_detected = (
            self.wrong_way_detected
            or self.wrong_way_time >= 0.20
        )

        comfort_deduction = sum(
            2.0 * duration / max(self.elapsed, 1e-6)
            for duration in self.violation_time.values()
        )
        comfort_score = max(0.0, 10.0 - comfort_deduction)
        if self.max_speed_ratio > 1.50:
            rule_deduction = 5.0
        elif self.max_speed_ratio > 1.20:
            rule_deduction = 2.5
        else:
            rule_deduction = 0.0
        if self.route_violation_detected:
            rule_deduction += 2.5
        if self.wrong_way_detected:
            rule_deduction += 2.5
        rule_score = max(0.0, 10.0 - rule_deduction)

        self.latest = {
            "measured_longitudinal_accel": longitudinal_accel,
            "measured_longitudinal_jerk": longitudinal_jerk,
            "estimated_lateral_accel": lateral_accel,
            "estimated_lateral_jerk": lateral_jerk,
            "measured_yaw_rate": yaw_rate,
            "speed_limit_ratio_actual": speed_ratio,
            "comfort_score_proxy": comfort_score,
            "rule_score_proxy": rule_score,
        }
        self.last_longitudinal_speed = longitudinal_speed
        self.last_longitudinal_accel = longitudinal_accel
        self.last_lateral_accel = lateral_accel
        return dict(self.latest)

    def summary(self, completed, now=None):
        if self.started_monotonic is not None:
            now = time.monotonic() if now is None else float(now)
            self.elapsed = max(
                self.elapsed, now - self.started_monotonic
            )
        elapsed = max(self.elapsed, 1e-6)
        expected_time = (
            self.distance_travelled / self.expected_speed
        )
        efficiency_score = (
            min(100.0, 50.0 + 50.0 * expected_time / elapsed)
            if completed
            else min(50.0, 50.0 * expected_time / elapsed)
        )
        comfort_score = self.latest["comfort_score_proxy"]
        rule_score = self.latest["rule_score_proxy"]
        # Full-score proxy assumes the ignored safety and coordination
        # dimensions remain at 10 points.
        full_score_proxy = math.sqrt(max(0.0, efficiency_score)) * (
            5.0 + 1.0 + 0.3 * comfort_score + 0.1 * rule_score
        )
        return {
            "completed": bool(completed),
            "elapsed": elapsed,
            "distance_travelled": self.distance_travelled,
            "expected_speed": self.expected_speed,
            "expected_time": expected_time,
            "route_expected_time": self.route_expected_time,
            "efficiency_score_proxy": efficiency_score,
            "comfort_score_proxy": comfort_score,
            "rule_score_proxy": rule_score,
            "full_score_proxy": full_score_proxy,
            "max_speed_ratio": self.max_speed_ratio,
            "route_violation_detected": (
                self.route_violation_detected
            ),
            "wrong_way_detected": self.wrong_way_detected,
            "rule_proxy_coverage": [
                "overspeed",
                "routed_lane_corridor",
                "wrong_way_or_reverse",
            ],
            "rule_inputs_not_available": [
                "traffic_light_phase",
                "other_vehicle_yielding",
                "other_vehicle_lane_change_effect",
            ],
            "violation_time": dict(self.violation_time),
        }


class LearningCsvLogger:
    FIELDNAMES = (
        "wall_time",
        "session_id",
        "sequence",
        "dt",
        "x",
        "y",
        "score_profile",
        "experiment_tag",
        "speed",
        "expected_speed",
        "legal_speed",
        "path_target_speed",
        "target_speed",
        "published_speed",
        "feedback_speed_guard",
        "requested_curvature_speed_cap",
        "speed_limit_ratio_actual",
        "pid_error",
        "pid_p",
        "pid_i",
        "pid_d",
        "pid_ff",
        "acceleration",
        "command_jerk",
        "measured_longitudinal_accel",
        "measured_longitudinal_jerk",
        "estimated_lateral_accel",
        "estimated_lateral_jerk",
        "measured_yaw_rate",
        "comfort_score_proxy",
        "rule_score_proxy",
        "station",
        "remaining",
        "path_distance",
        "lateral_error",
        "lateral_error_rate",
        "heading_error",
        "heading_error_rate",
        "curvature",
        "control_curvature",
        "curvature_preview_station",
        "lqr_model",
        "recovery_mode",
        "lateral_feedback_controller",
        "lateral_feedback_limit_deg",
        "lqr_model_speed",
        "lqr_k",
        "lqr_feedforward_deg",
        "lqr_model_feedforward_deg",
        "lqr_geometric_feedforward_deg",
        "lqr_feedback_deg",
        "lqr_feedback_unclipped_deg",
        "lateral_damping_blend",
        "road_wheel_deg",
        "road_wheel_rate_deg",
        "comfort_road_wheel_rate_limit_deg",
        "predicted_command_lateral_accel",
        "predicted_command_lateral_jerk",
        "longitudinal_accel_lateral_jerk_limit",
        "steering_wheel_deg",
        "lqr_spectral_radius",
        "dare_residual",
    )

    def __init__(self, output_dir, args=None):
        output_dir = Path(output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        raw_tag = "" if args is None else str(args.experiment_tag or "")
        safe_tag = "".join(
            char if (char.isalnum() or char in "-_") else "_"
            for char in raw_tag
        ).strip("_")
        tag_suffix = f"_{safe_tag}" if safe_tag else ""
        self.path = output_dir / (
            f"lqr_pid_{stamp}_{os.getpid()}{tag_suffix}.csv"
        )
        self.file = self.path.open(
            "w", newline="", encoding="utf-8"
        )
        self.writer = csv.DictWriter(
            self.file, fieldnames=self.FIELDNAMES
        )
        self.writer.writeheader()
        self.file.flush()
        self.profile_path = self.path.with_suffix(".profile.json")
        if args is not None:
            selected = {
                name: getattr(args, name)
                for name in (
                    "score_profile", "experiment_tag", "control_hz",
                    "max_speed", "speed_limit_ratio", "max_accel",
                    "max_decel", "max_longitudinal_jerk",
                    "max_lateral_accel", "max_lateral_jerk",
                    "max_yaw_rate", "speed_kp", "speed_ki", "speed_kd",
                    "max_road_wheel_rate_deg", "q_lateral_error",
                    "q_heading_error", "r_steering",
                    "steering_ratio", "steering_sign",
                    "max_road_wheel_deg",
                    "max_recovery_road_wheel_deg",
                    "max_lqr_feedback_road_wheel_deg",
                    "normal_feedback_speed_product_deg_mps",
                    "high_speed_damping_blend",
                    "high_speed_damping_gain",
                    "high_speed_damping_min_speed",
                    "high_speed_damping_full_speed",
                    "recovery_entry_corridor_ratio",
                    "normal_feedback_full_curvature",
                    "max_recovery_feedback_road_wheel_deg",
                    "recovery_feedback_speed_product_deg_mps",
                    "recovery_stanley_gain",
                    "recovery_stanley_softening_speed",
                    "recovery_exit_lateral_error",
                    "recovery_exit_heading_error_deg",
                    "curvature_preview_time",
                    "geometric_feedforward_blend",
                    "feedback_speed_guard_lateral_error",
                    "feedback_speed_guard_heading_error_deg",
                    "expected_speed_mps", "use_xodr_expected_speed",
                    "rule_lane_half_width", "rule_lane_margin",
                    "no_route_visualizer", "route_visualizer_width",
                    "route_visualizer_height", "route_visualizer_hz",
                    "route_visualizer_max_points",
                    "task_timeout",
                )
            }
            self.profile_path.write_text(
                json.dumps(selected, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

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
            kinematic_lateral_error=(
                args.q_kinematic_lateral_error
            ),
            kinematic_heading_error=(
                args.q_kinematic_heading_error
            ),
            kinematic_steering=args.r_kinematic_steering,
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
        self.recovery_active = False
        self.published_speed = None
        self.last_acceleration_command = 0.0
        self.score_tracker = ScoreProxyTracker()
        print(
            "[lqr-pid][steering-map] "
            "command_deg=road_wheel_deg"
            f"*{float(args.steering_ratio):.3f}"
            f"*{float(args.steering_sign):+.0f}",
            flush=True,
        )

    def set_route(
        self,
        route,
        initial_speed,
        legal_speed,
        expected_speed,
    ):
        self.reference = ReferencePath(
            route,
            sample_step=self.args.path_sample_step,
            max_speed=self.args.max_speed,
            max_lateral_accel=self.args.max_lateral_accel,
            max_accel=self.args.max_accel,
            max_decel=self.args.max_decel,
            stop_at_goal=self.args.stop_at_goal,
            speed_limit_ratio=self.args.speed_limit_ratio,
            max_yaw_rate=self.args.max_yaw_rate,
            max_lateral_jerk=self.args.max_lateral_jerk,
            max_longitudinal_jerk=(
                self.args.max_longitudinal_jerk
            ),
            legal_speed_mps=legal_speed,
            expected_speed_mps=expected_speed,
        )
        self.pid.reset(initial_speed)
        self.last_time = None
        self.last_road_wheel = 0.0
        self.recovery_active = False
        self.published_speed = max(0.0, float(initial_speed))
        self.last_acceleration_command = 0.0
        self.score_tracker.reset(
            self.reference.expected_speed_mps,
            self.reference.expected_time,
            lane_half_width=self.args.rule_lane_half_width,
            lane_margin=self.args.rule_lane_margin,
        )

    def reset(self):
        self.reference = None
        self.pid.reset()
        self.last_time = None
        self.last_road_wheel = 0.0
        self.recovery_active = False
        self.published_speed = None
        self.last_acceleration_command = 0.0
        self.score_tracker.reset(0.1, 0.0)

    def start_scoring(self, now=None):
        self.score_tracker.start(now)

    def synchronize_speed(self, actual_speed):
        """Synchronize command states to the first real INS measurement."""
        actual_speed = max(0.0, finite_float(actual_speed))
        self.pid.reset(actual_speed)
        self.published_speed = actual_speed
        self.last_acceleration_command = 0.0
        self.last_time = None

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
        ego["longitudinal_speed"] = vx_body
        lateral_error_rate = (
            vy_body + model_speed * math.sin(heading_error)
        )
        heading_error_rate = (
            ego["yaw_rate"]
            - model_speed * projection["curvature"]
        )
        routed_corridor = max(
            0.30,
            self.args.rule_lane_half_width
            - self.args.rule_lane_margin,
        )
        recovery_lateral_threshold = min(
            self.args.large_lateral_error,
            self.args.recovery_entry_corridor_ratio
            * routed_corridor,
        )
        recovery_entry = (
            abs(heading_error)
            >= math.radians(self.args.large_heading_error_deg)
            or abs(projection["lateral_error"])
            >= recovery_lateral_threshold
        )
        if self.recovery_active:
            recovery_exit = (
                abs(projection["lateral_error"])
                <= self.args.recovery_exit_lateral_error
                and abs(heading_error)
                <= math.radians(
                    self.args.recovery_exit_heading_error_deg
                )
            )
            if recovery_exit:
                self.recovery_active = False
        elif recovery_entry:
            self.recovery_active = True
        recovery_mode = self.recovery_active
        curvature_preview_distance = (
            model_speed * self.args.curvature_preview_time
        )
        control_curvature, curvature_preview_station = (
            self.reference.preview_curvature(
                projection["station"],
                curvature_preview_distance,
            )
        )
        if (
            recovery_mode
            or (
                projection["curvature"] * control_curvature < 0.0
                and abs(projection["curvature"]) > 0.005
            )
        ):
            control_curvature = projection["curvature"]
            curvature_preview_station = projection["station"]
        road_wheel_raw, lqr = self.lqr.control(
            projection["lateral_error"],
            lateral_error_rate,
            heading_error,
            heading_error_rate,
            model_speed,
            control_curvature,
            force_kinematic=recovery_mode,
        )
        model_feedforward = float(lqr["feedforward"])
        geometric_feedforward = math.atan(
            VEHICLE.wheelbase * control_curvature
        )
        feedforward_blend = clip(
            self.args.geometric_feedforward_blend, 0.0, 1.0
        )
        lqr["model_feedforward"] = model_feedforward
        lqr["geometric_feedforward"] = geometric_feedforward
        lqr["feedforward"] = (
            (1.0 - feedforward_blend) * model_feedforward
            + feedforward_blend * geometric_feedforward
        )
        feedback_controller = "lqr"
        lqr_feedback_raw = float(lqr["feedback"])
        straight_feedback_limit_deg = min(
            self.args.max_lqr_feedback_road_wheel_deg,
            (
                self.args.normal_feedback_speed_product_deg_mps
                / max(model_speed, 1.0)
            ),
        )
        curve_authority_blend = clip(
            abs(control_curvature)
            / self.args.normal_feedback_full_curvature,
            0.0,
            1.0,
        )
        feedback_limit_deg = (
            straight_feedback_limit_deg
            + curve_authority_blend
            * (
                self.args.max_lqr_feedback_road_wheel_deg
                - straight_feedback_limit_deg
            )
        )
        speed_damping_blend = clip(
            (
                model_speed
                - self.args.high_speed_damping_min_speed
            )
            / (
                self.args.high_speed_damping_full_speed
                - self.args.high_speed_damping_min_speed
            ),
            0.0,
            1.0,
        )
        damping_blend = (
            self.args.high_speed_damping_blend
            * speed_damping_blend
            * (1.0 - curve_authority_blend)
        )
        damped_feedback = -self.args.high_speed_damping_gain * (
            0.00261 * projection["lateral_error"]
            + 0.00182 * lateral_error_rate
            + 0.10599 * heading_error
            + 0.01139 * heading_error_rate
        )
        provisional_limit = math.radians(feedback_limit_deg)
        lqr_feedback_unclipped = (
            (1.0 - damping_blend)
            * clip(
                lqr_feedback_raw,
                -provisional_limit,
                provisional_limit,
            )
            + damping_blend * damped_feedback
        )
        if damping_blend > 0.0:
            feedback_controller = "pole_damped_lqr"
        if recovery_mode:
            # Raw kinematic LQR is intentionally aggressive outside its
            # linear region.  In the high-speed straight test it requested
            # +/-60 deg, crossed the route with substantial yaw, then did
            # the same in the opposite direction.  A Stanley-style recovery
            # law supplies the missing geometric damping, while the
            # speed-scaled cap preserves authority at low speed without
            # commanding a hairpin correction at motorway speed.
            feedback_controller = "stanley_recovery"
            damping_blend = 0.0
            stanley_cross_track = math.atan2(
                self.args.recovery_stanley_gain
                * projection["lateral_error"],
                model_speed
                + self.args.recovery_stanley_softening_speed,
            )
            lqr_feedback_unclipped = -(
                heading_error + stanley_cross_track
            )
            feedback_limit_deg = min(
                self.args.max_recovery_feedback_road_wheel_deg,
                max(
                    self.args.max_lqr_feedback_road_wheel_deg,
                    (
                        self.args
                        .recovery_feedback_speed_product_deg_mps
                        / max(model_speed, 1.0)
                    ),
                ),
            )
        feedback_limit = math.radians(feedback_limit_deg)
        lqr["feedback"] = clip(
            lqr_feedback_unclipped,
            -feedback_limit,
            feedback_limit,
        )
        lqr["feedback_controller"] = feedback_controller
        lqr["feedback_limit"] = feedback_limit
        lqr["damping_blend"] = damping_blend
        road_wheel_raw = (
            float(lqr["feedforward"]) + float(lqr["feedback"])
        )
        lqr["road_wheel_angle_raw"] = road_wheel_raw
        lqr["feedback_unclipped"] = lqr_feedback_unclipped

        path_target_speed = projection["target_speed"]
        target_speed = path_target_speed
        if recovery_mode:
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

        physical_max_road_wheel = math.radians(
            self.args.max_recovery_road_wheel_deg
            if recovery_mode
            else self.args.max_road_wheel_deg
        )
        physical_requested_road_wheel = clip(
            road_wheel_raw,
            -physical_max_road_wheel,
            physical_max_road_wheel,
        )
        requested_curvature = abs(
            road_wheel_curvature(
                physical_requested_road_wheel,
                VEHICLE.wheelbase,
            )
        )
        feedback_speed_guard = (
            recovery_mode
            or abs(projection["lateral_error"])
            >= self.args.feedback_speed_guard_lateral_error
            or abs(heading_error)
            >= math.radians(
                self.args.feedback_speed_guard_heading_error_deg
            )
        )
        requested_curvature_speed_cap = float("inf")
        if requested_curvature > 1e-8:
            requested_curvature_speed_cap = min(
                math.sqrt(
                    self.args.max_lateral_accel / requested_curvature
                ),
                self.args.max_yaw_rate / requested_curvature,
            )
            # The route profile already limits reference curvature, while the
            # actual road-wheel command below is independently clipped to the
            # current-speed comfort envelope.  Applying this cap to every raw
            # LQR correction made centimetre-scale errors reduce a 10 m/s
            # straight to roughly 3 m/s.  Reserve it for material tracking
            # errors and recovery, where slowing down genuinely restores
            # steering authority.
            if feedback_speed_guard:
                target_speed = min(
                    target_speed,
                    requested_curvature_speed_cap,
                )

        max_road_wheel = physical_max_road_wheel
        comfort_kappa_limit = comfort_curvature_limit(
            model_speed,
            self.args.max_lateral_accel,
            self.args.max_yaw_rate,
        )
        if math.isfinite(comfort_kappa_limit):
            max_road_wheel = min(
                max_road_wheel,
                math.atan(
                    VEHICLE.wheelbase * comfort_kappa_limit
                ),
            )
        requested_road_wheel = clip(
            physical_requested_road_wheel,
            -max_road_wheel,
            max_road_wheel,
        )

        # A large feedback correction implies curvature beyond the reference
        # envelope.  Couple speed to the actual requested road-wheel angle so
        # recovery cannot silently violate lateral acceleration/yaw limits.
        actuator_rate_limit = math.radians(
            self.args.max_road_wheel_rate_deg
        )
        comfort_rate_limit = comfort_road_wheel_rate_limit(
            model_speed,
            self.last_acceleration_command,
            self.last_road_wheel,
            VEHICLE.wheelbase,
            self.args.max_lateral_jerk,
        )
        road_wheel_rate_limit = min(
            actuator_rate_limit, comfort_rate_limit
        )
        max_step = road_wheel_rate_limit * dt
        previous_road_wheel = self.last_road_wheel
        road_wheel = clip(
            requested_road_wheel,
            previous_road_wheel - max_step,
            previous_road_wheel + max_step,
        )
        self.last_road_wheel = road_wheel
        road_wheel_rate = (
            road_wheel - previous_road_wheel
        ) / max(dt, 1e-6)
        steering_wheel_deg = (
            math.degrees(road_wheel)
            * self.args.steering_ratio
            * self.args.steering_sign
        )

        commanded_curvature = road_wheel_curvature(
            road_wheel, VEHICLE.wheelbase
        )
        curvature_rate = (
            math.cos(road_wheel) ** -2
            * road_wheel_rate
            / VEHICLE.wheelbase
        )
        steering_lateral_jerk = abs(
            model_speed * model_speed * curvature_rate
        )

        # Feed back measured lateral/yaw exposure.  This is intentionally a
        # speed reduction rather than an extra steering discontinuity.
        measured_yaw_rate = abs(
            finite_float(ego.get("yaw_rate", 0.0))
        )
        measured_lateral_accel = (
            abs(model_speed * measured_yaw_rate)
        )
        if measured_yaw_rate > self.args.max_yaw_rate:
            target_speed = min(
                target_speed,
                max(
                    0.5,
                    model_speed
                    * self.args.max_yaw_rate
                    / max(measured_yaw_rate, 1e-6),
                ),
            )
        if measured_lateral_accel > self.args.max_lateral_accel:
            target_speed = min(
                target_speed,
                max(
                    0.5,
                    model_speed
                    * math.sqrt(
                        self.args.max_lateral_accel
                        / max(measured_lateral_accel, 1e-6)
                    ),
                ),
            )

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
        raw_acceleration = clip(
            pid.output, -self.args.max_decel, self.args.max_accel
        )
        longitudinal_accel_lateral_jerk_limit = float("inf")
        if (
            model_speed > 0.50
            and abs(commanded_curvature) > 1e-8
        ):
            remaining_lateral_jerk = max(
                0.0,
                self.args.max_lateral_jerk
                - steering_lateral_jerk,
            )
            longitudinal_accel_lateral_jerk_limit = (
                remaining_lateral_jerk
                / (
                    2.0
                    * model_speed
                    * abs(commanded_curvature)
                )
            )
            raw_acceleration = clip(
                raw_acceleration,
                -longitudinal_accel_lateral_jerk_limit,
                longitudinal_accel_lateral_jerk_limit,
            )
        max_acceleration_step = (
            self.args.max_longitudinal_jerk * dt
        )
        previous_acceleration_command = self.last_acceleration_command
        acceleration = clip(
            raw_acceleration,
            previous_acceleration_command - max_acceleration_step,
            previous_acceleration_command + max_acceleration_step,
        )
        if math.isfinite(
            longitudinal_accel_lateral_jerk_limit
        ):
            # If the two jerk constraints temporarily have no intersection,
            # prefer ending the lateral exposure immediately; the resulting
            # longitudinal-jerk event lasts one control interval instead of
            # keeping lateral jerk above threshold through the bend.
            acceleration = clip(
                acceleration,
                -longitudinal_accel_lateral_jerk_limit,
                longitudinal_accel_lateral_jerk_limit,
            )
        command_jerk = (
            acceleration - previous_acceleration_command
        ) / max(dt, 1e-6)
        self.last_acceleration_command = acceleration
        previous_published_speed = self.published_speed
        candidate_speed = max(
            0.0, previous_published_speed + acceleration * dt
        )
        if target_speed >= previous_published_speed:
            self.published_speed = min(target_speed, candidate_speed)
        else:
            self.published_speed = max(target_speed, candidate_speed)
        predicted_command_lateral_accel = abs(
            model_speed * model_speed * commanded_curvature
        )
        predicted_command_lateral_jerk = (
            steering_lateral_jerk
            + abs(
                2.0
                * model_speed
                * acceleration
                * commanded_curvature
            )
        )
        score_metrics = self.score_tracker.update(
            ego, projection, now
        )
        return {
            "acceleration": acceleration,
            "command_jerk": command_jerk,
            "raw_acceleration": raw_acceleration,
            "score_metrics": score_metrics,
            "speed": self.published_speed,
            "path_target_speed": path_target_speed,
            "steering_wheel_deg": steering_wheel_deg,
            "road_wheel": road_wheel,
            "road_wheel_rate": road_wheel_rate,
            "comfort_road_wheel_rate_limit": (
                comfort_rate_limit
            ),
            "predicted_command_lateral_accel": (
                predicted_command_lateral_accel
            ),
            "predicted_command_lateral_jerk": (
                predicted_command_lateral_jerk
            ),
            "longitudinal_accel_lateral_jerk_limit": (
                longitudinal_accel_lateral_jerk_limit
            ),
            "projection": projection,
            "pid": pid,
            "lqr": lqr,
            "recovery_mode": recovery_mode,
            "lateral_feedback_controller": feedback_controller,
            "lateral_feedback_limit": feedback_limit,
            "dt": dt,
            "lateral_error_rate": lateral_error_rate,
            "heading_error": heading_error,
            "heading_error_rate": heading_error_rate,
            "target_speed": target_speed,
            "feedback_speed_guard": feedback_speed_guard,
            "requested_curvature_speed_cap": (
                requested_curvature_speed_cap
            ),
            "control_curvature": control_curvature,
            "curvature_preview_station": (
                curvature_preview_station
            ),
        }


    def score_summary(self, completed, now=None):
        return self.score_tracker.summary(completed, now=now)


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
    # The legacy OpenDRIVE parser imports pyplot at module import time even
    # though route calculation never plots. On the onsite Conda image that
    # unnecessary import reaches Pillow/libLerc and selects the system
    # libstdc++, which lacks GLIBCXX_3.4.29. Provide an explicit headless
    # placeholder so planning remains independent of the visualization ABI.
    if "matplotlib.pyplot" not in sys.modules:
        matplotlib_stub = types.ModuleType("matplotlib")
        matplotlib_stub.__path__ = []
        pyplot_stub = types.ModuleType("matplotlib.pyplot")

        def plotting_disabled(*_args, **_kwargs):
            raise RuntimeError(
                "matplotlib plotting is disabled in the headless "
                "LQR/PID runtime"
            )

        pyplot_stub.plot = plotting_disabled
        pyplot_stub.scatter = plotting_disabled
        pyplot_stub.savefig = plotting_disabled
        pyplot_stub.subplots = plotting_disabled
        pyplot_stub.figure = plotting_disabled
        pyplot_stub.show = plotting_disabled
        pyplot_stub.grid = plotting_disabled
        pyplot_stub.close = plotting_disabled
        matplotlib_stub.pyplot = pyplot_stub
        sys.modules["matplotlib"] = matplotlib_stub
        sys.modules["matplotlib.pyplot"] = pyplot_stub
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


def downsample_route_points(x_values, y_values, max_points=1200):
    """Return finite XY tuples with both endpoints and bounded draw cost."""
    x_array = np.asarray(x_values, dtype=float).reshape(-1)
    y_array = np.asarray(y_values, dtype=float).reshape(-1)
    if x_array.size != y_array.size:
        raise ValueError("visualizer route x/y lengths differ")
    finite = np.isfinite(x_array) & np.isfinite(y_array)
    x_array = x_array[finite]
    y_array = y_array[finite]
    if x_array.size < 2:
        raise ValueError("visualizer route needs at least two points")
    point_limit = max(2, int(max_points))
    if x_array.size > point_limit:
        indices = np.linspace(
            0, x_array.size - 1, point_limit, dtype=int
        )
        x_array = x_array[indices]
        y_array = y_array[indices]
    return list(zip(x_array.tolist(), y_array.tolist()))


def draw_global_route_window(window, state):
    """Draw one dependency-free X11 frame in the visualizer process."""
    window.x11.XClearWindow(window.display, window.window)
    route = state.get("route") or []
    ego = state.get("ego")
    if len(route) < 2:
        window._text(22, 30, "LQR Global Route Tracking")
        window._text(
            22,
            58,
            "Waiting for ActorPrepare and the global route...",
            "muted",
        )
        window._text(
            22,
            84,
            "Closing this window does not stop the controller.",
            "muted",
        )
        window.x11.XFlush(window.display)
        return

    trail = state.get("trail") or []
    world_points = list(route)
    trail_start = len(world_points)
    world_points.extend(trail)
    ego_index = None
    if ego is not None:
        ego_index = len(world_points)
        world_points.append((float(ego["x"]), float(ego["y"])))
    screen_points = window._world_to_screen(world_points)
    route_screen = screen_points[:len(route)]
    trail_screen = screen_points[
        trail_start:trail_start + len(trail)
    ]

    for index in range(1, len(route_screen)):
        previous = route_screen[index - 1]
        current = route_screen[index]
        window._line(
            previous[0],
            previous[1],
            current[0],
            current[1],
            "line",
            2,
        )

    route_length = max(0.0, float(state.get("route_length", 0.0)))
    station = max(0.0, float(state.get("station", 0.0)))
    progress = (
        clip(station / route_length, 0.0, 1.0)
        if route_length > 1e-6
        else 0.0
    )
    passed_count = min(
        len(route_screen),
        max(1, int(round(progress * (len(route_screen) - 1))) + 1),
    )
    for index in range(1, passed_count):
        previous = route_screen[index - 1]
        current = route_screen[index]
        window._line(
            previous[0],
            previous[1],
            current[0],
            current[1],
            "ego",
            4,
        )

    for index in range(1, len(trail_screen)):
        previous = trail_screen[index - 1]
        current = trail_screen[index]
        window._line(
            previous[0],
            previous[1],
            current[0],
            current[1],
            "muted",
            2,
        )

    start_screen = route_screen[0]
    goal_screen = route_screen[-1]
    window._circle(
        start_screen[0], start_screen[1], 6, "ego", True
    )
    window._circle(
        goal_screen[0], goal_screen[1], 9, "goal", False
    )
    window._circle(
        goal_screen[0], goal_screen[1], 3, "goal", True
    )
    window._text(
        start_screen[0] + 8,
        start_screen[1] - 8,
        "START",
        "ego",
    )
    window._text(
        goal_screen[0] + 8,
        goal_screen[1] - 8,
        "GOAL",
        "goal",
    )

    if ego_index is not None:
        ego_screen = screen_points[ego_index]
        heading = float(ego["heading"])
        arrow_length = 28.0
        arrow_end = (
            ego_screen[0] + arrow_length * math.cos(heading),
            ego_screen[1] - arrow_length * math.sin(heading),
        )
        window._circle(
            ego_screen[0], ego_screen[1], 8, "heading", True
        )
        window._line(
            ego_screen[0],
            ego_screen[1],
            arrow_end[0],
            arrow_end[1],
            "heading",
            4,
        )
        arrow_angle = math.atan2(
            -(arrow_end[1] - ego_screen[1]),
            arrow_end[0] - ego_screen[0],
        )
        for offset in (-2.55, 2.55):
            wing_angle = arrow_angle + offset
            window._line(
                arrow_end[0],
                arrow_end[1],
                arrow_end[0] + 11.0 * math.cos(wing_angle),
                arrow_end[1] - 11.0 * math.sin(wing_angle),
                "heading",
                3,
            )

    status = "RUNNING" if state.get("started") else "READY"
    window._text(22, 26, "LQR Global Route Tracking")
    window._text(
        22,
        50,
        f"state={status} session={state.get('session', '-')}",
    )
    if ego is None:
        window._text(22, 74, "Waiting for INS...", "muted")
    else:
        window._text(
            22,
            74,
            f"speed={float(ego['speed']):.2f}m/s  "
            f"target={float(state.get('target_speed', 0.0)):.2f}m/s  "
            f"heading={math.degrees(float(ego['heading'])):.1f}deg",
        )
    window._text(
        22,
        98,
        f"progress={station:.1f}/{route_length:.1f}m "
        f"({100.0 * progress:.1f}%)  "
        f"lateral_error={float(state.get('lateral_error', 0.0)):+.2f}m",
    )
    window._text(
        22,
        window.height - 18,
        "Blue: route  Green: completed  Gray: ego trace  Red: ego/front",
        "muted",
    )
    window.x11.XFlush(window.display)


def global_route_visualizer_worker(
    message_queue, width, height, update_hz, owner_pid
):
    """Own the X11 window and close it when its parent disappears."""
    try:
        communication = importlib.import_module("run_hmxzw")
        window_class = getattr(
            communication, "X11AlignmentWindow", None
        )
        if window_class is None:
            raise AttributeError(
                "run_hmxzw.X11AlignmentWindow is unavailable"
            )
        window = window_class(
            enabled=True, width=int(width), height=int(height)
        )
        if not window.open():
            return
        window.x11.XStoreName(
            window.display,
            window.window,
            b"LQR Global Route Tracking",
        )
        window.x11.XFlush(window.display)
        print(
            "[lqr-pid][route-window] opened in isolated process",
            flush=True,
        )
    except Exception as exc:
        print(
            "[lqr-pid][route-window][WARN] unavailable: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )
        return

    state = {
        "route": [],
        "trail": [],
        "ego": None,
        "session": "",
        "started": False,
        "route_length": 0.0,
        "station": 0.0,
        "target_speed": 0.0,
        "lateral_error": 0.0,
    }
    draw_period = 1.0 / max(1.0, float(update_hz))
    last_draw = 0.0
    running = True
    try:
        while running and window.enabled:
            # Normal shutdown arrives through the queue.  This parent check
            # is the fail-safe for SIGKILL, os._exit, interpreter crashes,
            # or supervisor termination that cannot execute parent cleanup.
            # An orphan is re-parented, so getppid() changes immediately.
            if os.getppid() != int(owner_pid):
                print(
                    "[lqr-pid][route-window] owner exited; closing",
                    flush=True,
                )
                break
            messages = []
            try:
                messages.append(message_queue.get(timeout=0.04))
                while True:
                    try:
                        messages.append(message_queue.get_nowait())
                    except queue.Empty:
                        break
            except queue.Empty:
                pass

            for message in messages:
                if message.get("type") == "stop":
                    running = False
                    break
                global_route_visualizer_apply(state, message)
            if not running:
                continue

            if not window._process_events():
                break
            now = time.monotonic()
            if now - last_draw >= draw_period:
                last_draw = now
                draw_global_route_window(window, state)
    finally:
        window.close()
        print("[lqr-pid][route-window] closed", flush=True)


def global_route_visualizer_apply(state, message):
    message_type = message.get("type")
    if message_type == "clear":
        state.update(
            {
                "route": [],
                "trail": [],
                "ego": None,
                "session": str(message.get("session", "")),
                "started": False,
                "route_length": 0.0,
                "station": 0.0,
                "target_speed": 0.0,
                "lateral_error": 0.0,
            }
        )
        return
    if message_type == "route":
        state.update(message)
        state["trail"] = []
        state["ego"] = None
        return
    if message_type != "ego":
        return
    ego = message.get("ego")
    if ego is not None:
        trail = state.setdefault("trail", [])
        point = (float(ego["x"]), float(ego["y"]))
        if (
            not trail
            or math.hypot(
                point[0] - trail[-1][0],
                point[1] - trail[-1][1],
            )
            >= 0.15
        ):
            trail.append(point)
            if len(trail) > 600:
                del trail[:len(trail) - 600]
    state.update(message)


class GlobalRouteLiveVisualizer:
    """Nonblocking parent-side publisher for the route window."""

    def __init__(self, args):
        self.enabled = not bool(args.no_route_visualizer)
        self.width = int(args.route_visualizer_width)
        self.height = int(args.route_visualizer_height)
        self.update_hz = max(
            1.0, float(args.route_visualizer_hz)
        )
        self.max_points = max(
            100, int(args.route_visualizer_max_points)
        )
        self.context = None
        self.message_queue = None
        self.process = None
        self.last_update = 0.0
        self.warned_stopped = False

    def start(self):
        if not self.enabled or self.process is not None:
            return
        try:
            self.context = multiprocessing.get_context("spawn")
            self.message_queue = self.context.Queue(maxsize=8)
            self.process = self.context.Process(
                target=global_route_visualizer_worker,
                args=(
                    self.message_queue,
                    self.width,
                    self.height,
                    self.update_hz,
                    os.getpid(),
                ),
                name="lqr-global-route-window",
                daemon=True,
            )
            self.process.start()
            print(
                "[lqr-pid][route-window] process started "
                f"pid={self.process.pid}",
                flush=True,
            )
        except Exception as exc:
            self.enabled = False
            print(
                "[lqr-pid][route-window][WARN] start failed; "
                f"control continues without visualization: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )

    def _publish(self, message, important=False):
        if (
            not self.enabled
            or self.process is None
            or self.message_queue is None
        ):
            return False
        if not self.process.is_alive():
            if not self.warned_stopped:
                self.warned_stopped = True
                print(
                    "[lqr-pid][route-window] process is closed; "
                    "control continues normally",
                    flush=True,
                )
            self.enabled = False
            return False
        try:
            if important:
                self.message_queue.put(message, timeout=0.20)
            else:
                self.message_queue.put_nowait(message)
            return True
        except queue.Full:
            return False
        except (BrokenPipeError, EOFError, OSError):
            self.enabled = False
            return False

    def clear(self, session=""):
        self._publish(
            {"type": "clear", "session": str(session)},
            important=True,
        )

    def set_route(self, route, session, route_length):
        try:
            points = downsample_route_points(
                route["x"],
                route["y"],
                max_points=self.max_points,
            )
        except Exception as exc:
            print(
                "[lqr-pid][route-window][WARN] route rejected: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            return
        self._publish(
            {
                "type": "route",
                "route": points,
                "session": str(session),
                "route_length": float(route_length),
                "started": False,
                "station": 0.0,
                "target_speed": 0.0,
                "lateral_error": 0.0,
            },
            important=True,
        )

    def update(self, ego, started, output=None):
        if ego is None:
            return
        now = time.monotonic()
        if now - self.last_update < 1.0 / self.update_hz:
            return
        self.last_update = now
        projection = (
            output.get("projection", {}) if output is not None else {}
        )
        self._publish(
            {
                "type": "ego",
                "ego": {
                    "x": float(ego["x"]),
                    "y": float(ego["y"]),
                    "heading": float(ego["heading"]),
                    "speed": float(ego["speed"]),
                },
                "started": bool(started),
                "station": float(projection.get("station", 0.0)),
                "target_speed": float(
                    output.get("target_speed", 0.0)
                    if output is not None
                    else 0.0
                ),
                "lateral_error": float(
                    projection.get("lateral_error", 0.0)
                ),
            }
        )

    def close(self):
        if self.process is None:
            return
        process = self.process
        message_queue = self.message_queue
        self._publish({"type": "stop"}, important=True)
        try:
            process.join(timeout=1.5)
            if process.is_alive():
                print(
                    "[lqr-pid][route-window] graceful close timed out; "
                    "terminating process",
                    flush=True,
                )
                process.terminate()
                process.join(timeout=1.0)
            if process.is_alive() and hasattr(process, "kill"):
                print(
                    "[lqr-pid][route-window] terminate timed out; "
                    "killing process",
                    flush=True,
                )
                process.kill()
                process.join(timeout=1.0)
        finally:
            if message_queue is not None:
                try:
                    message_queue.close()
                    message_queue.cancel_join_thread()
                except (OSError, ValueError):
                    pass
            if not process.is_alive():
                try:
                    process.close()
                except (OSError, ValueError):
                    pass
        self.process = None
        self.message_queue = None
        self.enabled = False


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
        self.last_controlled_ins_sequence = None
        self.last_control_output = None
        self.ego = None
        self.last_heading = None
        self.last_heading_time = None
        self.last_control_time = 0.0
        self.last_debug_time = 0.0
        self.test_started_monotonic = None
        self.last_ins_monotonic = None
        self.episode_task_timeout = float(args.task_timeout)
        self.filtered_yaw_rate = 0.0
        self.control_period = 1.0 / max(1.0, args.control_hz)
        self.controller = GlobalPathLqrPidController(args)
        self.route_planner = setup_global_route_planner(
            Path(args.route_cache_dir)
        )
        self.logger = LearningCsvLogger(args.log_dir, args)
        self.route_visualizer = GlobalRouteLiveVisualizer(args)
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
        self.last_controlled_ins_sequence = None
        self.last_control_output = None
        self.ego = None
        self.last_heading = None
        self.last_heading_time = None
        self.test_started_monotonic = None
        self.last_ins_monotonic = None
        self.episode_task_timeout = float(self.args.task_timeout)
        self.filtered_yaw_rate = 0.0
        self.controller.reset()
        self.session_id = session if keep_session else ""
        self.route_visualizer.clear(self.session_id)

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
            resolved_initial_speed = initial_state_speed_mps(initial)
            if resolved_initial_speed is None:
                resolved_initial_speed = finite_float(
                    initial.get("v", 0.0)
                )
            initial_speed = max(0.0, resolved_initial_speed)
            try:
                xodr_path = self.route_planner.resolve_map_path(
                    map_ref
                )
            except Exception:
                xodr_path = None
            expected_speed = resolve_expected_speed(
                brief_data,
                map_ref,
                xodr_path=xodr_path,
                command_line_mps=self.args.expected_speed_mps,
                use_xodr=self.args.use_xodr_expected_speed,
            )
            legal_speed = float(expected_speed["speed_mps"])
            self.controller.set_route(
                route,
                initial_speed,
                legal_speed=legal_speed,
                expected_speed=legal_speed,
            )
            self.route_visualizer.set_route(
                route,
                session=self.session_id,
                route_length=self.controller.reference.length,
            )
            if self.args.task_timeout > 0.0:
                self.episode_task_timeout = max(
                    float(self.args.task_timeout),
                    2.0 * self.controller.reference.expected_time
                    + 10.0,
                )
            print(
                "[lqr-pid][prepare] route ready "
                f"session={self.session_id} map={map_ref} "
                f"points={len(route['x'])} "
                f"length={self.controller.reference.length:.2f}m "
                f"initial_speed={initial_speed:.2f}m/s "
                f"legal_speed={legal_speed:.2f}m/s "
                f"legal_speed_source={expected_speed['source']} "
                f"expected_time="
                f"{self.controller.reference.expected_time:.2f}s "
                f"task_timeout={self.episode_task_timeout:.2f}s "
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
                self.controller.start_scoring(
                    self.test_started_monotonic
                )
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
            summary = self.controller.score_summary(
                notify.type == self.sdk["NT_FINISH_TEST"],
                now=time.monotonic(),
            )
            print(
                "[lqr-pid][score-proxy] "
                + json.dumps(summary, ensure_ascii=False),
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
                raw_yaw_rate = 0.0
            else:
                raw_yaw_rate = wrap_angle(
                    heading - self.last_heading
                ) / max(1e-3, now - self.last_heading_time)
        else:
            raw_yaw_rate = yaw_rate
        yaw_dt = (
            self.control_period
            if self.last_heading_time is None
            else max(1e-3, now - self.last_heading_time)
        )
        yaw_alpha = clip(yaw_dt / (0.08 + yaw_dt), 0.05, 1.0)
        self.filtered_yaw_rate += yaw_alpha * (
            raw_yaw_rate - self.filtered_yaw_rate
        )
        yaw_rate = self.filtered_yaw_rate
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
        self.route_visualizer.update(
            self.ego,
            self.started,
            output=self.last_control_output,
        )

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
        sequence = self.ego["sequence"]
        if sequence == self.last_controlled_ins_sequence:
            if self.last_control_output is not None:
                self.send_control(
                    self.last_control_output["acceleration"],
                    self.last_control_output["speed"],
                    self.last_control_output["steering_wheel_deg"],
                )
            return
        if self.last_controlled_ins_sequence is None:
            # prepare.init_state.v is not always the simulator's actual
            # spawn speed.  Starting the speed slew limiter from the first
            # INS avoids publishing an artificial speed jump.
            self.controller.synchronize_speed(self.ego["speed"])
        self.last_controlled_ins_sequence = sequence
        output = self.controller.control(self.ego, now)
        self.last_control_output = output
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
            "score_profile": self.args.score_profile,
            "experiment_tag": self.args.experiment_tag,
            "speed": self.ego["speed"],
            "expected_speed": (
                self.controller.reference.expected_speed_mps
            ),
            "legal_speed": projection["legal_speed"],
            "path_target_speed": output["path_target_speed"],
            "target_speed": output["target_speed"],
            "published_speed": output["speed"],
            "feedback_speed_guard": int(
                output["feedback_speed_guard"]
            ),
            "requested_curvature_speed_cap": (
                output["requested_curvature_speed_cap"]
                if math.isfinite(
                    output["requested_curvature_speed_cap"]
                )
                else ""
            ),
            "speed_limit_ratio_actual": output["score_metrics"][
                "speed_limit_ratio_actual"
            ],
            "pid_error": pid.error,
            "pid_p": pid.proportional,
            "pid_i": pid.integral,
            "pid_d": pid.derivative,
            "pid_ff": pid.feedforward,
            "acceleration": output["acceleration"],
            "command_jerk": output["command_jerk"],
            "measured_longitudinal_accel": output["score_metrics"][
                "measured_longitudinal_accel"
            ],
            "measured_longitudinal_jerk": output["score_metrics"][
                "measured_longitudinal_jerk"
            ],
            "estimated_lateral_accel": output["score_metrics"][
                "estimated_lateral_accel"
            ],
            "estimated_lateral_jerk": output["score_metrics"][
                "estimated_lateral_jerk"
            ],
            "measured_yaw_rate": output["score_metrics"][
                "measured_yaw_rate"
            ],
            "comfort_score_proxy": output["score_metrics"][
                "comfort_score_proxy"
            ],
            "rule_score_proxy": output["score_metrics"][
                "rule_score_proxy"
            ],
            "station": projection["station"],
            "remaining": projection["remaining"],
            "path_distance": projection["distance"],
            "lateral_error": projection["lateral_error"],
            "lateral_error_rate": output["lateral_error_rate"],
            "heading_error": output["heading_error"],
            "heading_error_rate": output["heading_error_rate"],
            "curvature": projection["curvature"],
            "control_curvature": output["control_curvature"],
            "curvature_preview_station": output[
                "curvature_preview_station"
            ],
            "lqr_model": lqr["model"],
            "recovery_mode": int(output["recovery_mode"]),
            "lateral_feedback_controller": output[
                "lateral_feedback_controller"
            ],
            "lateral_feedback_limit_deg": math.degrees(
                output["lateral_feedback_limit"]
            ),
            "lqr_model_speed": lqr["model_speed"],
            "lqr_k": json.dumps(
                lqr["k"].reshape(-1).tolist()
            ),
            "lqr_feedforward_deg": math.degrees(
                lqr["feedforward"]
            ),
            "lqr_model_feedforward_deg": math.degrees(
                lqr["model_feedforward"]
            ),
            "lqr_geometric_feedforward_deg": math.degrees(
                lqr["geometric_feedforward"]
            ),
            "lqr_feedback_deg": math.degrees(lqr["feedback"]),
            "lqr_feedback_unclipped_deg": math.degrees(
                lqr["feedback_unclipped"]
            ),
            "lateral_damping_blend": lqr["damping_blend"],
            "road_wheel_deg": math.degrees(
                output["road_wheel"]
            ),
            "road_wheel_rate_deg": math.degrees(
                output["road_wheel_rate"]
            ),
            "comfort_road_wheel_rate_limit_deg": (
                math.degrees(
                    output["comfort_road_wheel_rate_limit"]
                )
                if math.isfinite(
                    output["comfort_road_wheel_rate_limit"]
                )
                else ""
            ),
            "predicted_command_lateral_accel": output[
                "predicted_command_lateral_accel"
            ],
            "predicted_command_lateral_jerk": output[
                "predicted_command_lateral_jerk"
            ],
            "longitudinal_accel_lateral_jerk_limit": (
                output["longitudinal_accel_lateral_jerk_limit"]
                if math.isfinite(
                    output[
                        "longitudinal_accel_lateral_jerk_limit"
                    ]
                )
                else ""
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
                f"profile={self.args.score_profile} "
                f"v={self.ego['speed']:.2f}->"
                f"{output['target_speed']:.2f}m/s "
                f"path_v={output['path_target_speed']:.2f} "
                f"feedback_guard="
                f"{int(output['feedback_speed_guard'])} "
                f"ratio={output['score_metrics']['speed_limit_ratio_actual']:.3f} "
                f"ey={projection['lateral_error']:.3f}m "
                f"epsi={math.degrees(output['heading_error']):.2f}deg "
                f"kappa={projection['curvature']:.4f}->"
                f"{output['control_curvature']:.4f} "
                f"model={lqr['model']} "
                f"feedback_ctrl={lqr['feedback_controller']} "
                f"damp={lqr['damping_blend']:.2f} "
                f"delta_ff={math.degrees(lqr['feedforward']):.2f}deg "
                f"delta_fb={math.degrees(lqr['feedback']):.2f}/"
                f"{math.degrees(lqr['feedback_unclipped']):.2f}deg "
                f"steer={output['steering_wheel_deg']:.1f}deg "
                f"delta_rate="
                f"{math.degrees(output['road_wheel_rate']):.2f}deg/s "
                f"PID=[{pid.proportional:.2f},"
                f"{pid.integral:.2f},{pid.derivative:.2f}] "
                f"acc={output['acceleration']:.2f} "
                f"ax={output['score_metrics']['measured_longitudinal_accel']:.2f} "
                f"jx={output['score_metrics']['measured_longitudinal_jerk']:.2f} "
                f"ay={output['score_metrics']['estimated_lateral_accel']:.2f} "
                f"jy={output['score_metrics']['estimated_lateral_jerk']:.2f} "
                f"jy_cmd="
                f"{output['predicted_command_lateral_jerk']:.2f} "
                f"comfort={output['score_metrics']['comfort_score_proxy']:.2f} "
                f"rules={output['score_metrics']['rule_score_proxy']:.2f}",
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
            raise SystemExit(INS_STALL_EXIT_CODE)
        if (
            self.episode_task_timeout > 0.0
            and elapsed >= self.episode_task_timeout
        ):
            print(
                "[lqr-pid][WATCHDOG] task timeout "
                f"elapsed={elapsed:.2f}s "
                f"limit={self.episode_task_timeout:.2f}s",
                flush=True,
            )
            self.send_control(0.0, 0.0, 0.0)
            raise SystemExit(TASK_TIMEOUT_EXIT_CODE)

    def run(self):
        self.route_visualizer.start()
        try:
            self.create_channels()
            print(
                "[lqr-pid] controller ready "
                f"vehicle={asdict(VEHICLE)} "
                f"csv={self.logger.path}",
                flush=True,
            )
            while True:
                if (
                    self.args.supervisor_pid > 0
                    and os.getppid() != self.args.supervisor_pid
                ):
                    print(
                        "[lqr-pid] supervisor exited; "
                        "closing controller and visualization",
                        flush=True,
                    )
                    break
                self.poll_prepare()
                self.poll_notify()
                self.poll_ins()
                self.poll_feedback()
                self.publish_control()
                self.enforce_watchdogs()
                time.sleep(0.002)
        finally:
            self.route_visualizer.close()
            self.logger.close()


def run_self_test(args=None):
    official_limits = {
        "max_accel": 3.0,
        "max_decel": 3.0,
        "max_lateral_accel": 0.5,
        "max_longitudinal_jerk": 6.0,
        "max_lateral_jerk": 1.0,
        "max_yaw_rate": 0.5,
        "speed_limit_ratio": 1.20,
    }
    score_safe_profiles = ("comfort", "balanced", "efficiency")
    for profile_name in score_safe_profiles:
        profile = SCORE_PROFILES[profile_name]
        for field, official_limit in official_limits.items():
            if not float(profile[field]) < official_limit:
                raise AssertionError(
                    f"{profile_name}.{field} must stay below "
                    f"the scoring boundary {official_limit}"
                )
    attack = SCORE_PROFILES["attack"]
    for field in (
        "max_accel",
        "max_decel",
        "max_longitudinal_jerk",
        "speed_limit_ratio",
    ):
        if not float(attack[field]) < official_limits[field]:
            raise AssertionError(
                f"attack.{field} must stay below "
                f"the scoring boundary {official_limits[field]}"
            )
    if not (
        attack["max_lateral_accel"]
        > official_limits["max_lateral_accel"]
        and attack["max_lateral_jerk"]
        > official_limits["max_lateral_jerk"]
    ):
        raise AssertionError(
            "attack profile must explicitly trade lateral comfort for speed"
        )

    visual_points = downsample_route_points(
        np.arange(2000, dtype=float),
        np.zeros(2000, dtype=float),
        max_points=120,
    )
    if (
        len(visual_points) != 120
        or visual_points[0] != (0.0, 0.0)
        or visual_points[-1] != (1999.0, 0.0)
    ):
        raise AssertionError("route visualizer downsampling failed")
    visual_state = {}
    global_route_visualizer_apply(
        visual_state,
        {
            "type": "route",
            "route": visual_points,
            "session": "self-test",
            "route_length": 1999.0,
        },
    )
    global_route_visualizer_apply(
        visual_state,
        {
            "type": "ego",
            "ego": {
                "x": 1.0,
                "y": 0.0,
                "heading": 0.0,
                "speed": 1.0,
            },
            "station": 1.0,
        },
    )
    if len(visual_state["trail"]) != 1:
        raise AssertionError("route visualizer ego trail update failed")

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

    recovery_controller = DynamicBicycleLQR(
        vehicle,
        LqrWeights(
            kinematic_lateral_error=0.8,
            kinematic_heading_error=3.0,
            kinematic_steering=12.0,
        ),
        dt=0.02,
    )
    recovery_angle, recovery_solution = recovery_controller.control(
        lateral_error=0.537,
        lateral_error_rate=0.0,
        heading_error=math.radians(0.78),
        heading_error_rate=0.0,
        speed=5.0,
        curvature=0.0,
        force_kinematic=True,
    )
    if recovery_solution["model"] != "kinematic":
        raise AssertionError("recovery did not force kinematic LQR")
    if not (-math.radians(15.0) < recovery_angle < 0.0):
        raise AssertionError(
            "recovery LQR response is not bounded/right-turning"
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
    scored_reference = ReferencePath(
        route,
        max_speed=30.0,
        max_lateral_accel=0.43,
        max_accel=2.7,
        max_decel=2.6,
        speed_limit_ratio=1.15,
        max_yaw_rate=0.43,
        max_lateral_jerk=0.80,
        max_longitudinal_jerk=4.80,
        legal_speed_mps=20.0,
        expected_speed_mps=20.0,
    )
    if not np.allclose(scored_reference.legal_speed, 20.0):
        raise AssertionError(
            "planner design speed leaked into legal speed"
        )
    if abs(float(scored_reference.speed[-1]) - 23.0) > 1e-6:
        raise AssertionError(
            "score-safe legal-speed ratio was not applied"
        )
    if abs(scored_reference.expected_time - 5.0) > 1e-6:
        raise AssertionError(
            "expected duration must use evaluator expected speed"
        )

    rate_limit = comfort_road_wheel_rate_limit(
        speed=20.0,
        longitudinal_accel=0.0,
        road_wheel_angle=0.0,
        wheelbase=vehicle.wheelbase,
        max_lateral_jerk=0.80,
    )
    resulting_lateral_jerk = (
        20.0 ** 2 / vehicle.wheelbase * rate_limit
    )
    if resulting_lateral_jerk > 0.80 + 1e-9:
        raise AssertionError(
            "comfort road-wheel rate limit exceeds jerk budget"
        )

    tracker = ScoreProxyTracker()
    tracker.reset(20.0, 5.0)
    tracker.start(100.0)
    score_projection = {
        "legal_speed": 20.0,
        "lateral_error": 0.0,
        "heading": 0.0,
    }
    tracker.update(
        {
            "x": 0.0,
            "y": 0.0,
            "speed": 20.0,
            "longitudinal_speed": 20.0,
            "yaw_rate": 0.0,
            "heading": 0.0,
        },
        score_projection,
        now=100.0,
    )
    tracker.update(
        {
            "x": 2.0,
            "y": 0.0,
            "speed": 20.0,
            "longitudinal_speed": 20.0,
            "yaw_rate": 0.0,
            "heading": 0.0,
        },
        score_projection,
        now=100.1,
    )
    score_summary = tracker.summary(True, now=100.1)
    if abs(score_summary["efficiency_score_proxy"] - 100.0) > 1e-6:
        raise AssertionError(
            "efficiency proxy must use wall time and actual distance"
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

    if args is not None:
        attack_args = argparse.Namespace(**vars(args))
        for name, value in SCORE_PROFILES["attack"].items():
            setattr(attack_args, name, value)
        straight_x = np.linspace(0.0, 8.0, 41)
        straight_y = np.zeros_like(straight_x)
        arc_angle = np.linspace(-math.pi / 2.0, 0.0, 158)
        arc_x = 8.0 + 20.0 * np.cos(arc_angle)
        arc_y = 20.0 + 20.0 * np.sin(arc_angle)
        intersection_route = {
            "x": np.concatenate((straight_x, arc_x[1:])),
            "y": np.concatenate((straight_y, arc_y[1:])),
        }
        attack_controller = GlobalPathLqrPidController(attack_args)
        attack_controller.set_route(
            intersection_route,
            initial_speed=8.9,
            legal_speed=20.0,
            expected_speed=20.0,
        )
        sim_x = 1.0
        sim_y = 0.34
        sim_heading = 0.0
        sim_speed = 8.9
        sim_output = None
        max_intersection_error = 0.0
        reached_intersection_goal = False
        for index in range(350):
            sim_yaw_rate = (
                0.0
                if sim_output is None
                else (
                    sim_speed
                    * math.tan(sim_output["road_wheel"])
                    / VEHICLE.wheelbase
                )
            )
            sim_ego = {
                "x": sim_x,
                "y": sim_y,
                "heading": sim_heading,
                "vx_world": sim_speed * math.cos(sim_heading),
                "vy_world": sim_speed * math.sin(sim_heading),
                "yaw_rate": sim_yaw_rate,
                "speed": sim_speed,
            }
            sim_output = attack_controller.control(
                sim_ego, now=10.0 + 0.02 * index
            )
            sim_road_wheel = sim_output["road_wheel"]
            sim_heading = wrap_angle(
                sim_heading
                + sim_speed
                * math.tan(sim_road_wheel)
                / VEHICLE.wheelbase
                * 0.02
            )
            sim_x += sim_speed * math.cos(sim_heading) * 0.02
            sim_y += sim_speed * math.sin(sim_heading) * 0.02
            sim_speed = max(
                0.0,
                sim_speed + sim_output["acceleration"] * 0.02,
            )
            max_intersection_error = max(
                max_intersection_error,
                abs(
                    sim_output["projection"]["lateral_error"]
                ),
            )
            if sim_output["projection"]["remaining"] < 0.40:
                reached_intersection_goal = True
                break
        if (
            not reached_intersection_goal
            or max_intersection_error > 0.80
        ):
            raise AssertionError(
                "attack intersection capture failed: "
                f"reached={reached_intersection_goal} "
                f"max_error={max_intersection_error:.3f}m"
            )

        # Dynamic-bicycle regression for the onsite "high-speed snake"
        # trace.  A kinematic replay hid the tyre/yaw lag that repeatedly
        # carried the vehicle through the centreline, so preserve the
        # measured initial [e, e_dot, heading, yaw-rate] state here.
        high_speed_route = {
            "x": np.linspace(0.0, 420.0, 1401).tolist(),
            "y": np.zeros(1401).tolist(),
            "speed_limit": np.full(
                1401, 33.333333
            ).tolist(),
        }
        high_speed_controller = GlobalPathLqrPidController(
            attack_args
        )
        high_speed_controller.set_route(
            high_speed_route,
            initial_speed=29.8033,
            legal_speed=33.333333,
            expected_speed=33.333333,
        )
        sim_x = 6.60
        sim_speed = 29.8033
        sim_lateral_state = np.array(
            [
                [0.61656],
                [0.70381],
                [math.radians(1.353)],
                [0.0],
            ],
            dtype=float,
        )
        high_speed_errors = []
        high_speed_recovery_samples = 0
        reached_high_speed_goal = False
        for index in range(800):
            sim_y = float(sim_lateral_state[0, 0])
            sim_lateral_error_rate = float(
                sim_lateral_state[1, 0]
            )
            sim_heading = float(sim_lateral_state[2, 0])
            sim_yaw_rate = float(sim_lateral_state[3, 0])
            sim_vy_body = (
                sim_lateral_error_rate
                - sim_speed * math.sin(sim_heading)
            )
            cos_heading = math.cos(sim_heading)
            sin_heading = math.sin(sim_heading)
            sim_vx_world = (
                cos_heading * sim_speed
                - sin_heading * sim_vy_body
            )
            sim_vy_world = (
                sin_heading * sim_speed
                + cos_heading * sim_vy_body
            )
            sim_ego = {
                "x": sim_x,
                "y": sim_y,
                "heading": sim_heading,
                "vx_world": sim_vx_world,
                "vy_world": sim_vy_world,
                "yaw_rate": sim_yaw_rate,
                "speed": math.hypot(
                    sim_vx_world, sim_vy_world
                ),
            }
            sim_output = high_speed_controller.control(
                sim_ego, now=30.0 + 0.02 * index
            )
            high_speed_errors.append(
                abs(
                    sim_output["projection"]["lateral_error"]
                )
            )
            high_speed_recovery_samples += int(
                sim_output["recovery_mode"]
            )
            sim_road_wheel = sim_output["road_wheel"]
            continuous_a, continuous_b, continuous_e = (
                high_speed_controller.lqr
                .dynamic_continuous_matrices(
                    max(sim_speed, 3.0)
                )
            )
            discrete_a, discrete_b, _ = exact_zero_order_hold(
                continuous_a,
                continuous_b,
                continuous_e,
                dt=0.02,
            )
            sim_lateral_state = (
                discrete_a @ sim_lateral_state
                + discrete_b * sim_road_wheel
            )
            sim_x += sim_speed * cos_heading * 0.02
            sim_speed = max(
                0.0,
                sim_speed + sim_output["acceleration"] * 0.02,
            )
            if sim_output["projection"]["remaining"] < 0.50:
                reached_high_speed_goal = True
                break
        if (
            not reached_high_speed_goal
            or max(high_speed_errors) > 1.05
            or max(high_speed_errors[-50:]) > 0.02
            or high_speed_recovery_samples != 0
            or (
                sim_output["lqr"]["feedback_controller"]
                != "pole_damped_lqr"
            )
        ):
            raise AssertionError(
                "attack high-speed straight damping failed: "
                f"reached={reached_high_speed_goal} "
                f"max_error={max(high_speed_errors):.3f}m "
                f"tail_error={max(high_speed_errors[-50:]):.3f}m "
                f"recovery_samples={high_speed_recovery_samples}"
            )

    print("[self-test] vehicle parameters:", asdict(vehicle))
    print(
        "[self-test] PASS "
        f"DARE_residual={straight['dare_residual']:.3e} "
        f"spectral_radius={straight['spectral_radius']:.6f} "
        f"left_ff={math.degrees(left['feedforward']):.3f}deg"
    )


def terminate_subprocess(process, label):
    """Best-effort fallback for older run_hmxzw modules."""
    if process is None or process.poll() is not None:
        return
    print(
        f"[lqr-pid][supervisor] terminate {label} pid={process.pid}",
        flush=True,
    )
    process.terminate()
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5.0)


def supervise_runtime(args):
    """Supervise the controller across old and new run_hmxzw versions."""
    monitor = importlib.import_module("run_hmxzw")

    acquire_lock = getattr(monitor, "acquire_supervisor_lock", None)
    if callable(acquire_lock):
        supervisor_lock = acquire_lock()
        if supervisor_lock is False:
            return 2
    else:
        supervisor_lock = None
        print(
            "[lqr-pid][supervisor][compat] run_hmxzw has no "
            "acquire_supervisor_lock; continuing without the optional "
            "duplicate-instance lock",
            flush=True,
        )

    terminate_process = getattr(
        monitor, "_terminate_managed_process", terminate_subprocess
    )
    management_helper_names = (
        "_find_driver_sim_pids",
        "_start_managed_simulator",
        "_wait_for_driver_sim",
        "_archive_simulator_diagnostics",
        "_kill_simulator_residuals",
    )
    missing_management_helpers = [
        name
        for name in management_helper_names
        if not callable(getattr(monitor, name, None))
    ]
    if not args.no_manage_simulator and missing_management_helpers:
        print(
            "[lqr-pid][supervisor][FATAL] this run_hmxzw version "
            "cannot manage DriverSim; start DriverSim manually and add "
            "--no-manage-simulator. Missing: "
            + ", ".join(missing_management_helpers),
            flush=True,
        )
        return 2

    find_driver_sim_pids = getattr(
        monitor, "_find_driver_sim_pids", None
    )
    start_managed_simulator = getattr(
        monitor, "_start_managed_simulator", None
    )
    wait_for_driver_sim = getattr(
        monitor, "_wait_for_driver_sim", None
    )
    archive_simulator_diagnostics = getattr(
        monitor, "_archive_simulator_diagnostics", None
    )
    kill_simulator_residuals = getattr(
        monitor, "_kill_simulator_residuals", None
    )

    child_argv = [
        sys.executable,
        str(Path(__file__).resolve()),
        *sys.argv[1:],
        "--runtime-child",
        "--supervisor-pid",
        str(os.getpid()),
    ]
    restart_count = 0
    start_failures = 0
    simulator_launcher = None
    runtime = None
    while True:
        try:
            if not args.no_manage_simulator:
                driver_pids = find_driver_sim_pids(
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
                        start_managed_simulator(args)
                    )
                    driver_pids = wait_for_driver_sim(
                        args, simulator_launcher
                    )
                if not driver_pids:
                    start_failures += 1
                    archive_simulator_diagnostics(
                        args,
                        "lqr_pid_startup_process_missing",
                        simulator_launcher,
                    )
                    terminate_process(
                        simulator_launcher,
                        "DriverSim launcher",
                    )
                    kill_simulator_residuals(
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
                    and not find_driver_sim_pids(
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
            terminate_process(
                runtime, "LQR PID runtime"
            )
            terminate_process(
                simulator_launcher, "DriverSim launcher"
            )
            if not args.no_manage_simulator:
                kill_simulator_residuals(
                    args.simulator_dir,
                    "lqr_pid_supervisor_interrupted",
                )
            return 130

        if restart_reason is None:
            terminate_process(
                simulator_launcher, "DriverSim launcher"
            )
            if not args.no_manage_simulator:
                kill_simulator_residuals(
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
            archive_simulator_diagnostics(
                args, restart_reason, simulator_launcher
            )
        terminate_process(
            runtime, "LQR PID runtime"
        )
        terminate_process(
            simulator_launcher, "DriverSim launcher"
        )
        if not args.no_manage_simulator:
            kill_simulator_residuals(
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
        "--supervisor-pid",
        type=int,
        default=0,
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
    parser.add_argument(
        "--score-profile",
        choices=tuple(SCORE_PROFILES),
        default="balanced",
        help="preset baseline; every value can be overridden explicitly",
    )
    parser.add_argument(
        "--experiment-tag",
        default="",
        help="tag appended to CSV/profile filenames for repeated trials",
    )
    parser.add_argument("--max-speed", type=float, default=None)
    parser.add_argument("--max-accel", type=float, default=None)
    parser.add_argument("--max-decel", type=float, default=None)
    parser.add_argument(
        "--max-lateral-accel", type=float, default=None
    )
    parser.add_argument(
        "--max-longitudinal-jerk", type=float, default=None
    )
    parser.add_argument(
        "--max-lateral-jerk", type=float, default=None
    )
    parser.add_argument(
        "--max-yaw-rate", type=float, default=None
    )
    parser.add_argument(
        "--speed-limit-ratio",
        type=float,
        default=None,
        help="commanded map-limit multiplier; keep below 1.20 for full rule score",
    )
    parser.add_argument(
        "--expected-speed-mps",
        type=float,
        default=None,
        help=(
            "override ActorPrepare/map expected and legal speed in m/s"
        ),
    )
    parser.add_argument(
        "--use-xodr-expected-speed",
        action="store_true",
        help=(
            "use the median XODR speed declaration when ActorPrepare "
            "does not provide an expected speed"
        ),
    )
    parser.add_argument(
        "--rule-lane-half-width",
        type=float,
        default=1.55,
        help="nominal permitted half-width around the routed lane centre",
    )
    parser.add_argument(
        "--rule-lane-margin",
        type=float,
        default=0.15,
        help="inside margin used by the route/line-crossing score proxy",
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
        "--recovery-speed", type=float, default=None
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
        "--feedback-speed-guard-lateral-error",
        type=float,
        default=0.40,
        help=(
            "apply raw LQR-curvature speed cap only beyond this "
            "normal-tracking lateral error"
        ),
    )
    parser.add_argument(
        "--feedback-speed-guard-heading-error-deg",
        type=float,
        default=8.0,
        help=(
            "apply raw LQR-curvature speed cap only beyond this "
            "normal-tracking heading error"
        ),
    )
    parser.add_argument(
        "--path-sample-step", type=float, default=0.20
    )

    parser.add_argument("--speed-kp", type=float, default=None)
    parser.add_argument("--speed-ki", type=float, default=None)
    parser.add_argument("--speed-kd", type=float, default=None)
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
        "--q-kinematic-lateral-error",
        type=float,
        default=0.8,
        help="low-speed/recovery kinematic LQR lateral-error weight",
    )
    parser.add_argument(
        "--q-kinematic-heading-error",
        type=float,
        default=3.0,
        help="low-speed/recovery kinematic LQR heading-error weight",
    )
    parser.add_argument(
        "--r-kinematic-steering",
        type=float,
        default=12.0,
        help="low-speed/recovery kinematic LQR steering penalty",
    )
    parser.add_argument(
        "--dynamic-lqr-min-speed", type=float, default=3.0
    )
    parser.add_argument(
        "--lqr-speed-recompute-delta", type=float, default=0.20
    )
    parser.add_argument(
        "--max-lqr-feedback-road-wheel-deg",
        type=float,
        default=None,
        help=(
            "normal-mode LQR feedback-angle cap before adding curvature "
            "feedforward"
        ),
    )
    parser.add_argument(
        "--max-recovery-feedback-road-wheel-deg",
        type=float,
        default=None,
        help=(
            "maximum Stanley recovery feedback angle at low speed"
        ),
    )
    parser.add_argument(
        "--normal-feedback-speed-product-deg-mps",
        type=float,
        default=None,
        help=(
            "straight-road normal feedback cap times speed; attack mode "
            "uses a small value to prevent high-speed bang-bang steering"
        ),
    )
    parser.add_argument(
        "--normal-feedback-full-curvature",
        type=float,
        default=0.02,
        help=(
            "reference curvature at which normal LQR regains its full "
            "feedback-angle authority"
        ),
    )
    parser.add_argument(
        "--high-speed-damping-blend",
        type=float,
        default=None,
        help=(
            "blend from clipped LQR to pole-placed straight-road "
            "damping at high speed"
        ),
    )
    parser.add_argument(
        "--high-speed-damping-gain",
        type=float,
        default=1.5,
        help="gain multiplier for high-speed straight-road damping",
    )
    parser.add_argument(
        "--high-speed-damping-min-speed",
        type=float,
        default=15.0,
        help="m/s speed where straight-road damping begins blending in",
    )
    parser.add_argument(
        "--high-speed-damping-full-speed",
        type=float,
        default=25.0,
        help="m/s speed where straight-road damping reaches full authority",
    )
    parser.add_argument(
        "--recovery-feedback-speed-product-deg-mps",
        type=float,
        default=80.0,
        help=(
            "speed-scaled recovery feedback cap in deg*m/s; the active "
            "cap is this value divided by speed"
        ),
    )
    parser.add_argument(
        "--recovery-stanley-gain",
        type=float,
        default=1.0,
        help="cross-track gain for bounded nonlinear recovery steering",
    )
    parser.add_argument(
        "--recovery-stanley-softening-speed",
        type=float,
        default=2.0,
        help="m/s softening term in nonlinear recovery steering",
    )
    parser.add_argument(
        "--recovery-exit-lateral-error",
        type=float,
        default=0.40,
        help="recovery exits only inside this lateral-error band",
    )
    parser.add_argument(
        "--recovery-entry-corridor-ratio",
        type=float,
        default=None,
        help=(
            "fraction of the legal routed corridor that triggers "
            "nonlinear recovery"
        ),
    )
    parser.add_argument(
        "--recovery-exit-heading-error-deg",
        type=float,
        default=8.0,
        help="recovery exits only inside this heading-error band",
    )
    parser.add_argument(
        "--curvature-preview-time",
        type=float,
        default=None,
        help="seconds of speed-scaled curvature preview for LQR feedforward",
    )
    parser.add_argument(
        "--geometric-feedforward-blend",
        type=float,
        default=None,
        help=(
            "blend from dynamic-LQR affine feedforward (0) to "
            "atan(wheelbase*preview_curvature) feedforward (1)"
        ),
    )

    parser.add_argument(
        "--steering-ratio",
        type=float,
        default=1.65,
        help=(
            "DriverSim command / LQR road-wheel angle ratio; "
            "Cam6 INS/yaw calibration gives about 1.65"
        ),
    )
    parser.add_argument(
        "--steering-sign",
        type=float,
        default=1.0,
        help=(
            "DriverSim command sign; Cam6 positive command produces "
            "positive yaw, so the calibrated default is +1"
        ),
    )
    parser.add_argument(
        "--max-road-wheel-deg", type=float, default=35.0
    )
    parser.add_argument(
        "--max-recovery-road-wheel-deg",
        type=float,
        default=20.0,
        help="road-wheel limit while outside the dynamic-LQR linear region",
    )
    parser.add_argument(
        "--max-road-wheel-rate-deg",
        type=float,
        default=None,
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
        "--no-route-visualizer",
        action="store_true",
        help="disable the separate X11 global-route/ego window",
    )
    parser.add_argument(
        "--route-visualizer-width", type=int, default=1000
    )
    parser.add_argument(
        "--route-visualizer-height", type=int, default=760
    )
    parser.add_argument(
        "--route-visualizer-hz",
        type=float,
        default=10.0,
        help="visual refresh/state publication rate; control remains 50 Hz",
    )
    parser.add_argument(
        "--route-visualizer-max-points",
        type=int,
        default=1200,
        help="maximum downsampled route points drawn per frame",
    )
    parser.add_argument(
        "--debug-period", type=float, default=0.5
    )
    parser.add_argument(
        "--task-timeout", type=float, default=60.0
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
    return resolve_score_profile(parser.parse_args())


def validate_args(args):
    positive = {
        "control_hz": args.control_hz,
        "max_speed": args.max_speed,
        "max_accel": args.max_accel,
        "max_decel": args.max_decel,
        "max_lateral_accel": args.max_lateral_accel,
        "max_longitudinal_jerk": args.max_longitudinal_jerk,
        "max_lateral_jerk": args.max_lateral_jerk,
        "max_yaw_rate": args.max_yaw_rate,
        "speed_limit_ratio": args.speed_limit_ratio,
        "rule_lane_half_width": args.rule_lane_half_width,
        "path_sample_step": args.path_sample_step,
        "feedback_speed_guard_lateral_error": (
            args.feedback_speed_guard_lateral_error
        ),
        "feedback_speed_guard_heading_error_deg": (
            args.feedback_speed_guard_heading_error_deg
        ),
        "steering_ratio": args.steering_ratio,
        "max_road_wheel_deg": args.max_road_wheel_deg,
        "max_recovery_road_wheel_deg": (
            args.max_recovery_road_wheel_deg
        ),
        "max_road_wheel_rate_deg": args.max_road_wheel_rate_deg,
        "max_lqr_feedback_road_wheel_deg": (
            args.max_lqr_feedback_road_wheel_deg
        ),
        "max_recovery_feedback_road_wheel_deg": (
            args.max_recovery_feedback_road_wheel_deg
        ),
        "normal_feedback_speed_product_deg_mps": (
            args.normal_feedback_speed_product_deg_mps
        ),
        "high_speed_damping_gain": args.high_speed_damping_gain,
        "high_speed_damping_min_speed": (
            args.high_speed_damping_min_speed
        ),
        "high_speed_damping_full_speed": (
            args.high_speed_damping_full_speed
        ),
        "recovery_entry_corridor_ratio": (
            args.recovery_entry_corridor_ratio
        ),
        "normal_feedback_full_curvature": (
            args.normal_feedback_full_curvature
        ),
        "recovery_feedback_speed_product_deg_mps": (
            args.recovery_feedback_speed_product_deg_mps
        ),
        "recovery_stanley_gain": args.recovery_stanley_gain,
        "recovery_stanley_softening_speed": (
            args.recovery_stanley_softening_speed
        ),
        "recovery_exit_lateral_error": (
            args.recovery_exit_lateral_error
        ),
        "recovery_exit_heading_error_deg": (
            args.recovery_exit_heading_error_deg
        ),
        "curvature_preview_time": args.curvature_preview_time,
        "route_visualizer_width": args.route_visualizer_width,
        "route_visualizer_height": args.route_visualizer_height,
        "route_visualizer_hz": args.route_visualizer_hz,
        "route_visualizer_max_points": (
            args.route_visualizer_max_points
        ),
        "r_steering": args.r_steering,
        "r_kinematic_steering": args.r_kinematic_steering,
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
        args.q_kinematic_lateral_error,
        args.q_kinematic_heading_error,
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
    if (
        not math.isfinite(args.steering_sign)
        or abs(abs(args.steering_sign) - 1.0) > 1e-9
    ):
        raise ValueError("--steering-sign must be exactly +1 or -1")
    if (
        not math.isfinite(args.geometric_feedforward_blend)
        or not 0.0 <= args.geometric_feedforward_blend <= 1.0
    ):
        raise ValueError(
            "--geometric-feedforward-blend must be between 0 and 1"
        )
    if not 0.0 <= args.high_speed_damping_blend <= 1.0:
        raise ValueError(
            "--high-speed-damping-blend must be between 0 and 1"
        )
    if not 0.0 < args.recovery_entry_corridor_ratio < 1.0:
        raise ValueError(
            "--recovery-entry-corridor-ratio must be between 0 and 1"
        )
    if (
        args.high_speed_damping_full_speed
        <= args.high_speed_damping_min_speed
    ):
        raise ValueError(
            "--high-speed-damping-full-speed must be greater than "
            "--high-speed-damping-min-speed"
        )
    if (
        args.expected_speed_mps is not None
        and (
            not math.isfinite(args.expected_speed_mps)
            or args.expected_speed_mps <= 0.0
        )
    ):
        raise ValueError("--expected-speed-mps must be positive")
    if (
        not math.isfinite(args.rule_lane_margin)
        or args.rule_lane_margin < 0.0
        or args.rule_lane_margin >= args.rule_lane_half_width
    ):
        raise ValueError(
            "--rule-lane-margin must be nonnegative and smaller "
            "than --rule-lane-half-width"
        )
    if not 0.0 < args.speed_limit_ratio < 1.20:
        raise ValueError(
            "--speed-limit-ratio must stay below the 1.20 score boundary"
        )
    recovery_lateral_entry = min(
        args.large_lateral_error,
        args.recovery_entry_corridor_ratio
        * (args.rule_lane_half_width - args.rule_lane_margin),
    )
    if args.recovery_exit_lateral_error >= recovery_lateral_entry:
        raise ValueError(
            "--recovery-exit-lateral-error must be smaller than the "
            "recovery entry threshold"
        )
    if (
        args.recovery_exit_heading_error_deg
        >= args.large_heading_error_deg
    ):
        raise ValueError(
            "--recovery-exit-heading-error-deg must be smaller than "
            "--large-heading-error-deg"
        )


def install_shutdown_signal_handlers():
    """Convert termination signals into exceptions so finally blocks run."""
    shutdown_requested = False

    def handle_shutdown(signum, _frame):
        nonlocal shutdown_requested
        if shutdown_requested:
            return
        shutdown_requested = True
        try:
            received_name = signal.Signals(signum).name
        except (TypeError, ValueError):
            received_name = str(signum)
        print(
            f"[lqr-pid] received {received_name}; cleaning up",
            flush=True,
        )
        raise KeyboardInterrupt

    for signal_name in ("SIGINT", "SIGTERM", "SIGHUP"):
        shutdown_signal = getattr(signal, signal_name, None)
        if shutdown_signal is None:
            continue
        try:
            signal.signal(shutdown_signal, handle_shutdown)
        except (OSError, RuntimeError, ValueError):
            pass


def main():
    install_shutdown_signal_handlers()
    args = parse_args()
    validate_args(args)
    if args.self_test:
        run_self_test(args)
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
