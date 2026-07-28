#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直线恒加速测试入口。

控制目标：
1. 使用 ActorPrepare 中给出的真实起点和终点；
2. 规划参考线仅为“起点 -> 终点”的直线；
3. 忽略道路中心线、全局路径和旁车；
4. 从有效控制开始持续给出 2.0 m/s^2 的正加速度；
5. 到达终点前不主动减速、不制动。

说明：
- 该文件是 run_ct.py 的固定参数启动器。
- run_ct.py 负责复用 run.py 的组播通信、场景准备、INS 接收和控制发布。
- 启动阶段仍可能读取地图信息，这是 OnSite 运行协议的一部分；
  但实际 CT 规划不会使用地图道路或全局路径。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


CONSTANT_ACCEL_MPS2 = 2000.0
SPRINT_SPEED_FACTOR = 1.19


def main() -> int:
    current_dir = Path(__file__).resolve().parent
    run_ct_path = current_dir / "run_ct.py"

    if not run_ct_path.is_file():
        print(
            f"[straight-accel][FATAL] 找不到依赖文件: {run_ct_path}\n"
            "请把 run_straight_accel.py 放在 run.py、run_ct.py 同一目录。",
            file=sys.stderr,
        )
        return 2

    # 不创建旁车真值订阅，也不让外部规划控制覆盖直线冲刺命令。
    os.environ["E2E_NPC_TRUTH_ENABLED"] = "0"
    os.environ["E2E_PERCEPTION_SOURCE"] = "none"
    os.environ["E2E_ENABLE_ROS_PLANNING"] = "0"

    # 丢弃 INS 中倒退或缓存的序列，避免控制状态在 SPRINT/STOP 间跳变。
    os.environ["E2E_INS_MONOTONIC_SEQUENCE"] = "1"

    # 用户原有的 run.py 参数继续透传。
    # 固定策略参数放在最后，确保同名参数以这里的值为准。
    forwarded_args = list(sys.argv[1:])
    forced_args = [
        "--ct-speed-factor",
        str(SPRINT_SPEED_FACTOR),
        "--ct-accel",
        str(CONSTANT_ACCEL_MPS2),
        "--ct-alignment-accel",
        str(CONSTANT_ACCEL_MPS2),
        "--ct-alignment-duration",
        "0.02",
        "--ct-alignment-settle-duration",
        "0.10",
        "--ct-channel-settle-duration",
        "0.0",
        "--sprint",
        "--sprint_ignore_obstacles",
        "--perception_source",
        "none",
    ]

    command = [
        sys.executable,
        str(run_ct_path),
        *forwarded_args,
        *forced_args,
    ]

    print(
        "[straight-accel] policy=DIRECT_START_TO_GOAL "
        f"accel={CONSTANT_ACCEL_MPS2:.1f}m/s2 "
        f"speed_factor={SPRINT_SPEED_FACTOR:.2f} "
        "road=IGNORED global_path=IGNORED "
        "background_vehicles=IGNORED braking=DISABLED_BEFORE_FINISH",
        flush=True,
    )

    # 用 run_ct.py 替换当前进程，使它原有的子进程监督逻辑仍然生效。
    os.execv(sys.executable, command)
    return 0  # os.execv 成功后不会执行到这里。


if __name__ == "__main__":
    raise SystemExit(main())
