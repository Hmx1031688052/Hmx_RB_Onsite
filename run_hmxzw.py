"""Minimal multicast start-to-goal straight sprint runner.

Kept:
  - ActorPrepare / ActorPrepareResult
  - Notify start / finish / abort
  - INS pose reception
  - VehicleControl transmission and VehicleFeedback draining

Removed:
  - reinforcement learning
  - camera, lidar and NPC perception
  - Predictor
  - ROS and global planning
  - map loading and obstacle handling
"""

import argparse
import json
import math
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path


# The onsite SDK keeps the generated ``chassis`` and ``main`` protobuf
# packages beside Hmx_RB_Onsite. When this file is launched as
# ``python Hmx_RB_Onsite/run_hmxzw.py``, Python adds only the script directory
# to sys.path, not its parent. Add both explicitly before importing the SDK.
_SCRIPT_DIR = Path(__file__).resolve().parent
_ONSITE_ROOT = _SCRIPT_DIR.parent
for _module_root in (_ONSITE_ROOT, _SCRIPT_DIR):
    _module_root_text = str(_module_root)
    if _module_root_text not in sys.path:
        sys.path.insert(0, _module_root_text)

import libMulticastNetwork


def _add_onsite_proto_root():
    """Locate generated SDK protobuf packages in common onsite layouts."""
    search_bases = [
        _ONSITE_ROOT,
        Path.cwd().resolve(),
        Path(getattr(libMulticastNetwork, "__file__", _SCRIPT_DIR)).resolve().parent,
    ]
    patterns = (
        "chassis/proto/chassis_enums_pb2.py",
        "*/chassis/proto/chassis_enums_pb2.py",
        "*/*/chassis/proto/chassis_enums_pb2.py",
    )
    checked = set()
    for base in search_bases:
        if base in checked or not base.is_dir():
            continue
        checked.add(base)
        for pattern in patterns:
            for enum_file in base.glob(pattern):
                # .../<root>/chassis/proto/chassis_enums_pb2.py
                proto_root = enum_file.parents[2]
                if not (
                    proto_root / "main" / "proto" / "enums_pb2.py"
                ).is_file():
                    continue
                proto_root_text = str(proto_root)
                if proto_root_text not in sys.path:
                    sys.path.insert(0, proto_root_text)
                return proto_root
    return None


_ONSITE_PROTO_ROOT = _add_onsite_proto_root()

from chassis.proto.chassis_enums_pb2 import (
    VEHICLE_CONTROL,
    VEHICLE_FEEDBACK,
)
from chassis.proto.chassis_messages_pb2 import (
    VehicleControl,
    VehicleFeedback,
)
from get_ip import get_ip_address
from main.proto.enums_pb2 import (
    MT_ACTOR_PREPARE,
    MT_ACTOR_PREPARE_RESULT,
    MT_NOTIFY,
    NT_ABORT_TEST,
    NT_COLLIDE_ROLE,
    NT_DESTROY_ROLE,
    NT_FINISH_TEST,
    NT_START_TEST,
)
from main.proto.messages_pb2 import (
    ActorPrepare,
    ActorPrepareResult,
    Notify,
)


TASK_TIMEOUT_RESTART_EXIT_CODE = 75
SIMULATOR_STALL_RESTART_EXIT_CODE = 76


def wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def session_order_key(value):
    match = re.search(r"_(\d{14})_(\d+)$", str(value or "").strip())
    if match is None:
        return None
    try:
        return int(match.group(1)), int(match.group(2))
    except ValueError:
        return None


def is_newer_session(candidate, current):
    candidate_key = session_order_key(candidate)
    current_key = session_order_key(current)
    return (
        candidate_key is not None
        and current_key is not None
        and candidate_key > current_key
    )


class StraightSprintController:
    def __init__(self, args):
        self.align_speed = max(0.0, float(args.align_speed))
        self.align_acceleration = float(args.align_acceleration)
        self.align_tolerance_deg = max(
            0.0, float(args.align_tolerance_deg)
        )
        self.align_confirm_frames = max(
            1, int(args.align_confirm_frames)
        )
        self.steer_kp = float(args.steer_kp)
        self.steer_sign = float(args.steer_sign)
        self.steer_limit_deg = abs(float(args.steer_limit_deg))
        self.steer_min_deg = min(
            self.steer_limit_deg,
            abs(float(args.steer_min_deg)),
        )
        self.sprint_acceleration = float(args.sprint_acceleration)
        self.sprint_speed = max(0.0, float(args.sprint_speed))
        self.line_sample_step = max(0.1, float(args.line_sample_step))

        self.start_xy = None
        self.goal_xy = None
        self.line_heading = None
        self.line_points = []
        self.state = "IDLE"
        self.confirm_count = 0
        self.last_confirm_sequence = None
        self.last_log_time = 0.0

    def reset(self):
        self.start_xy = None
        self.goal_xy = None
        self.line_heading = None
        self.line_points = []
        self.state = "IDLE"
        self.confirm_count = 0
        self.last_confirm_sequence = None
        self.last_log_time = 0.0

    def configure(self, init_state, target_state):
        self.reset()
        try:
            start_x = float(init_state["x"])
            start_y = float(init_state["y"])
            goal_x = float(target_state["x"])
            goal_y = float(target_state["y"])
        except (KeyError, TypeError, ValueError) as exc:
            print(f"[run3][prepare][ERROR] invalid start/goal: {exc}")
            self.state = "INVALID"
            return False

        coordinates = (start_x, start_y, goal_x, goal_y)
        if not all(math.isfinite(value) for value in coordinates):
            print(
                "[run3][prepare][ERROR] non-finite start/goal "
                f"values={coordinates}"
            )
            self.state = "INVALID"
            return False

        dx = goal_x - start_x
        dy = goal_y - start_y
        line_length = math.hypot(dx, dy)
        if line_length < 0.5:
            print(
                "[run3][prepare][ERROR] start/goal distance too small "
                f"distance={line_length:.3f}m"
            )
            self.state = "INVALID"
            return False

        self.start_xy = (start_x, start_y)
        self.goal_xy = (goal_x, goal_y)
        self.line_heading = math.atan2(dy, dx)
        segment_count = max(
            1, int(math.ceil(line_length / self.line_sample_step))
        )
        self.line_points = [
            (
                start_x + dx * index / segment_count,
                start_y + dy * index / segment_count,
            )
            for index in range(segment_count + 1)
        ]
        self.state = "ALIGN"
        print(
            "[run3][straight-line] ready "
            f"start=({start_x:.3f},{start_y:.3f}) "
            f"goal=({goal_x:.3f},{goal_y:.3f}) "
            f"length={line_length:.3f}m "
            f"heading={math.degrees(self.line_heading):.3f}deg "
            f"points={len(self.line_points)}",
            flush=True,
        )
        return True

    def command(self, ego_x, ego_y, ego_heading, ego_speed, ins_sequence):
        if (
            self.state not in ("ALIGN", "SPRINT")
            or self.goal_xy is None
            or self.line_heading is None
        ):
            return 0.0, 0.0, 0.0

        heading_error = wrap_angle(self.line_heading - ego_heading)
        heading_error_deg = math.degrees(heading_error)
        distance_to_goal = math.hypot(
            self.goal_xy[0] - ego_x,
            self.goal_xy[1] - ego_y,
        )

        fresh_ins = (
            ins_sequence is not None
            and ins_sequence != self.last_confirm_sequence
        )
        if fresh_ins:
            self.last_confirm_sequence = ins_sequence

        if self.state == "ALIGN":
            if fresh_ins:
                if abs(heading_error_deg) <= self.align_tolerance_deg:
                    self.confirm_count += 1
                else:
                    self.confirm_count = 0

            if self.confirm_count >= self.align_confirm_frames:
                self.state = "SPRINT"
                print(
                    "[run3] ALIGN -> SPRINT "
                    f"heading_error={heading_error_deg:.3f}deg "
                    f"confirm={self.confirm_count}/"
                    f"{self.align_confirm_frames} "
                    f"acc={self.sprint_acceleration:.1f} "
                    f"speed={self.sprint_speed:.1f}",
                    flush=True,
                )

        if self.state == "SPRINT":
            acceleration = self.sprint_acceleration
            target_speed = self.sprint_speed
            steering = 0.0
        else:
            acceleration = self.align_acceleration
            target_speed = self.align_speed
            if abs(heading_error_deg) <= self.align_tolerance_deg:
                # Centre immediately inside the acceptance band so delayed
                # steering feedback does not carry the chassis through it.
                steering = 0.0
            else:
                raw_steering = (
                    self.steer_sign
                    * self.steer_kp
                    * heading_error_deg
                )
                steering_magnitude = max(
                    self.steer_min_deg,
                    abs(raw_steering),
                )
                steering = math.copysign(
                    min(self.steer_limit_deg, steering_magnitude),
                    raw_steering,
                )

        now = time.monotonic()
        if now - self.last_log_time >= 0.25:
            self.last_log_time = now
            print(
                "[run3][control] "
                f"state={self.state} "
                f"ego=({ego_x:.3f},{ego_y:.3f}) "
                f"ego_heading={math.degrees(ego_heading):.3f}deg "
                f"line_heading={math.degrees(self.line_heading):.3f}deg "
                f"heading_error={heading_error_deg:.3f}deg "
                f"ego_speed={ego_speed:.3f}m/s "
                f"goal_distance={distance_to_goal:.3f}m "
                f"acc={acceleration:.1f} speed={target_speed:.1f} "
                f"steer={steering:.3f}",
                flush=True,
            )

        return acceleration, target_speed, steering


class Run3:
    def __init__(self, args):
        self.args = args
        self.actor_id = str(args.actor_id)
        self.role_id = self.actor_id
        self.session_id = ""
        self.map_name = ""
        self.prepared = False
        self.started = False
        self.prepare_result_sent = False
        self.test_started_monotonic = None
        self.last_ins_monotonic = None

        self.ego = None
        self.first_current_ins_ready = False
        self.last_ins_sequence = None
        self.last_ins_warning_time = 0.0
        self.start_gate_xy = None
        self.start_gate_tolerance = max(
            0.0, float(args.ins_start_gate_tolerance)
        )

        self.controller = StraightSprintController(args)
        self.latest_feedback = None
        self.control_period = 1.0 / max(1.0, float(args.control_hz))
        self.last_control_time = 0.0

        self.prepare_channel = None
        self.notify_channel = None
        self.ins_channel = None
        self.control_channel = None

    def create_channels(self):
        param = libMulticastNetwork.CreateChannelsParam()
        param.config_center_addr = self.args.config_center
        param.local_ip = get_ip_address(self.args.net_interface)
        param.net_interface_name = self.args.net_interface
        param.field_id = self.args.field_id
        param.log_level = 1
        param.client_name = self.actor_id
        param.recv_self_msg = False

        print(f"local_ip {param.local_ip}", flush=True)
        channels = libMulticastNetwork.ChannelPtrVector()
        result = libMulticastNetwork.create_channels(param, channels)
        if result:
            raise RuntimeError(f"create channels failed, ret={result}")

        channel_map = {}
        for channel in channels:
            print(
                "message channel name: {}, id: {}".format(
                    channel.name(), channel.id()
                ),
                flush=True,
            )
            channel_map[channel.name()] = channel

        required = ("prepare", "notify", "ins", "vehiclecontrol")
        missing = [name for name in required if name not in channel_map]
        if missing:
            raise RuntimeError(
                f"required multicast channels missing: {missing}; "
                f"available={sorted(channel_map)}"
            )

        self.prepare_channel = channel_map["prepare"]
        self.notify_channel = channel_map["notify"]
        self.ins_channel = channel_map["ins"]
        self.control_channel = channel_map["vehiclecontrol"]
        print(
            "[run3] communication ready: "
            "prepare/notify/ins/vehiclecontrol",
            flush=True,
        )

    def reset_episode_state(self, keep_session=True):
        current_session = self.session_id
        self.prepared = False
        self.started = False
        self.prepare_result_sent = False
        self.test_started_monotonic = None
        self.last_ins_monotonic = None
        self.map_name = ""
        self.role_id = self.actor_id
        self.ego = None
        self.first_current_ins_ready = False
        self.last_ins_sequence = None
        self.start_gate_xy = None
        self.controller.reset()
        if not keep_session:
            self.session_id = ""
        else:
            self.session_id = current_session

    def send_prepare_result(self, accepted):
        result = ActorPrepareResult()
        result.session_id = self.session_id
        result.actor_id = self.actor_id
        result.result = bool(accepted)
        payload = result.SerializeToString()
        ret = self.prepare_channel.put(
            MT_ACTOR_PREPARE_RESULT, len(payload), payload
        )
        if ret != 0:
            print(
                f"[run3][prepare][ERROR] send result failed ret={ret}",
                flush=True,
            )
            return False
        self.prepare_result_sent = bool(accepted)
        print(
            "[run3][prepare] result sent "
            f"session={self.session_id} accepted={int(bool(accepted))}",
            flush=True,
        )
        return True

    def handle_prepare(self, prepare_message):
        incoming_session = str(
            prepare_message.session_id or ""
        ).strip()
        if not incoming_session:
            print("[run3][prepare][WARN] empty session ignored")
            return

        if incoming_session == self.session_id:
            if self.prepare_result_sent:
                print(
                    "[run3][prepare] duplicate ignored "
                    f"session={incoming_session}",
                    flush=True,
                )
            return

        if self.session_id and not is_newer_session(
            incoming_session, self.session_id
        ):
            print(
                "[run3][prepare][WARN] stale/unordered prepare ignored "
                f"received={incoming_session} current={self.session_id}",
                flush=True,
            )
            return

        if self.session_id:
            print(
                "[run3][prepare] supersede session "
                f"old={self.session_id} new={incoming_session}",
                flush=True,
            )

        self.reset_episode_state(keep_session=False)
        self.session_id = incoming_session
        try:
            brief_data = json.loads(
                prepare_message.archive_info.brief_data
            )
            testees = brief_data.get("testees") or []
            if not testees:
                raise ValueError("brief_data has no testees")
            testee = testees[0]
            init_state = testee["init_state"]
            target_state = testee["target_state"]
            self.role_id = str(
                testee.get("role_id") or self.actor_id
            )
            self.map_name = str(
                brief_data.get("zjl_odv_file")
                or brief_data.get("map_name")
                or brief_data.get("map_id")
                or ""
            )
            self.start_gate_xy = (
                float(init_state["x"]),
                float(init_state["y"]),
            )
            line_ready = self.controller.configure(
                init_state, target_state
            )
        except Exception as exc:
            print(
                "[run3][prepare][ERROR] brief_data rejected "
                f"session={self.session_id} error={type(exc).__name__}: "
                f"{exc}",
                flush=True,
            )
            self.send_prepare_result(False)
            return

        self.prepared = bool(line_ready)
        self.started = False
        print(
            "[run3][prepare] accepted "
            f"session={self.session_id} role={self.role_id} "
            f"map={self.map_name}",
            flush=True,
        )
        self.send_prepare_result(self.prepared)

    def poll_prepare(self):
        for _ in range(16):
            ret, message = self.prepare_channel.get()
            if message is None:
                return
            if ret < 0 or message.type() != MT_ACTOR_PREPARE:
                continue
            payload = libMulticastNetwork.getMessageData(message)
            prepare_message = ActorPrepare()
            prepare_message.ParseFromString(payload)
            self.handle_prepare(prepare_message)

    def handle_notify(self, notify):
        notify_session = str(notify.session_id or "").strip()
        session_sensitive = notify.type in (
            NT_START_TEST,
            NT_FINISH_TEST,
            NT_ABORT_TEST,
        )
        if session_sensitive and self.session_id:
            if not notify_session:
                print(
                    "[run3][notify][WARN] empty-session event ignored "
                    f"type={notify.type} current={self.session_id}",
                    flush=True,
                )
                return
            if notify_session != self.session_id:
                print(
                    "[run3][notify][WARN] stale session ignored "
                    f"type={notify.type} received={notify_session} "
                    f"current={self.session_id}",
                    flush=True,
                )
                return

        if notify.type == NT_START_TEST:
            if not self.prepared or not self.prepare_result_sent:
                print(
                    "[run3][notify][WARN] start ignored before prepare "
                    f"session={notify_session}",
                    flush=True,
                )
                return
            self.started = True
            self.test_started_monotonic = time.monotonic()
            print(
                "[run3][notify] START "
                f"session={self.session_id} role={self.role_id} "
                f"timeout={self.args.task_timeout:.1f}s",
                flush=True,
            )
            return

        if notify.type in (NT_FINISH_TEST, NT_ABORT_TEST):
            event_name = (
                "FINISH" if notify.type == NT_FINISH_TEST else "ABORT"
            )
            print(
                f"[run3][notify] {event_name} "
                f"session={self.session_id} "
                f"controller_state={self.controller.state}",
                flush=True,
            )
            self.send_control(0.0, 0.0, 0.0)
            self.reset_episode_state(keep_session=True)
            return

        if notify.type == NT_DESTROY_ROLE:
            print("[run3][notify] DESTROY_ROLE", flush=True)
            self.send_control(0.0, 0.0, 0.0)
            self.reset_episode_state(keep_session=True)
            return

        if notify.type == NT_COLLIDE_ROLE:
            print(
                "[run3][notify] COLLIDE_ROLE; wait for finish/abort",
                flush=True,
            )

    def poll_notify(self):
        for _ in range(32):
            ret, message = self.notify_channel.get()
            if message is None:
                return
            if ret < 0 or message.type() != MT_NOTIFY:
                continue
            payload = libMulticastNetwork.getMessageData(message)
            notify = Notify()
            notify.ParseFromString(payload)
            self.handle_notify(notify)

    def poll_ins(self):
        ins = self.ins_channel.get_ins()
        try:
            sequence = int(ins.sequence_num)
        except Exception:
            return
        if sequence <= 1 or sequence > 1_000_000:
            return
        if sequence == self.last_ins_sequence:
            return

        try:
            x = float(ins.position.x)
            y = float(ins.position.y)
            heading = float(ins.heading)
            vx = float(ins.linear_velocity.x)
            vy = float(ins.linear_velocity.y)
        except Exception as exc:
            self.warn_ins(f"missing/invalid fields: {exc}")
            return
        if not all(math.isfinite(value) for value in (
            x, y, heading, vx, vy
        )):
            self.warn_ins(
                f"non-finite sample seq={sequence} "
                f"x={x} y={y} heading={heading}"
            )
            return

        if (
            self.prepared
            and not self.first_current_ins_ready
            and self.start_gate_xy is not None
        ):
            start_distance = math.hypot(
                x - self.start_gate_xy[0],
                y - self.start_gate_xy[1],
            )
            if start_distance > self.start_gate_tolerance:
                self.warn_ins(
                    "stale pre-start sample ignored "
                    f"seq={sequence} pos=({x:.3f},{y:.3f}) "
                    f"start=({self.start_gate_xy[0]:.3f},"
                    f"{self.start_gate_xy[1]:.3f}) "
                    f"distance={start_distance:.3f}m "
                    f"threshold={self.start_gate_tolerance:.3f}m"
                )
                return
            self.first_current_ins_ready = True
            print(
                "[run3][ins] first current-episode sample accepted "
                f"seq={sequence} distance_to_start={start_distance:.3f}m",
                flush=True,
            )

        self.last_ins_sequence = sequence
        self.last_ins_monotonic = time.monotonic()
        self.ego = {
            "x": x,
            "y": y,
            "heading": heading,
            "speed": math.hypot(vx, vy),
            "sequence": sequence,
        }

    def warn_ins(self, text):
        now = time.monotonic()
        if now - self.last_ins_warning_time < 1.0:
            return
        self.last_ins_warning_time = now
        print(f"[run3][ins][WARN] {text}", flush=True)

    def poll_feedback(self):
        for _ in range(8):
            ret, message = self.control_channel.get()
            if message is None:
                return
            if ret < 0 or message.type() != VEHICLE_FEEDBACK:
                continue
            payload = libMulticastNetwork.getMessageData(message)
            feedback = VehicleFeedback()
            feedback.ParseFromString(payload)
            self.latest_feedback = feedback

    def send_control(self, acceleration, speed, steering):
        command = VehicleControl()
        command.acceleration = float(acceleration)
        command.speed = float(speed)
        command.steering_control.target_steering_wheel_angle = float(
            steering
        )
        payload = command.SerializeToString()
        ret = self.control_channel.put(
            VEHICLE_CONTROL, len(payload), payload
        )
        if ret != 0:
            print(
                f"[run3][control][ERROR] send failed ret={ret}",
                flush=True,
            )
            return False
        return True

    def send_active_control(self):
        now = time.monotonic()
        if now - self.last_control_time < self.control_period:
            return
        self.last_control_time = now

        if not self.started:
            return
        if not self.first_current_ins_ready or self.ego is None:
            self.send_control(0.0, 0.0, 0.0)
            return

        acceleration, speed, steering = self.controller.command(
            self.ego["x"],
            self.ego["y"],
            self.ego["heading"],
            self.ego["speed"],
            self.ego["sequence"],
        )
        self.send_control(acceleration, speed, steering)

    def enforce_task_timeout(self):
        if (
            not self.started
            or self.test_started_monotonic is None
        ):
            return
        elapsed = time.monotonic() - self.test_started_monotonic
        last_ins_time = (
            self.test_started_monotonic
            if self.last_ins_monotonic is None
            else max(
                self.test_started_monotonic,
                self.last_ins_monotonic,
            )
        )
        ins_silence = time.monotonic() - last_ins_time
        if (
            self.args.ins_stall_timeout > 0.0
            and ins_silence >= self.args.ins_stall_timeout
        ):
            print(
                "[run3][WATCHDOG] INS stalled "
                f"for {ins_silence:.3f}s "
                f"(limit={self.args.ins_stall_timeout:.1f}s); "
                f"session={self.session_id}; "
                "restart DriverSim and runtime",
                flush=True,
            )
            self.send_control(0.0, 0.0, 0.0)
            os._exit(SIMULATOR_STALL_RESTART_EXIT_CODE)

        if self.args.task_timeout <= 0.0:
            return
        if elapsed < self.args.task_timeout:
            return

        print(
            "[run3][WATCHDOG] task did not finish within "
            f"{self.args.task_timeout:.1f}s; "
            f"session={self.session_id} elapsed={elapsed:.3f}s; "
            "terminate runtime child for a clean reconnect",
            flush=True,
        )
        self.send_control(0.0, 0.0, 0.0)
        # A process-level exit is intentional. It guarantees that native
        # multicast channels and sockets are released before the supervisor
        # starts a fresh runtime and re-registers the testee.
        os._exit(TASK_TIMEOUT_RESTART_EXIT_CODE)

    def run(self):
        self.create_channels()
        print(
            "[run3] minimal straight sprint started "
            f"align_speed={self.controller.align_speed:.2f}m/s "
            f"align_acc={self.controller.align_acceleration:.2f}m/s2 "
            f"align_tolerance={self.controller.align_tolerance_deg:.2f}deg "
            f"confirm_frames={self.controller.align_confirm_frames} "
            f"steer_kp={self.controller.steer_kp:.2f} "
            f"steer_range="
            f"[{self.controller.steer_min_deg:.1f},"
            f"{self.controller.steer_limit_deg:.1f}]deg "
            f"sprint_acc={self.controller.sprint_acceleration:.1f} "
            f"sprint_speed={self.controller.sprint_speed:.1f}",
            flush=True,
        )
        while True:
            self.poll_prepare()
            self.poll_notify()
            self.poll_ins()
            self.poll_feedback()
            self.send_active_control()
            self.enforce_task_timeout()
            time.sleep(min(0.002, 0.25 * self.control_period))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Minimal start-to-goal straight sprint runner"
    )
    parser.add_argument(
        "--config_center",
        default="47.110.233.70:52009",
    )
    parser.add_argument(
        "--field_id",
        default="field-zd-test1-22-0331134113-888",
    )
    parser.add_argument(
        "--net_interface",
        default="usb0",
    )
    parser.add_argument("--actor-id", default="apollo_testee")

    parser.add_argument(
        "--align-speed",
        type=float,
        default=6.0,
        help="fast heading-alignment target (default: 6.0 m/s)",
    )
    parser.add_argument(
        "--align-acceleration",
        type=float,
        default=10.0,
        help="fast heading-alignment acceleration (default: 10.0 m/s^2)",
    )
    parser.add_argument(
        "--align-tolerance-deg", type=float, default=3.0
    )
    parser.add_argument(
        "--align-confirm-frames",
        type=int,
        default=2,
        help="fresh in-tolerance INS frames before sprint (default: 2)",
    )
    parser.add_argument(
        "--steer-kp",
        type=float,
        default=3.0,
        help="steering-wheel degrees per heading-error degree (default: 3.0)",
    )
    parser.add_argument("--steer-sign", type=float, default=1.0)
    parser.add_argument(
        "--steer-min-deg",
        type=float,
        default=12.0,
        help="minimum wheel command outside tolerance (default: 12 deg)",
    )
    parser.add_argument(
        "--steer-limit-deg",
        type=float,
        default=42.0,
        help="alignment steering-wheel limit (default: 42 deg)",
    )
    parser.add_argument(
        "--sprint-acceleration", type=float, default=10000.0
    )
    parser.add_argument(
        "--sprint-speed", type=float, default=10000.0
    )
    parser.add_argument(
        "--line-sample-step", type=float, default=1.0
    )
    parser.add_argument("--control-hz", type=float, default=100.0)
    parser.add_argument(
        "--ins-start-gate-tolerance",
        type=float,
        default=50.0,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="accepted for launcher compatibility; control logs are always on",
    )
    parser.add_argument(
        "--task-timeout",
        type=float,
        default=20.0,
        help=(
            "restart the complete runtime if a started task has not ended "
            "within this many seconds; <=0 disables (default: 20)"
        ),
    )
    parser.add_argument(
        "--restart-delay",
        type=float,
        default=3.0,
        help=(
            "delay allowing the daemon to observe the disconnect before "
            "reconnecting (default: 3.0)"
        ),
    )
    parser.add_argument(
        "--simulator-dir",
        default=(
            "/media/pc/FanXiang2T/Onsite_FirstWithForth/"
            "LinuxNoEditor416"
        ),
        help="directory containing the managed DriverSim start.sh",
    )
    parser.add_argument(
        "--simulator-ready-delay",
        type=float,
        default=6.0,
        help="wait after DriverSim start before launching runtime (default: 6)",
    )
    parser.add_argument(
        "--ins-stall-timeout",
        type=float,
        default=4.0,
        help=(
            "after START, restart DriverSim if no fresh INS arrives for this "
            "many seconds; <=0 disables (default: 4)"
        ),
    )
    parser.add_argument(
        "--no-manage-simulator",
        action="store_true",
        help="leave LinuxNoEditor external and restart only this runtime",
    )
    parser.add_argument(
        "--runtime-child",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def _terminate_managed_process(process, label):
    if process is None or process.poll() is not None:
        return
    print(
        f"[run3][supervisor] terminate {label} pid={process.pid}",
        flush=True,
    )
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        process.wait(timeout=5.0)
        return
    except (ProcessLookupError, subprocess.TimeoutExpired):
        pass
    if process.poll() is None:
        print(
            f"[run3][supervisor] kill unresponsive {label} "
            f"pid={process.pid}",
            flush=True,
        )
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            pass
        process.wait()


def _path_is_within(path, root):
    try:
        return os.path.commonpath(
            (str(Path(path).resolve()), str(Path(root).resolve()))
        ) == str(Path(root).resolve())
    except (OSError, ValueError):
        return False


def _kill_simulator_residuals(simulator_dir, reason):
    """SIGKILL detached DriverSim/crash-window processes under one build."""
    if os.name != "posix":
        return

    simulator_root = Path(simulator_dir).expanduser().resolve()
    own_pid = os.getpid()
    residual_pids = []
    for proc_dir in Path("/proc").glob("[0-9]*"):
        try:
            pid = int(proc_dir.name)
        except ValueError:
            continue
        if pid == own_pid:
            continue

        try:
            exe_path = Path(os.readlink(proc_dir / "exe"))
        except OSError:
            exe_path = None
        try:
            cwd_path = Path(os.readlink(proc_dir / "cwd"))
        except OSError:
            cwd_path = None
        try:
            command_parts = (
                (proc_dir / "cmdline")
                .read_bytes()
                .decode(errors="replace")
                .split("\0")
            )
        except OSError:
            command_parts = []

        executable_name = (
            "" if exe_path is None else exe_path.name.lower()
        )
        managed_binary = (
            exe_path is not None
            and _path_is_within(exe_path, simulator_root)
            and (
                executable_name.startswith("driversim")
                or "crashreport" in executable_name
            )
        )
        managed_launcher = (
            cwd_path is not None
            and _path_is_within(cwd_path, simulator_root)
            and any(
                Path(part).name == "start.sh"
                for part in command_parts
                if part
            )
        )
        if managed_binary or managed_launcher:
            residual_pids.append(pid)

    for pid in sorted(set(residual_pids), reverse=True):
        print(
            "[run3][supervisor] SIGKILL simulator residual "
            f"pid={pid} reason={reason}",
            flush=True,
        )
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    if residual_pids:
        time.sleep(0.5)


def _start_managed_simulator(args):
    simulator_dir = Path(args.simulator_dir).expanduser().resolve()
    start_script = simulator_dir / "start.sh"
    if not start_script.is_file():
        raise FileNotFoundError(
            f"DriverSim start script not found: {start_script}"
        )
    print(
        "[run3][supervisor] start DriverSim "
        f"cwd={simulator_dir} command='bash start.sh'",
        flush=True,
    )
    process = subprocess.Popen(
        ["bash", "start.sh"],
        cwd=str(simulator_dir),
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    simulator_failure = threading.Event()
    failure_state = {"reason": None}

    def forward_simulator_output():
        failure_markers = (
            ("登录失败", "login_failure"),
            ("http request failed", "login_failure"),
            ("channels info is empty", "login_failure"),
            ("fatal error:", "fatal_error"),
            ("crashreportclientversion=", "crash_report"),
            ("crashreportcorelog:", "crash_report"),
        )
        output = process.stdout
        if output is None:
            return
        for line in output:
            print(line, end="", flush=True)
            lowered = line.lower()
            for marker, reason in failure_markers:
                if marker.lower() not in lowered:
                    continue
                if not simulator_failure.is_set():
                    failure_state["reason"] = reason
                    simulator_failure.set()
                    print(
                        "[run3][supervisor] DriverSim failure "
                        f"detected from log reason={reason} "
                        f"marker={marker!r}",
                        flush=True,
                    )
                break

    threading.Thread(
        target=forward_simulator_output,
        name="DriverSim-log-monitor",
        daemon=True,
    ).start()
    return process, simulator_failure, failure_state


def supervise_runtime(args):
    child_argv = [
        sys.executable,
        str(Path(__file__).resolve()),
        *sys.argv[1:],
        "--runtime-child",
    ]
    restart_count = 0
    simulator = None
    simulator_failure = None
    simulator_failure_state = None
    runtime = None
    while True:
        try:
            if not args.no_manage_simulator:
                _kill_simulator_residuals(
                    args.simulator_dir,
                    reason="before_start",
                )
                (
                    simulator,
                    simulator_failure,
                    simulator_failure_state,
                ) = (
                    _start_managed_simulator(args)
                )
                ready_delay = max(
                    0.0, float(args.simulator_ready_delay)
                )
                ready_deadline = time.monotonic() + ready_delay
                while time.monotonic() < ready_deadline:
                    if simulator_failure.is_set():
                        break
                    simulator_code = simulator.poll()
                    if simulator_code is not None:
                        print(
                            "[run3][supervisor] DriverSim exited during "
                            f"startup code={simulator_code}",
                            flush=True,
                        )
                        break
                    time.sleep(0.2)
                if (
                    simulator.poll() is not None
                    or simulator_failure.is_set()
                ):
                    startup_reason = (
                        "simulator_"
                        f"{simulator_failure_state['reason'] or 'log_failure'}"
                        if simulator_failure.is_set()
                        else f"simulator_start_exit_{simulator.returncode}"
                    )
                    _terminate_managed_process(
                        simulator,
                        "DriverSim",
                    )
                    _kill_simulator_residuals(
                        args.simulator_dir,
                        reason=startup_reason,
                    )
                    restart_count += 1
                    restart_delay = max(
                        0.0, float(args.restart_delay)
                    )
                    print(
                        "[run3][supervisor] DriverSim startup restart "
                        f"reason={startup_reason} "
                        f"count={restart_count} "
                        f"delay={restart_delay:.1f}s",
                        flush=True,
                    )
                    time.sleep(restart_delay)
                    simulator = None
                    simulator_failure = None
                    simulator_failure_state = None
                    continue

            print(
                "[run3][supervisor] start runtime "
                f"attempt={restart_count + 1} "
                f"task_timeout={args.task_timeout:.1f}s "
                f"ins_stall_timeout={args.ins_stall_timeout:.1f}s",
                flush=True,
            )
            runtime = subprocess.Popen(
                child_argv,
                start_new_session=True,
            )
            restart_reason = None
            return_code = None
            while restart_reason is None:
                return_code = runtime.poll()
                if return_code is not None:
                    if return_code == TASK_TIMEOUT_RESTART_EXIT_CODE:
                        restart_reason = "task_timeout"
                    elif (
                        return_code
                        == SIMULATOR_STALL_RESTART_EXIT_CODE
                    ):
                        restart_reason = "ins_stall"
                    else:
                        break
                if (
                    simulator is not None
                    and simulator.poll() is not None
                ):
                    restart_reason = (
                        "simulator_exit_"
                        f"{simulator.returncode}"
                    )
                if (
                    simulator_failure is not None
                    and simulator_failure.is_set()
                ):
                    restart_reason = (
                        "simulator_"
                        f"{simulator_failure_state['reason'] or 'log_failure'}"
                    )
                if restart_reason is None and return_code is None:
                    time.sleep(0.2)
        except KeyboardInterrupt:
            print(
                "[run3][supervisor] interrupted; stop managed processes",
                flush=True,
            )
            _terminate_managed_process(runtime, "runtime")
            _terminate_managed_process(simulator, "DriverSim")
            if not args.no_manage_simulator:
                _kill_simulator_residuals(
                    args.simulator_dir,
                    reason="supervisor_interrupted",
                )
            return 130

        if restart_reason is None:
            _terminate_managed_process(simulator, "DriverSim")
            if not args.no_manage_simulator:
                _kill_simulator_residuals(
                    args.simulator_dir,
                    reason="runtime_exit",
                )
            print(
                "[run3][supervisor] runtime exited "
                f"code={return_code}; supervisor stops",
                flush=True,
            )
            return return_code

        _terminate_managed_process(runtime, "runtime")
        _terminate_managed_process(simulator, "DriverSim")
        if not args.no_manage_simulator:
            _kill_simulator_residuals(
                args.simulator_dir,
                reason=restart_reason,
            )
        runtime = None
        simulator = None
        simulator_failure = None
        simulator_failure_state = None
        restart_count += 1
        restart_delay = max(0.0, float(args.restart_delay))
        print(
            "[run3][supervisor] restart complete stack "
            f"reason={restart_reason} count={restart_count} "
            f"delay={restart_delay:.1f}s",
            flush=True,
        )
        if restart_delay > 0.0:
            time.sleep(restart_delay)


def main():
    try:
        args = parse_args()
        if not args.runtime_child:
            raise SystemExit(supervise_runtime(args))
        Run3(args).run()
    except KeyboardInterrupt:
        print("[run3] stopped by user", flush=True)
    except Exception as exc:
        print(
            f"[run3][FATAL] {type(exc).__name__}: {exc}",
            flush=True,
        )
        raise


if __name__ == "__main__":
    main()
