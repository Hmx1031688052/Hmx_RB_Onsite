"""Adapt decoded DriveSim NPC truth to rule-planner obstacle objects."""

import math


OBS_TYPE_VEHICLE = 1


class GroundTruthObstacle:
    """Mutable obstacle contract consumed by RuleBasedPlanner."""

    pass


def _wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class GroundTruthObstacleAdapter:
    """Convert stable NPC role IDs and ego-frame poses into map-frame tracks."""

    DISPLAY_SIZE_FALLBACKS = {
        "Veh_Lynkco": (4.8, 2.0, 1.7),
        "Veh_GeometryC2": (4.8, 2.0, 1.7),
        "Veh_GeometryE": (4.8, 2.0, 1.7),
    }

    def __init__(
        self,
        sensor_offset_x=0.1,
        sensor_offset_y=0.013,
        velocity_filter=0.55,
        trust_reported_velocity=True,
        track_hold_seconds=1.0,
        innovation_gate_m=4.0,
        innovation_gate_speed=12.0,
    ):
        self.sensor_offset_x = float(sensor_offset_x)
        self.sensor_offset_y = float(sensor_offset_y)
        self.velocity_filter = max(
            0.0, min(1.0, float(velocity_filter))
        )
        self.trust_reported_velocity = bool(
            trust_reported_velocity
        )
        self.track_hold_seconds = max(
            0.0, float(track_hold_seconds)
        )
        self.innovation_gate_m = max(
            0.0, float(innovation_gate_m)
        )
        self.innovation_gate_speed = max(
            0.0, float(innovation_gate_speed)
        )
        self._tracks = {}
        self._latest_timestamp = None

    def reset(self):
        self._tracks.clear()
        self._latest_timestamp = None

    @staticmethod
    def _world_vector(local_x, local_y, ego_yaw):
        cos_yaw = math.cos(ego_yaw)
        sin_yaw = math.sin(ego_yaw)
        return (
            cos_yaw * local_x - sin_yaw * local_y,
            sin_yaw * local_x + cos_yaw * local_y,
        )

    @staticmethod
    def _ego_values(ego):
        if isinstance(ego, dict):
            return (
                float(ego["x"]),
                float(ego["y"]),
                float(ego["theta"]),
            )
        return float(ego.x), float(ego.y), float(ego.theta)

    @staticmethod
    def _obstacle_from_state(state, query_timestamp):
        age = max(
            0.0, float(query_timestamp) - float(state["timestamp"])
        )
        obstacle = GroundTruthObstacle()
        obstacle.id = state["id"]
        obstacle.x = float(state["x"]) + float(state["vx"]) * age
        obstacle.y = float(state["y"]) + float(state["vy"]) * age
        obstacle.theta = float(state["theta"])
        obstacle.length = float(state["length"])
        obstacle.width = float(state["width"])
        obstacle.obs_type = OBS_TYPE_VEHICLE
        obstacle.roleType = "RoleType.VEHICLE"
        obstacle.detector_label = 2
        obstacle.score = 1.0
        obstacle.speed = float(state["speed"])
        obstacle.speed_valid = bool(state["speed_valid"])
        obstacle.world_vx = (
            float(state["vx"]) if obstacle.speed_valid else 0.0
        )
        obstacle.world_vy = (
            float(state["vy"]) if obstacle.speed_valid else 0.0
        )
        obstacle.is_static = (
            not obstacle.speed_valid or obstacle.speed < 0.25
        )
        obstacle.is_virtual = False
        obstacle.track_hits = int(state["hits"])
        obstacle.track_misses = int(state["misses"])
        obstacle.track_predicted = (
            age > 1e-6 or obstacle.track_misses > 0
        )
        obstacle.source = "npc_ground_truth"
        obstacle.model_name = str(state["model_name"])
        obstacle.innovation_rejected = bool(
            state.get("innovation_rejected", False)
        )
        return obstacle

    def predict_at(self, timestamp):
        """Return smooth world tracks extrapolated to ``timestamp``."""
        try:
            timestamp = float(timestamp)
        except (TypeError, ValueError):
            return []
        if not math.isfinite(timestamp):
            return []
        obstacles = []
        expired = []
        for role_id, state in self._tracks.items():
            age = timestamp - float(state["timestamp"])
            if age < -1e-3:
                continue
            if age > self.track_hold_seconds + 1e-9:
                expired.append(role_id)
                continue
            obstacles.append(
                self._obstacle_from_state(state, timestamp)
            )
        for role_id in expired:
            self._tracks.pop(role_id, None)
        return obstacles

    def update(self, npc_truth, ego):
        if not isinstance(npc_truth, dict) or ego is None:
            return []
        try:
            timestamp = float(npc_truth["timestamp_s"])
            ego_x, ego_y, ego_yaw = self._ego_values(ego)
        except (KeyError, TypeError, ValueError, AttributeError):
            return []
        if not all(
            math.isfinite(value)
            for value in (timestamp, ego_x, ego_y, ego_yaw)
        ):
            return []

        self._latest_timestamp = timestamp
        active_ids = set()
        for role in npc_truth.get("roles", []):
            position = role.get("position", {})
            dimensions = role.get("dimensions", {})
            raw_velocity = role.get("vector_raw", {})
            try:
                local_x = (
                    float(position["x"]) + self.sensor_offset_x
                )
                local_y = (
                    float(position["y"]) + self.sensor_offset_y
                )
                local_yaw = float(role["yaw"])
                raw_vx = float(raw_velocity.get("x", 0.0))
                raw_vy = float(raw_velocity.get("y", 0.0))
                length = float(dimensions["length"])
                width = float(dimensions["width"])
            except (KeyError, TypeError, ValueError):
                continue
            if not all(
                math.isfinite(value)
                for value in (
                    local_x,
                    local_y,
                    local_yaw,
                    raw_vx,
                    raw_vy,
                    length,
                    width,
                )
            ):
                continue

            world_dx, world_dy = self._world_vector(
                local_x, local_y, ego_yaw
            )
            world_x = ego_x + world_dx
            world_y = ego_y + world_dy
            role_id = str(
                role.get("role_name")
                or "npc_{}".format(role.get("role_index", 0))
            )
            if role_id in active_ids:
                role_id = "{}#{}".format(
                    role_id, role.get("role_index", 0)
                )
            active_ids.add(role_id)

            raw_world_vx, raw_world_vy = self._world_vector(
                raw_vx, raw_vy, ego_yaw
            )
            state = self._tracks.get(role_id)
            hits = 1
            misses = 0
            innovation_rejected = False
            speed_valid = (
                math.hypot(raw_world_vx, raw_world_vy) <= 60.0
            )
            world_vx = raw_world_vx if speed_valid else 0.0
            world_vy = raw_world_vy if speed_valid else 0.0
            if state is not None:
                hits = int(state["hits"])
                dt = timestamp - float(state["timestamp"])
                if abs(dt) < 1e-6:
                    # The control loop may reuse one GT frame several times.
                    # Never transform the same relative measurement with a
                    # newer ego pose; retain the original world measurement.
                    world_x = float(state["x"])
                    world_y = float(state["y"])
                    world_vx = float(state["vx"])
                    world_vy = float(state["vy"])
                    speed_valid = bool(state["speed_valid"])
                elif 0.02 <= dt <= 1.0:
                    predicted_x = (
                        float(state["x"])
                        + float(state["vx"]) * dt
                    )
                    predicted_y = (
                        float(state["y"])
                        + float(state["vy"]) * dt
                    )
                    innovation = math.hypot(
                        world_x - predicted_x,
                        world_y - predicted_y,
                    )
                    innovation_limit = (
                        self.innovation_gate_m
                        + self.innovation_gate_speed * dt
                    )
                    if innovation > innovation_limit:
                        # A replay teleport or corrupt pose must not move an
                        # existing track by tens of metres in one frame.
                        world_x = predicted_x
                        world_y = predicted_y
                        world_vx = float(state["vx"])
                        world_vy = float(state["vy"])
                        speed_valid = bool(state["speed_valid"])
                        misses = int(state["misses"]) + 1
                        innovation_rejected = True
                if (
                    not self.trust_reported_velocity
                    and 0.02 <= dt <= 1.0
                    and not innovation_rejected
                ):
                    measured_vx = (
                        world_x - float(state["x"])
                    ) / dt
                    measured_vy = (
                        world_y - float(state["y"])
                    ) / dt
                    if math.hypot(measured_vx, measured_vy) <= 60.0:
                        alpha = self.velocity_filter
                        world_vx = (
                            alpha * measured_vx
                            + (1.0 - alpha) * float(state["vx"])
                        )
                        world_vy = (
                            alpha * measured_vy
                            + (1.0 - alpha) * float(state["vy"])
                        )
                        speed_valid = True
                if 0.02 <= dt <= 1.0 and not innovation_rejected:
                    hits += 1

            if not role.get("dimensions_valid", False):
                length, width, _ = self.DISPLAY_SIZE_FALLBACKS.get(
                    str(role.get("model_name", "")),
                    (4.7, 2.0, 1.7),
                )
            length = max(0.6, float(length))
            width = max(0.5, float(width))
            world_theta = _wrap(ego_yaw + local_yaw)
            model_name = str(role.get("model_name", ""))
            if innovation_rejected and state is not None:
                world_theta = float(state["theta"])
                length = float(state["length"])
                width = float(state["width"])
                model_name = str(state["model_name"])
            speed = math.hypot(world_vx, world_vy)
            # Rejected measurements are represented by a prediction at the
            # new timestamp. Advancing the state avoids repeatedly integrating
            # the same interval while preserving the previous velocity.
            self._tracks[role_id] = {
                "id": role_id,
                "timestamp": timestamp,
                "x": world_x,
                "y": world_y,
                "vx": world_vx,
                "vy": world_vy,
                "speed_valid": speed_valid,
                "speed": speed if speed_valid else 0.0,
                "theta": world_theta,
                "length": length,
                "width": width,
                "model_name": model_name,
                "hits": hits,
                "misses": misses,
                "innovation_rejected": innovation_rejected,
            }

        for role_id, state in list(self._tracks.items()):
            if role_id in active_ids:
                continue
            age = timestamp - float(state["timestamp"])
            if age > self.track_hold_seconds + 1e-9:
                self._tracks.pop(role_id, None)
            elif age > 1e-6:
                state["misses"] = int(state["misses"]) + 1
                state["innovation_rejected"] = False

        return self.predict_at(timestamp)
