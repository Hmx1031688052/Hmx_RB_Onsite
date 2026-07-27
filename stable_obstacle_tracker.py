"""Small dependency-free multi-object tracker for 3D detector box centres.

The detector supplies map-frame centres and dimensions.  This tracker adds:

* deterministic global one-to-one association;
* constant-velocity prediction and robust median velocity filtering;
* compatible Car/Cyclist association to survive class flicker;
* two-hit confirmation for low-confidence detections;
* short coasting through fresh detector misses.

It deliberately does not alter or invoke the detector model.
"""

import math
import statistics
import time


TYPE_VEHICLE = 1
TYPE_CYCLIST = 3
TYPE_PEDESTRIAN = 4


def _finite(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    return value if math.isfinite(value) else float(default)


def _angle_difference(first, second):
    return math.atan2(
        math.sin(float(first) - float(second)),
        math.cos(float(first) - float(second)),
    )


def _compatible_type(track_type, detection_type):
    if int(track_type) == int(detection_type):
        return True
    road_users = {TYPE_VEHICLE, TYPE_CYCLIST}
    return int(track_type) in road_users and int(detection_type) in road_users


class StableObstacleTracker(object):
    """Constant-velocity tracker with deterministic global greedy matching."""

    def __init__(
        self,
        confirmation_hits=2,
        coast_time=0.60,
        max_age=1.20,
        default_dt=0.10,
    ):
        self.confirmation_hits = max(1, int(confirmation_hits))
        self.coast_time = max(0.0, float(coast_time))
        self.max_age = max(self.coast_time, float(max_age))
        self.default_dt = max(0.01, float(default_dt))
        self.tracks = {}
        self.next_id = 1
        self.last_timestamp = None
        self.last_stats = {}

    def reset(self):
        self.tracks = {}
        self.next_id = 1
        self.last_timestamp = None
        self.last_stats = {}

    def _timestamp(self, timestamp):
        stamp = _finite(timestamp, float("nan"))
        if not math.isfinite(stamp):
            stamp = time.monotonic()
        if (
            self.last_timestamp is not None
            and stamp <= self.last_timestamp
        ):
            stamp = self.last_timestamp + self.default_dt
        self.last_timestamp = stamp
        return stamp

    @staticmethod
    def _prediction(track, timestamp):
        dt = max(
            0.0, timestamp - float(track.get("update_stamp", timestamp))
        )
        return (
            track["x"] + track["vx"] * dt,
            track["y"] + track["vy"] * dt,
            dt,
        )

    @staticmethod
    def _association_gate(track, detection, dt):
        obstacle_type = int(track["type"])
        if obstacle_type == TYPE_PEDESTRIAN:
            base = 1.35
        elif obstacle_type == TYPE_CYCLIST:
            base = 1.70
        else:
            base = 2.10
        speed = math.hypot(track["vx"], track["vy"])
        size = 0.15 * max(track["length"], detection["length"])
        return min(6.0, base + size + 0.35 * speed * dt)

    def _association_cost(
        self, track, detection, predicted_x, predicted_y, dt
    ):
        if not _compatible_type(track["type"], detection["type"]):
            return None
        distance = math.hypot(
            detection["x"] - predicted_x,
            detection["y"] - predicted_y,
        )
        gate = self._association_gate(track, detection, dt)
        if distance > gate:
            return None
        length_scale = max(
            0.5, track["length"], detection["length"]
        )
        width_scale = max(0.3, track["width"], detection["width"])
        size_cost = (
            abs(track["length"] - detection["length"]) / length_scale
            + abs(track["width"] - detection["width"]) / width_scale
        )
        heading_cost = abs(
            _angle_difference(track["heading"], detection["heading"])
        ) / math.pi
        class_cost = (
            0.0
            if int(track["type"]) == int(detection["type"])
            else 0.20
        )
        return distance / max(gate, 1e-6) + 0.25 * size_cost + (
            0.10 * heading_cost + class_cost
        )

    @staticmethod
    def _prepare_detection(raw):
        obstacle_type = int(raw.get("type", TYPE_VEHICLE))
        return {
            "x": _finite(raw.get("x")),
            "y": _finite(raw.get("y")),
            "length": max(0.3, _finite(raw.get("length"), 4.0)),
            "width": max(0.2, _finite(raw.get("width"), 1.8)),
            "heading": _finite(raw.get("heading")),
            "type": obstacle_type,
            "score": max(
                0.0, min(1.0, _finite(raw.get("score"), 0.0))
            ),
        }

    def _allocate_id(self):
        track_id = str(self.next_id)
        self.next_id += 1
        return track_id

    @staticmethod
    def _single_hit_threshold(obstacle_type):
        if int(obstacle_type) == TYPE_VEHICLE:
            return 0.80
        return 0.70

    @staticmethod
    def _maximum_speed(obstacle_type):
        if int(obstacle_type) == TYPE_PEDESTRIAN:
            return 7.0
        if int(obstacle_type) == TYPE_CYCLIST:
            return 22.0
        return 45.0

    @staticmethod
    def _stable_type(track):
        votes = track["type_votes"]
        current = int(track["type"])
        best = max(votes, key=votes.get)
        if best == current:
            return current
        if votes[best] >= 1.20 * max(votes.get(current, 0.0), 1e-6):
            return int(best)
        return current

    def _new_track(self, detection, timestamp):
        track_id = self._allocate_id()
        votes = {
            TYPE_VEHICLE: 0.0,
            TYPE_CYCLIST: 0.0,
            TYPE_PEDESTRIAN: 0.0,
        }
        votes[detection["type"]] = max(0.1, detection["score"])
        confirmed = (
            self.confirmation_hits <= 1
            or detection["score"]
            >= self._single_hit_threshold(detection["type"])
        )
        self.tracks[track_id] = {
            "id": track_id,
            "x": detection["x"],
            "y": detection["y"],
            "vx": 0.0,
            "vy": 0.0,
            "length": detection["length"],
            "width": detection["width"],
            "heading": detection["heading"],
            "type": detection["type"],
            "type_votes": votes,
            "score": detection["score"],
            "hits": 1,
            "misses": 0,
            "confirmed": confirmed,
            "velocity_valid": False,
            "velocity_samples_x": [],
            "velocity_samples_y": [],
            "created_stamp": timestamp,
            "update_stamp": timestamp,
            "seen_stamp": timestamp,
            "predicted": False,
        }
        return track_id

    def _update_track(self, track, detection, timestamp):
        old_x = track["x"]
        old_y = track["y"]
        predicted_x, predicted_y, dt = self._prediction(
            track, timestamp
        )
        dt = max(dt, self.default_dt * 0.5)
        measured_vx = (detection["x"] - old_x) / dt
        measured_vy = (detection["y"] - old_y) / dt
        measured_speed = math.hypot(measured_vx, measured_vy)
        velocity_valid = (
            measured_speed
            <= self._maximum_speed(detection["type"])
        )
        if velocity_valid:
            if measured_speed < 0.35:
                measured_vx = 0.0
                measured_vy = 0.0
            samples_x = track["velocity_samples_x"]
            samples_y = track["velocity_samples_y"]
            samples_x.append(measured_vx)
            samples_y.append(measured_vy)
            del samples_x[:-5]
            del samples_y[:-5]
            median_vx = statistics.median(samples_x)
            median_vy = statistics.median(samples_y)
            alpha = 0.45
            track["vx"] += alpha * (median_vx - track["vx"])
            track["vy"] += alpha * (median_vy - track["vy"])
        else:
            track["vx"] *= 0.8
            track["vy"] *= 0.8

        position_alpha = 0.85
        track["x"] = (
            position_alpha * detection["x"]
            + (1.0 - position_alpha) * predicted_x
        )
        track["y"] = (
            position_alpha * detection["y"]
            + (1.0 - position_alpha) * predicted_y
        )
        dimension_alpha = 0.25
        track["length"] += dimension_alpha * (
            detection["length"] - track["length"]
        )
        track["width"] += dimension_alpha * (
            detection["width"] - track["width"]
        )
        track["heading"] += 0.35 * _angle_difference(
            detection["heading"], track["heading"]
        )
        for obstacle_type in tuple(track["type_votes"]):
            track["type_votes"][obstacle_type] *= 0.90
        track["type_votes"][detection["type"]] = (
            track["type_votes"].get(detection["type"], 0.0)
            + max(0.1, detection["score"])
        )
        track["type"] = self._stable_type(track)
        track["score"] = (
            0.65 * detection["score"] + 0.35 * track["score"]
        )
        track["hits"] += 1
        track["misses"] = 0
        track["confirmed"] = bool(
            track["confirmed"]
            or track["hits"] >= self.confirmation_hits
        )
        track["velocity_valid"] = bool(
            velocity_valid and track["hits"] >= 2
        )
        track["update_stamp"] = timestamp
        track["seen_stamp"] = timestamp
        track["predicted"] = False

    def _coast_track(self, track, timestamp):
        predicted_x, predicted_y, _ = self._prediction(track, timestamp)
        track["x"] = predicted_x
        track["y"] = predicted_y
        track["vx"] *= 0.96
        track["vy"] *= 0.96
        track["misses"] += 1
        track["update_stamp"] = timestamp
        track["predicted"] = True
        track["score"] *= 0.92

    @staticmethod
    def _output(track, timestamp):
        return {
            "id": track["id"],
            "x": track["x"],
            "y": track["y"],
            "vx": track["vx"],
            "vy": track["vy"],
            "speed": math.hypot(track["vx"], track["vy"]),
            "speed_valid": bool(track["velocity_valid"]),
            "length": track["length"],
            "width": track["width"],
            "heading": track["heading"],
            "type": int(track["type"]),
            "score": max(0.0, min(1.0, track["score"])),
            "hits": int(track["hits"]),
            "misses": int(track["misses"]),
            "age": max(0.0, timestamp - track["created_stamp"]),
            "predicted": bool(track["predicted"]),
        }

    def update(self, detections, timestamp=None):
        timestamp = self._timestamp(timestamp)
        prepared = [
            self._prepare_detection(item) for item in (detections or [])
        ]
        track_ids = list(self.tracks)
        predictions = {
            track_id: self._prediction(
                self.tracks[track_id], timestamp
            )
            for track_id in track_ids
        }

        # Sort all eligible edges globally.  This is deterministic and avoids
        # the input-order bias of assigning each detection independently.
        edges = []
        for track_id in track_ids:
            track = self.tracks[track_id]
            predicted_x, predicted_y, dt = predictions[track_id]
            for detection_index, detection in enumerate(prepared):
                cost = self._association_cost(
                    track,
                    detection,
                    predicted_x,
                    predicted_y,
                    dt,
                )
                if cost is not None:
                    edges.append((cost, track_id, detection_index))
        edges.sort(key=lambda item: (item[0], int(item[1]), item[2]))

        matched_tracks = set()
        matched_detections = set()
        matches = []
        for cost, track_id, detection_index in edges:
            if (
                track_id in matched_tracks
                or detection_index in matched_detections
            ):
                continue
            matched_tracks.add(track_id)
            matched_detections.add(detection_index)
            matches.append((track_id, detection_index, cost))

        for track_id, detection_index, _ in matches:
            self._update_track(
                self.tracks[track_id],
                prepared[detection_index],
                timestamp,
            )

        for track_id in track_ids:
            if track_id not in matched_tracks:
                self._coast_track(self.tracks[track_id], timestamp)

        new_ids = []
        for detection_index, detection in enumerate(prepared):
            if detection_index not in matched_detections:
                new_ids.append(self._new_track(detection, timestamp))

        expired = [
            track_id
            for track_id, track in self.tracks.items()
            if timestamp - track["seen_stamp"] > self.max_age
        ]
        for track_id in expired:
            del self.tracks[track_id]

        outputs = []
        coasted = 0
        for track in self.tracks.values():
            since_seen = timestamp - track["seen_stamp"]
            if not track["confirmed"] or since_seen > self.coast_time:
                continue
            if track["predicted"]:
                coasted += 1
            outputs.append(self._output(track, timestamp))
        outputs.sort(key=lambda item: int(item["id"]))
        self.last_stats = {
            "detections": len(prepared),
            "active_tracks": len(self.tracks),
            "confirmed_outputs": len(outputs),
            "matches": len(matches),
            "new_tracks": len(new_ids),
            "coasted_outputs": coasted,
        }
        return outputs
