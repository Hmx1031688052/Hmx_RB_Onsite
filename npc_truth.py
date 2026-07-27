"""Decode DriveSim's custom binary NPC truth channel.

The ``npc`` multicast payload is not the ``PubRole`` protobuf used by the
``pubrole`` channel.  It starts with a little-endian role count followed by
variable-length role names/model names and a fixed 172-byte numeric record.
Unknown fields are deliberately retained so captures remain useful when the
vendor schema is clarified.
"""

import math
import struct


_ROLE_HEADER = struct.Struct("<QQQIQ")
_NUMERIC_BYTES = 172
_POSITION = struct.Struct("<3d")
_QUATERNION = struct.Struct("<4f")
_VECTOR3 = struct.Struct("<3f")
_DIMENSIONS = struct.Struct("<3f")
_UINT32 = struct.Struct("<I")
_FLOAT4 = struct.Struct("<4f")
_TAIL = struct.Struct("<2f")


class NpcTruthDecodeError(ValueError):
    pass


def ensure_npc_truth_timestamp(
    npc_truth,
    previous_truth=None,
    received_monotonic=None,
    wall_timestamp=None,
):
    """Give a valid empty NPC frame a timestamp derived from its receipt.

    DriveSim's zero-role payload contains only the role count, so there is no
    actor timestamp to promote to the frame.  The empty payload is still a
    healthy and semantically important observation: it means that no NPC is
    currently visible.  Use the receive-clock delta to keep its timestamp
    monotonic with the previous frame, falling back to wall time when there is
    no previous frame.
    """
    if not isinstance(npc_truth, dict):
        return npc_truth
    try:
        timestamp = float(npc_truth.get("timestamp_s"))
    except (TypeError, ValueError):
        timestamp = float("nan")
    if math.isfinite(timestamp):
        return npc_truth

    try:
        received_monotonic = float(received_monotonic)
    except (TypeError, ValueError):
        received_monotonic = float("nan")
    try:
        wall_timestamp = float(wall_timestamp)
    except (TypeError, ValueError):
        wall_timestamp = float("nan")

    previous_timestamp = float("nan")
    previous_received = float("nan")
    if isinstance(previous_truth, dict):
        try:
            previous_timestamp = float(
                previous_truth.get("timestamp_s")
            )
        except (TypeError, ValueError):
            pass
        try:
            previous_received = float(
                previous_truth.get("_received_monotonic")
            )
        except (TypeError, ValueError):
            pass

    if (
        math.isfinite(previous_timestamp)
        and math.isfinite(previous_received)
        and math.isfinite(received_monotonic)
    ):
        timestamp = previous_timestamp + max(
            1e-6, received_monotonic - previous_received
        )
    elif math.isfinite(wall_timestamp):
        timestamp = wall_timestamp
    elif math.isfinite(received_monotonic):
        timestamp = received_monotonic
    else:
        raise ValueError(
            "cannot timestamp NPC truth without a valid receive clock"
        )

    npc_truth["timestamp_s"] = float(timestamp)
    npc_truth["timestamp_source"] = "receive_clock_empty_frame"
    return npc_truth


def _require(payload, offset, size, label):
    if offset < 0 or offset + size > len(payload):
        raise NpcTruthDecodeError(
            "{} truncated at offset {}: need {}, payload {}".format(
                label, offset, size, len(payload)
            )
        )


def _quaternion_yaw(x, y, z, w):
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def _vec(names, values):
    return {
        name: float(value)
        for name, value in zip(names, values)
    }


def decode_npc_payload(payload):
    """Return a JSON-serializable representation of one NPC message."""
    payload = bytes(payload)
    _require(payload, 0, 4, "role count")
    role_count = struct.unpack_from("<I", payload, 0)[0]
    # A corrupt count should not make the probe walk arbitrary memory-sized
    # loops or emit misleading partial truth.
    if role_count > 4096:
        raise NpcTruthDecodeError(
            "unreasonable role count {}".format(role_count)
        )

    offset = 4
    roles = []
    for role_index in range(role_count):
        role_start = offset
        _require(
            payload,
            offset,
            _ROLE_HEADER.size + 2,
            "role {} header".format(role_index),
        )
        (
            timestamp_1_ms,
            timestamp_2_ms,
            frame_counter,
            unknown_header_u32,
            timestamp_3_ms,
        ) = _ROLE_HEADER.unpack_from(payload, offset)
        offset += _ROLE_HEADER.size

        role_kind = payload[offset]
        encoded_name_length = payload[offset + 1]
        offset += 2
        # Captures consistently store strlen(name)+1, followed immediately by
        # a 0x22 marker rather than a C NUL byte.
        if encoded_name_length < 1:
            raise NpcTruthDecodeError(
                "role {} invalid encoded name length {}".format(
                    role_index, encoded_name_length
                )
            )
        name_length = encoded_name_length - 1
        _require(
            payload,
            offset,
            name_length + 2,
            "role {} name".format(role_index),
        )
        try:
            role_name = payload[
                offset:offset + name_length
            ].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise NpcTruthDecodeError(
                "role {} name is not UTF-8: {}".format(
                    role_index, exc
                )
            )
        offset += name_length

        name_marker = payload[offset]
        model_length = payload[offset + 1]
        offset += 2
        _require(
            payload,
            offset,
            model_length + _NUMERIC_BYTES,
            "role {} model/numeric data".format(role_index),
        )
        try:
            model_name = payload[
                offset:offset + model_length
            ].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise NpcTruthDecodeError(
                "role {} model is not UTF-8: {}".format(
                    role_index, exc
                )
            )
        offset += model_length

        numeric = payload[offset:offset + _NUMERIC_BYTES]
        offset += _NUMERIC_BYTES
        position = _POSITION.unpack_from(numeric, 0)
        quaternion = _QUATERNION.unpack_from(numeric, 24)
        vector_raw = _VECTOR3.unpack_from(numeric, 40)
        dimensions = _DIMENSIONS.unpack_from(numeric, 52)
        object_type = _UINT32.unpack_from(numeric, 64)[0]
        camera_box_raw = _FLOAT4.unpack_from(numeric, 68)
        auxiliary = [
            _FLOAT4.unpack_from(numeric, 84 + 16 * index)
            for index in range(5)
        ]
        reported_yaw, tail_value = _TAIL.unpack_from(
            numeric, 164
        )
        quaternion_yaw = _quaternion_yaw(*quaternion)
        dimensions_valid = (
            2.0 <= dimensions[0] <= 30.0
            and 0.5 <= dimensions[1] <= 8.0
            and 0.5 <= dimensions[2] <= 8.0
        )
        roles.append(
            {
                "role_index": role_index,
                "role_name": role_name,
                "model_name": model_name,
                "role_kind": int(role_kind),
                "name_marker": int(name_marker),
                "object_type": int(object_type),
                "class_name": (
                    "Vehicle"
                    if object_type == 6
                    else "Unknown"
                ),
                "timestamp_ms": int(timestamp_1_ms),
                "timestamp_s": timestamp_1_ms / 1000.0,
                "timestamps_ms": [
                    int(timestamp_1_ms),
                    int(timestamp_2_ms),
                    int(timestamp_3_ms),
                ],
                "frame_counter": int(frame_counter),
                "unknown_header_u32": int(
                    unknown_header_u32
                ),
                # The samples strongly indicate an ego/sensor-relative frame:
                # z is near -1.9 m and the timestamps align with lidar. Keep
                # the frame explicitly unverified until a static-pose check.
                "position": _vec(
                    ("x", "y", "z"), position
                ),
                "coordinate_frame": "ego_or_lidar_unverified",
                "orientation_quaternion": _vec(
                    ("x", "y", "z", "w"), quaternion
                ),
                "yaw": float(reported_yaw),
                "quaternion_yaw": float(quaternion_yaw),
                "yaw_consistency_error": float(
                    math.atan2(
                        math.sin(reported_yaw - quaternion_yaw),
                        math.cos(reported_yaw - quaternion_yaw),
                    )
                ),
                "vector_raw": _vec(
                    ("x", "y", "z"), vector_raw
                ),
                "dimensions": _vec(
                    ("length", "width", "height"),
                    dimensions,
                ),
                "dimensions_valid": bool(dimensions_valid),
                "camera_box_raw": _vec(
                    ("x", "y", "width", "height"),
                    camera_box_raw,
                ),
                "auxiliary_float4": [
                    [float(value) for value in group]
                    for group in auxiliary
                ],
                "tail_value": float(tail_value),
                "record_offset": int(role_start),
                "record_size": int(offset - role_start),
            }
        )

    if offset != len(payload):
        raise NpcTruthDecodeError(
            "payload has {} trailing bytes after {} roles".format(
                len(payload) - offset, role_count
            )
        )
    timestamps = [
        role["timestamp_s"] for role in roles
    ]
    return {
        "decoder": "drivesim_npc_binary_v1",
        "role_count": int(role_count),
        "timestamp_s": (
            min(timestamps) if timestamps else None
        ),
        "roles": roles,
        "payload_size": len(payload),
    }
