import math
import os
import re
import xml.etree.ElementTree as ET


_MPS_KEYS = {
    "expectedspeedmps",
    "expectvehiclespeedmps",
    "scenespeedlimitmps",
    "speedlimitmps",
    "targetspeedmps",
}
_KMH_KEYS = {
    "expectedspeedkmh",
    "expectedspeedkph",
    "scenespeedlimitkmh",
    "scenespeedlimitkph",
    "speedlimitkmh",
    "speedlimitkph",
    "targetspeedkmh",
    "targetspeedkph",
}
_GENERIC_KEYS = {
    "expectedspeed",
    "expectvehiclespeed",
    "scenespeedlimit",
    "speedlimit",
}
_UNIT_KEYS = {
    "speedunit",
    "unit",
    "velocityunit",
}


def scene_speed_limit_for_map(map_file):
    name = os.path.splitext(os.path.basename(str(map_file or "")))[0].lower()
    if "highway" in name or "merge" in name:
        return 120.0 / 3.6
    if "intersection" in name or "rule_complaince" in name or "rule_compliance" in name:
        return 20.0
    if "urbanvillage" in name or "urban_village" in name or "non-motorized" in name:
        return 20.0
    if "lefthand_traffic" in name or "third_wheel" in name or "tongji" in name:
        return 20.0
    return 10.0


def _normalise_key(value):
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower())


def _semantic_priority(normalised_key):
    if normalised_key.startswith("expected"):
        return 0
    if normalised_key.startswith("scene"):
        return 1
    if normalised_key.startswith("speedlimit"):
        return 2
    return 3


def _numeric_speed(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, dict):
        for key in ("value", "speed", "max", "limit"):
            if key in value:
                return _numeric_speed(value[key])
        return None
    try:
        if isinstance(value, str):
            match = re.search(
                r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)",
                value,
            )
            if match is None:
                return None
            result = float(match.group(0))
        else:
            result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result) or result <= 0.0:
        return None
    return result


def initial_state_speed_mps(init_state):
    """Read launch speed without confusing it with a road speed limit."""
    if not isinstance(init_state, dict):
        return None
    value = init_state.get("speed")
    if isinstance(value, dict):
        value = value.get("value", value.get("speed"))
    try:
        speed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(speed) or speed < 0.0:
        return None
    return speed


def _unit_from_value(value, siblings=None):
    unit = ""
    if isinstance(value, dict):
        for key, item in value.items():
            if _normalise_key(key) in _UNIT_KEYS:
                unit = str(item or "")
                break
    if not unit and isinstance(siblings, dict):
        for key, item in siblings.items():
            if _normalise_key(key) in _UNIT_KEYS:
                unit = str(item or "")
                break
    if not unit and isinstance(value, str):
        unit = value
    compact = _normalise_key(unit)
    if "kmh" in compact or "kph" in compact:
        return "km/h"
    if "mph" in compact:
        return "mph"
    if (
        "mps" in compact
        or "ms" == compact
        or "meterpersecond" in compact
        or "metrepersecond" in compact
    ):
        return "m/s"
    return ""


def _to_mps(value, unit):
    if unit == "km/h":
        return value / 3.6
    if unit == "mph":
        return value * 0.44704
    return value


def find_prepare_speed_candidates(brief_data):
    """Return only explicit expected/scene speed fields from brief_data.

    Vehicle initial/target-state ``speed`` values are intentionally excluded;
    they describe state, not the evaluator's expected cruising speed.
    """
    candidates = []

    def walk(value, path):
        if isinstance(value, dict):
            for key, item in value.items():
                normalised = _normalise_key(key)
                item_path = path + [str(key)]
                numeric = _numeric_speed(item)
                unit = ""
                priority = None
                if normalised in _MPS_KEYS:
                    unit = "m/s"
                    priority = _semantic_priority(normalised)
                elif normalised in _KMH_KEYS:
                    unit = "km/h"
                    priority = _semantic_priority(normalised)
                elif normalised in _GENERIC_KEYS:
                    unit = _unit_from_value(item, value)
                    priority = (
                        _semantic_priority(normalised)
                        + (0 if unit else 1)
                    )
                if numeric is not None and priority is not None:
                    # DriveSim state is SI.  A generic unitless value remains
                    # SI, but receives lower priority and is clearly marked.
                    speed_mps = _to_mps(numeric, unit or "m/s")
                    if 0.1 <= speed_mps <= 100.0:
                        candidates.append(
                            {
                                "path": ".".join(item_path),
                                "raw_value": numeric,
                                "unit": unit or "m/s-assumed",
                                "speed_mps": speed_mps,
                                "priority": priority,
                            }
                        )
                walk(item, item_path)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, path + [str(index)])

    walk(brief_data, [])
    candidates.sort(
        key=lambda item: (
            item["priority"],
            len(item["path"].split(".")),
            item["path"],
        )
    )
    return candidates


def xodr_speed_summary(xodr_path):
    """Read XODR speed declarations for diagnostics, without building a map."""
    values = []
    if not xodr_path or not os.path.isfile(xodr_path):
        return {
            "count": 0,
            "min_mps": None,
            "median_mps": None,
            "max_mps": None,
        }
    try:
        root = ET.parse(xodr_path).getroot()
        for element in root.iter():
            if str(element.tag).split("}")[-1].lower() != "speed":
                continue
            raw = _numeric_speed(element.attrib.get("max"))
            if raw is None:
                continue
            unit_text = str(element.attrib.get("unit", "m/s"))
            compact = _normalise_key(unit_text)
            unit = (
                "km/h"
                if "kmh" in compact or "kph" in compact
                else ("mph" if "mph" in compact else "m/s")
            )
            converted = _to_mps(raw, unit)
            if 0.1 <= converted <= 100.0:
                values.append(converted)
    except (OSError, ET.ParseError):
        values = []
    values.sort()
    count = len(values)
    return {
        "count": count,
        "min_mps": values[0] if count else None,
        "median_mps": (
            values[count // 2]
            if count % 2
            else (
                0.5 * (values[count // 2 - 1] + values[count // 2])
                if count
                else None
            )
        ),
        "max_mps": values[-1] if count else None,
    }


def resolve_expected_speed(
    brief_data,
    map_file,
    xodr_path=None,
    command_line_mps=None,
    use_xodr=False,
):
    """Resolve the evaluator-style expected speed with an auditable source."""
    candidates = find_prepare_speed_candidates(brief_data)
    xodr = xodr_speed_summary(xodr_path)
    if command_line_mps is not None:
        speed = _numeric_speed(command_line_mps)
        if speed is not None:
            source = "command-line"
        else:
            speed = None
            source = ""
    elif candidates:
        speed = float(candidates[0]["speed_mps"])
        source = "prepare:{}".format(candidates[0]["path"])
    elif use_xodr and xodr["median_mps"] is not None:
        speed = float(xodr["median_mps"])
        source = "xodr:median-speed-declaration"
    else:
        speed = float(scene_speed_limit_for_map(map_file))
        source = "map-category-fallback"
    return {
        "speed_mps": speed,
        "source": source,
        "prepare_candidates": candidates,
        "xodr": xodr,
    }
