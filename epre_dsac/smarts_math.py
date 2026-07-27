import numpy as np
import math
from typing import Callable, List, Optional, Sequence, Tuple, Union


def wrap_value(value: Union[int, float], _min: float, _max: float) -> float:
    """Wraps the value around if it goes over max or under min."""
    v = value
    assert isinstance(value, (int, float))
    diff = _max - _min
    if value <= _min:
        v = _max - (_min - value) % diff
    if value > _max:
        v = _min + (value - _max) % diff
    return v


def _gen_ego_frame_matrix(ego_heading):
    transform_matrix = np.eye(3)
    transform_matrix[0, 0] = np.cos(-ego_heading)
    transform_matrix[0, 1] = -np.sin(-ego_heading)
    transform_matrix[1, 0] = np.sin(-ego_heading)
    transform_matrix[1, 1] = np.cos(-ego_heading)
    return transform_matrix


def position_to_ego_frame(position, ego_position, ego_heading):
    """
    Get the position in ego vehicle frame given the pose (of either a vehicle or some point) in global frame.
    Egocentric frame: The ego position becomes origin, and ego heading direction is positive x-axis.
    Args:
        position: [x,y,z]
        ego_position: Ego vehicle [x,y,z]
        ego_heading: Ego vehicle heading in radians
    Returns:
        new_pose: The pose [x,y,z] in egocentric view
    """
    transform_matrix = _gen_ego_frame_matrix(ego_heading)
    ego_rel_position = np.asarray(position) - np.asarray(ego_position)
    new_position = np.matmul(transform_matrix, ego_rel_position.T).T
    return new_position.tolist()