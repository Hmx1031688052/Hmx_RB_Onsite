from collections import deque
from dataclasses import asdict, dataclass
import random

import numpy as np

from epre_dsac.parameters import agent_par


@dataclass
class FDPITransition:
    state: np.ndarray
    env_input: np.ndarray
    env_map: np.ndarray
    action: np.ndarray
    reward: float
    cost: float
    next_state: np.ndarray
    next_env_input: np.ndarray
    next_env_map: np.ndarray
    terminated: bool
    truncated: bool
    behavior_policy: str
    logp_main: float
    logp_dual: float
    log_is_to_main: float
    log_is_to_dual: float


class Reply_Buffer:
    """Replay buffer with a named FDPI schema and a legacy DSAC mode."""

    def __init__(self, buffer_size, fdpi_enabled=None):
        self.buffer = deque()
        self.buffer_size = int(buffer_size)
        self.fdpi_enabled = (
            bool(agent_par.get("fdpi_enabled", False))
            if fdpi_enabled is None else bool(fdpi_enabled)
        )

    def __len__(self):
        return len(self.buffer)

    def append(self, item):
        if len(self.buffer) >= self.buffer_size:
            self.buffer.popleft()
        if getattr(self, "fdpi_enabled", False) and isinstance(item, dict):
            item = FDPITransition(**item)
        self.buffer.append(item)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        if not getattr(self, "fdpi_enabled", False):
            return self._sample_legacy(batch)

        rows = [asdict(item) if isinstance(item, FDPITransition) else dict(item) for item in batch]
        result = {}
        for name in FDPITransition.__dataclass_fields__:
            values = [row[name] for row in rows]
            if name == "behavior_policy":
                result[name] = np.asarray(values, dtype=np.str_)
            else:
                result[name] = np.asarray(values)
        return result

    @staticmethod
    def _sample_legacy(batch):
        columns = [[] for _ in range(10)]
        for item in batch:
            columns[0].append(item[0][0])
            columns[1].append(item[1])
            columns[2].append(item[2])
            columns[3].append(item[3][0])
            columns[4].append(float(item[4]))
            for index in range(5, 10):
                columns[index].append(item[index])
        return tuple(np.stack(column) for column in columns)
