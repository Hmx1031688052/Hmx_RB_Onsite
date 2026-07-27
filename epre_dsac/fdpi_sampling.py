import math

import numpy as np


def select_episode_behavior_policy(
    train, fdpi_enabled, dual_active, dual_sample_ratio, random_value=None
):
    if not train or not fdpi_enabled or not dual_active:
        return "main"
    if random_value is None:
        random_value = np.random.rand()
    return "dual" if random_value < dual_sample_ratio else "main"


def accumulate_importance_weights(
    behavior_policy,
    logp_main,
    logp_dual,
    cumulative_log_is_to_main,
    cumulative_log_is_to_dual,
    beta,
    min_weight,
    max_weight,
):
    if behavior_policy == "main":
        step_to_main = 0.0
        step_to_dual = float(logp_dual - logp_main)
    elif behavior_policy == "dual":
        step_to_main = float(logp_main - logp_dual)
        step_to_dual = 0.0
    else:
        raise ValueError("behavior_policy must be 'main' or 'dual'")

    to_main = beta * (cumulative_log_is_to_main + step_to_main)
    to_dual = beta * (cumulative_log_is_to_dual + step_to_dual)
    lower, upper = math.log(min_weight), math.log(max_weight)
    return float(np.clip(to_main, lower, upper)), float(np.clip(to_dual, lower, upper))
