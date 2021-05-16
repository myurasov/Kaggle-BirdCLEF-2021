# augmentation levels config

import numpy as np


def _msg_random_power_fn(max_power, chance):
    if np.random.uniform(0, 1) <= chance:
        return np.random.uniform(0.5, max_power)
    return max_power


c = {}

c["v0"] = {
    "msg.random_power.fn": lambda max_power: _msg_random_power_fn(
        max_power, chance=0.33
    )
}
