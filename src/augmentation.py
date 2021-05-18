# augmentation levels config

import numpy as np


def chance(probability):
    return np.random.uniform(0, 1) <= probability


def msg_random_power(max_power):
    if chance(0.33):
        return np.random.uniform(0.5, max_power)
    return max_power


c = {}

c["v0"] = {
    "msg.random_power": msg_random_power,
}
