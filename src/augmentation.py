# augmentation config

c = {}

c["v0"] = {
    "msg.random_power": {"chance": 0.5, "min_power": 0.5, "max_power": 3},
}

c["v1"] = {
    "wave.same_class_mixing": {
        "chance": 1.0,
        # coefficients to multiply samples to
        # (randomly sampled in the provided ranges)
        "coeffs": [
            [0.75, 1],
            [0.5, 0.75],
            [0.25, 0.5],
            # [0.125, 0.25],
            # [0.0625, 0.125],
        ],
        "labels": True,  # also mix labels
    },
}
