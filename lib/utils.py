import hashlib
import json
import math
import os
import random
import shutil
from collections import namedtuple
from pathlib import Path

import IPython
import numpy as np
import tensorflow as tf
from tensorflow import keras


def create_dir(dir, remove=True):
    if remove:
        shutil.rmtree(dir, ignore_errors=True)
    os.makedirs(dir, exist_ok=True)


def rename_dir(dir1, dir2, remove_dir2=True):
    if remove_dir2:
        shutil.rmtree(dir2, ignore_errors=True)
    os.rename(dir1, dir2)


def dict_to_struct(d):
    return namedtuple("Struct", d.keys())(*d.values())


def fix_random_seed(seed=777):

    # set in Dokerfile
    # os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def list_indexes(list, cols=None):
    """
    Creates a dictionary mapping values to indexes
    """
    if cols is None:
        cols = list
    return dict([(x, list.index(x)) for x in cols])


def md5_file(path):
    """
    Calculate file MD5 hash
    """

    with open(path, "rb") as f:
        hash = hashlib.md5()

        while chunk := f.read(2 << 20):
            hash.update(chunk)

    return hash.hexdigest()


def show_keras_model(model: keras.Model, expand_nested=False):
    """Display model structure in notebook"""
    return IPython.display.SVG(
        keras.utils.model_to_dot(
            model=model,
            show_shapes=True,
            show_dtype=True,
            show_layer_names=True,
            expand_nested=expand_nested,
        ).create(
            prog="dot",
            format="svg",
        )
    )


def save_keras_model(model, filename="model.svg", expand_nested=False, **kwargs):
    """Save Keras model visualization to image file"""

    dot = keras.utils.model_to_dot(
        model=model,
        show_shapes=True,
        show_dtype=True,
        show_layer_names=True,
        expand_nested=expand_nested,
        **kwargs,
    )

    with open(filename, "wb+") as f:
        f.write(
            getattr(
                dot,
                "create_" + filename[-3:].lower(),
            )()
        )


def bin_number(number, bins=10, val_range=[0.0, 1.0]):
    """Bin a float number"""
    number -= val_range[0]
    number = number / (val_range[1] - val_range[0])
    return int(max(0, min(math.floor(number * bins), bins - 1)))


def coarsen_number(number, bins=10, val_range=[0.0, 1.0]):
    """Coarsen a float number"""
    number -= val_range[0]
    number = number / (val_range[1] - val_range[0])
    bin_n = max(0, min(round(number * bins), bins))
    return (bin_n / bins) * (val_range[1] - val_range[0]) + val_range[0]


def read_json(filename):
    json.loads(Path(filename).read_text())


def write_json(data, filename):
    print(
        json.dumps(data, indent=4),
        file=open(filename, "w"),
    )
