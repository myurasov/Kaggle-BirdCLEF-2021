#!/usr/bin/python

import argparse
import os
import shlex
import sys
from pprint import pformat

import numpy as np
import pandas as pd
import tensorflow_addons as tfa
from lib.keras_tb_logger import TensorBoard_Logger, gpu_temp_logger, lr_logger
from lib.utils import (
    create_dir,
    fix_random_seed,
    read_json,
    save_keras_model,
    write_json,
)
from src.config import c
from src.generator import Generator
from src.models import build_model
from src.services import get_msg_provider, get_wave_provider
from tensorflow import keras

# args for debugging
_debug_args = None

# region: read arguments
parser = argparse.ArgumentParser(
    description="Train feature extractor model",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "--dataset",
    type=str,
    default="dataset.pickle",
    help="Input .pickle or .csv DF file",
)

parser.add_argument(
    "--run",
    type=str,
    default="tst",
    help="Run name",
)

parser.add_argument(
    "--model",
    type=str,
    default="msg_enb0_imagenet",
    help="Model",
)

parser.add_argument(
    "--monitor_metric",
    type=str,
    default="val_f1_score",
    help="Metric to monitor for loss scaling and best model selection",
)

parser.add_argument(
    "--val_fold",
    type=float,
    default=1,
    help="Validation fold. Use float value <1 for a val split rather than fold.",
)

parser.add_argument(
    "--preload_val_data",
    type=int,
    default=1,
    help="Preload validation generator into memory to speed up validation runs.",
)

# TODO
# parser.add_argument(
#     "--aug",
#     type=int,
#     default=0,
#     help="Augmentation level",
# )

parser.add_argument(
    "--batch",
    type=int,
    default=32,
    help="Batch size",
)

parser.add_argument(
    "--epochs",
    type=int,
    default=100,
)

parser.add_argument(
    "--samples_per_epoch",
    type=int,
    default=0,
    help="Number of samples per each epoch. 0 = all available samples.",
)

parser.add_argument(
    "--early_stop_patience",
    type=int,
    default=10,
    help="Stop after N epochs val accuracy is not improving",
)

parser.add_argument(
    "--lr",
    type=float,
    default=1e-3,
    help="Inital LR",
)


parser.add_argument(
    "--lr_factor",
    type=float,
    default=0.2,
    help="Factor by which LR is multiplied after unfreezing base modela nd on plateau.",
)

parser.add_argument(
    "--lr_patience",
    type=int,
    default=3,
    help="LR reduction patience",
)

parser.add_argument(
    "--amp",
    type=int,
    default=0,
    help="Enable AMP?",
)

parser.add_argument(
    "--multiprocessing",
    type=str,
    default="1",
    help='Number of generator threads and workers in "<threads>x<workers>" format',
)

parser.add_argument(
    "--weight_by_rareness",
    type=int,
    default=1,
    help="Use primary label rareness for sample weighting",
)

args = parser.parse_args()
if _debug_args is not None:
    args = parser.parse_args(shlex.split(_debug_args))

print(f"* Arguments:\n{pformat(vars(args))}")
# endregion

# region: bootstrap
fix_random_seed(c["SEED"])
os.makedirs(c["WORK_DIR"], exist_ok=True)
os.chdir(c["WORK_DIR"])

# turn amp on
if args.amp:
    print("* Using AMP")
    keras.mixed_precision.set_global_policy("mixed_float16")

# load dataset and metadata
df = getattr(pd, f'read_{args.dataset.split(".")[-1].lower()}')(args.dataset)
meta = read_json(args.dataset + ".json")
# endregion

# region: create train/val dataframes
if args.val_fold < 1:
    # split into sets based on fraction for val
    df = df.sample(frac=1).reset_index(drop=True)
    val_df = df[: int(args.val_fold * df.shape[0])]
    train_df = df[val_df.shape[0] :]
else:
    # split into sets based on folds
    val_df = df[df._fold == int(args.val_fold)]
    train_df = df[df._fold != int(args.val_fold)]
    args.run = f"{args.run}.fold_{args.val_fold:.0f}"

assert val_df.shape[0] + train_df.shape[0] == df.shape[0]
train_df.reset_index(inplace=True, drop=True)
val_df.reset_index(inplace=True, drop=True)
print(f"* Training set size: {train_df.shape[0]}")
print(f"* Validation set size: {val_df.shape[0]}")
# endregion

# region: prepare paths
td_dir = f"/app/.tensorboard/{args.run}"
create_dir(td_dir, remove=True)

checkpoint_path = f"{c['WORK_DIR']}/models/{args.run}"
create_dir(checkpoint_path, remove=True)
# endregion

# region: create model
model = build_model(args.model, len(meta["labels"]))
print(f"* Model output size: {model.output.shape[1]}")

# save picture
save_keras_model(model, f"{checkpoint_path}/{args.run}.png", dpi=75, rankdir="LR")
# endregion

# region: save train run metadata
train_meta_file = f"{checkpoint_path}/{args.run}.json"

write_json(
    {
        "args": vars(args),
        "cmd": " ".join(sys.argv),
        "labels": meta["labels"],
    },
    train_meta_file,
)

print(f"* Saved train run metadata to {train_meta_file}")
# endregion

# region: create generators

input_shape = None
input_type = None
train_g = None
val_g = None
wave_p = None
msg_p = None

try:
    input_shape = model.get_layer("i_msg").input_shape[0][1:]
    input_type = "melspectrogram"
    wave_p = get_wave_provider(c)
    msg_p = get_msg_provider(c, n_mels=input_shape[0], time_steps=input_shape[1])

    train_g = Generator(
        df=train_df,
        shuffle=True,
        augmentation=None,
        rating_as_sw=True,
        rareness_as_sw=args.weight_by_rareness > 0,
        msg_provider=msg_p,
        wave_provider=wave_p,
        batch_size=args.batch,
        msg_as_rgb=3 == input_shape[-1],
        geo_coordinates_bins=c["GEO_COORDINATES_BINS"],
    )

    val_g = Generator(
        df=val_df,
        shuffle=False,
        augmentation=None,
        rating_as_sw=False,
        rareness_as_sw=False,
        msg_provider=msg_p,
        wave_provider=wave_p,
        msg_as_rgb=3 == input_shape[-1],
        geo_coordinates_bins=c["GEO_COORDINATES_BINS"],
        batch_size=val_df.shape[0] if args.preload_val_data else args.batch,
    )


except ValueError:
    raise RuntimeError("Unsupported input type")

print(f"* Model input: {input_type} of size {input_shape}")

val_x, val_y, val_sw = None, None, None
if args.preload_val_data:
    print("* Preloading validation data...")
    val_x, val_y, val_sw = val_g.__getitem__(0)

# endregion

# region: callbacks

callbacks = []

callbacks.append(
    keras.callbacks.EarlyStopping(
        monitor=args.monitor_metric,
        patience=args.early_stop_patience,
        restore_best_weights=True,
        verbose=1,
    )
)

callbacks.append(
    TensorBoard_Logger(
        log_dir=td_dir,
        histogram_freq=0,
        loggers=[
            lr_logger,
            gpu_temp_logger,
        ],
    )
)

monitoring_mode = "min" if args.monitor_metric.find("loss") else "max"
print(f"* Monitoring {args.monitor_metric} in {monitoring_mode.upper()} mode")

callbacks.append(
    keras.callbacks.ReduceLROnPlateau(
        monitor=args.monitor_metric,
        mode=monitoring_mode,
        factor=args.lr_factor,
        patience=args.lr_patience,
        min_lr=1e-6,
        verbose=1,
    )
)


callbacks.append(
    keras.callbacks.ModelCheckpoint(
        checkpoint_path + f"/{args.run}.h5",
        monitor=args.monitor_metric,
        mode=monitoring_mode,
        save_freq="epoch",
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
    )
)

# endregion

# region: finally do something useful

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=args.lr),
    loss="binary_crossentropy",
    metrics=[
        # TODO: replace with https://www.kaggle.com/shonenkov/competition-metrics
        tfa.metrics.F1Score(
            num_classes=len(meta["labels"]),
            threshold=0.5,
            average="micro",
        ),
    ],
)

# calc steps (batches) per epoch
steps_per_epoch = (
    args.samples_per_epoch // args.batch if args.samples_per_epoch > 0 else None
)

# raise exceptions on all errors
np.seterr(all="raise")

# multiprocessing options
if "x" not in args.multiprocessing:
    # use queue of 1 by default
    args.multiprocessing += "x1"
mp_workers, mp_queue = list(map(int, args.multiprocessing.split("x")))

print(f"* Using {mp_workers} workers and queue of {mp_queue}")

model.fit(
    x=train_g,
    validation_data=(val_x, val_y, val_sw) if args.preload_val_data else val_g,
    epochs=args.epochs,
    steps_per_epoch=steps_per_epoch,
    callbacks=callbacks,
    use_multiprocessing=mp_workers > 1,
    workers=mp_workers,
    max_queue_size=mp_queue,
    verbose=1,
)

# endregion
