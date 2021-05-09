#!/usr/bin/python

import argparse
import os
import shutil
import sys
from pprint import pformat

import pandas as pd
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
from tqdm import tqdm

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
    default="msg_enb0",
    help="Model",
)

parser.add_argument(
    "--val_fold",
    type=float,
    default=1,
    help="Validation fold. Use float value <1 for a val split rather than fold.",
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
    type=int,
    default=1,
    help="Number of generator threads",
)

args = parser.parse_args()
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
save_keras_model(model, f"{checkpoint_path}/model.png", dpi=75, rankdir="LR")
# endregion

# region: save train run metadata
train_meta_file = f"{checkpoint_path}/meta.json"

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

try:
    input_shape = model.get_layer("i_msg").input_shape[0][1:]
    input_type = "melspectrogram"

    train_g = Generator(
        train_df,
        wave_provider=get_wave_provider(c),
        msg_provider=get_msg_provider(c),
        batch_size=args.batch,
        shuffle=True,
        augmentation=None,
        msg_as_rgb=3 == input_shape[-1],
        rating_as_sw=True,
        geo_coordinates_bins=c["GEO_COORDINATES_BINS"],
    )

    val_g = Generator(
        val_df,
        wave_provider=get_wave_provider(c),
        msg_provider=get_msg_provider(c),
        batch_size=args.batch,
        shuffle=False,
        augmentation=None,
        msg_as_rgb=3 == input_shape[-1],
        rating_as_sw=True,
        geo_coordinates_bins=c["GEO_COORDINATES_BINS"],
    )


except ValueError:
    raise RuntimeError("Unsupported input type")

print(f"* Model input: {input_type} of size {input_shape}")

# endregion

# region: callbacks

callbacks = []

callbacks.append(
    keras.callbacks.EarlyStopping(
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

callbacks.append(
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=args.lr_factor,
        patience=args.lr_patience,
        min_lr=1e-9,
        verbose=1,
    )
)

callbacks.append(
    keras.callbacks.ModelCheckpoint(
        checkpoint_path + ".h5",
        verbose=1,
        save_freq="epoch",
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=False,
    )
)

# endregion

# region: finally do something useful

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=args.lr),
    loss="binary_crossentropy",
)

steps_per_epoch = (
    args.samples_per_epoch // args.batch if args.samples_per_epoch > 0 else None
)

model.fit(
    x=train_g,
    validation_data=val_g,
    epochs=args.epochs,
    steps_per_epoch=steps_per_epoch,
    callbacks=callbacks,
    use_multiprocessing=args.multiprocessing > 1,
    workers=args.multiprocessing,
    max_queue_size=1,
    verbose=1,
)

# endregion
