#!/usr/bin/python

import argparse
import os
from pprint import pformat

import pandas as pd
from lib.utils import create_dir, fix_random_seed, read_json, save_keras_model
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

parser.add_argument(
    "--batch",
    type=int,
    default=2,
    help="Batch size",
)

parser.add_argument(
    "--epochs",
    type=int,
    default=100,
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
    "--multiprocessing",
    type=int,
    default=1,
    help="Number of generator threads",
)

parser.add_argument(
    "--amp",
    type=int,
    default=0,
    help="Enable AMP?",
)

parser.add_argument(
    "--aug",
    type=int,
    default=0,
    help="Augmentation level",
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
