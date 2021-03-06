#!/usr/bin/python

import argparse
import datetime
import os
import re
from collections import defaultdict
from glob import glob
from pprint import pformat

import pandas as pd
from lib.utils import fix_random_seed
from src.config import c
from src.data_utils import normalize_soundscapes_df, read_soundscapes_info
from tqdm import tqdm

# see README.md for details on the dataset creation

# region: read arguments
parser = argparse.ArgumentParser(
    description="Prepare soundscapes audio data",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "--out_csv",
    type=str,
    default="long.csv",
    help="Output CSV file path",
)

parser.add_argument(
    "--split_multilabel",
    type=int,
    default=1,
    help="Split multi-label items into multiple rows",
)

parser.add_argument(
    "--in_csv",
    type=str,
    default="train_soundscape_labels.csv",
    help="Split multi-label items into multiple rows",
)

args = parser.parse_args()
print(f"* Arguments:\n{pformat(vars(args))}")
# endregion

# region: bootstrap
fix_random_seed(c["SEED"])
os.makedirs(c["WORK_DIR"], exist_ok=True)
os.chdir(c["WORK_DIR"])
# endregion

# region: read sounscapes csv and info
csv_path = os.path.join(c["COMPETITION_DATA"], args.in_csv)
df = pd.read_csv(csv_path)
df.audio_id = df.audio_id.astype("str")
print(f"* Total {df.shape[0]:,} rows in {csv_path}")

# add dummy 'birds' column for test file
if "birds" not in df.columns:
    df["birds"] = [""] * df.shape[0]

soundscapes_info = read_soundscapes_info(
    os.path.join(
        c["COMPETITION_DATA"],
        "test_soundscapes",
    )
)
# endregion

# region: split multilabel rows into separate single-label ones

if args.split_multilabel:

    # other labels will be stored in '_extra_primary_labels' column
    if "_extra_primary_labels" not in df:
        df["_extra_primary_labels"] = ""

    extra_df = pd.DataFrame()

    print("* Splitting multilabel rows...")
    n_extra_rows = 0

    for ix, row in tqdm(df.iterrows(), total=df.shape[0]):
        if " " in row.birds:
            birds = row.birds.split(" ")

            if len(birds) > 1:
                n_extra_rows += len(birds) - 1
                df = df.drop(ix)

                for bird in birds:
                    row.birds = bird
                    # save other labels just in case
                    row._extra_primary_labels = " ".join(
                        sorted(list(set(birds) ^ set([bird])))
                    )
                    extra_df = extra_df.append(row, ignore_index=True)

    if extra_df.shape[0] > 0:
        df = df.append(extra_df[df.columns], ignore_index=True)  # type: ignore

# endregion

# region: add info fields

print("* Normalizing to match dataset format...")

df = normalize_soundscapes_df(
    df,
    seconds=5,
    rating=5.0,
    source="long",
    quiet=False,
)

# endregion

# region: save output df

df["_source"] = ["long"] * df.shape[0]

# add standard columns
for col in c["DATASET_COLS"]:
    if col not in df:
        df[col] = ""

df = df[c["DATASET_COLS"] + ["row_id"]]

df.to_csv(args.out_csv, index=False)
print(f'* Saved CSV to "{args.out_csv}"')

# endregion
