#!/usr/bin/python

# merge datasets (short, long, external)
# add folds
# save class information

import argparse
import os
import sys
from pprint import pformat

import numpy as np
import pandas as pd
from lib.utils import fix_random_seed, list_indexes, write_json
from src.config import c
from tqdm import tqdm

# region: read arguments
parser = argparse.ArgumentParser(
    description="Create dataset",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "--in_csvs",
    type=str,
    nargs="+",
    default=["short.csv", "long.csv"],
    help="Input CSV files",
)

parser.add_argument(
    "--out_pickle",
    type=str,
    default="dataset.pickle",
    help="Output pickle file",
)

parser.add_argument(
    "--secondary_label_p",
    type=float,
    default=0.75,
    help="Probability value to assign to secondary labels in one-hots",
)

args = parser.parse_args()
print(f"* Arguments:\n{pformat(vars(args))}")
# endregion

# region: bootstrap
fix_random_seed(c["SEED"])
os.makedirs(c["WORK_DIR"], exist_ok=True)
os.chdir(c["WORK_DIR"])
# endregion

# read csvs
df = pd.DataFrame()
for csv in args.in_csvs:
    curr_df = pd.read_csv(csv)
    df = df.append(curr_df, ignore_index=True)  # type: ignore
    print(f'* Added "{csv}" with {curr_df.shape[0]:,} rows')

print(f"* Total rows: {df.shape[0]:,}")

# region: read and convert labels

# read primary and secondary labels

labels = set()

for col in ["_primary_labels", "_secondary_labels"]:
    df[[col]] = df[[col]].fillna(value="")
    labels.update(set(" ".join(df[col].unique()).split(" ")))

if "" in labels:
    labels.remove("")

labels = sorted(list(labels))
print(f"* Total labels: {len(labels):,}")

# convert to one-hots
print("* Converting labels to one-hots...")

Y_labels = []
labels_ixs = list_indexes(labels)

for ix, row in tqdm(df.iterrows(), total=df.shape[0]):
    Y_labels.append(np.zeros((len(labels)), dtype=np.float16))

    # secondary
    row_labels = row._secondary_labels.split(" ")
    for row_label in row_labels:
        if row_label != "":
            Y_labels[-1][labels_ixs[row_label]] = args.secondary_label_p

    # primary
    row_labels = row._primary_labels.split(" ")
    for row_label in row_labels:
        if row_label != "":
            Y_labels[-1][labels_ixs[row_label]] = 1

df["_Y_labels"] = Y_labels

# endregion

# save metadata
write_json(
    filename=args.out_pickle + ".json",
    data={
        "cmd": " ".join(sys.argv),
        "args": vars(args),
        "labels": labels,
    },
)

# save csv
print(f'* Saving to "{args.out_pickle}"...')
df.to_pickle(args.out_pickle)
