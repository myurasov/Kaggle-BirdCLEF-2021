#!/usr/bin/python

# merge datasets (short, long, external)
# save class information

import argparse
import os
import sys
from pprint import pformat

import numpy as np
import pandas as pd
from lib.utils import fix_random_seed, list_indexes, write_json
from src.config import c
from src.data_utils import add_folds, rectify_class_counts
from src.geo_filter import filters as geo_filters
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
    # default=["long-tst2.csv"],
    default=["short.csv", "long.csv"],
    help="Input CSV files",
)

parser.add_argument(
    "--out",
    type=str,
    default="dataset.pickle",
    help="Output DataFrame file",
)

parser.add_argument(
    "--secondary_label_p",
    type=float,
    default=0.75,
    help="Probability value to assign to secondary labels in one-hots",
)

parser.add_argument(
    "--rectify_class_balance",
    type=float,
    default=[1.5, 0.25],
    nargs="+",
    help="Randomly drop rows with too many entries (> mean*[0]). "
    + 'Repeat rows with too little entries (< mean*[1]). Set to "0" to disable',
)

parser.add_argument(
    "--folds",
    type=int,
    default=5,
    help="Number of folds",
)

parser.add_argument(
    "--geofilter",
    type=str,
    # default="all-500mi-0mo_tolerance.SNE",
    default=None,
    help="Geofilter to use (from src/geo_filters.py)."
    + ' Eg: "all-500mi-1mo_tolerance" or "all-500mi-1mo_tolerance.SNE"',
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
    print(f'* Added "{csv}" with {curr_df.shape[0]:,} rows')  # type: ignore

print(f"* Total rows: {df.shape[0]:,}")

# region: cleanup

#  filter out "bad" sound files,
excluded_df = df[df["filename"].isin(c["EXCLUDE_FILES"])]
df = df.drop(excluded_df.index)
print(f"* Removed {excluded_df.shape[0]} rows with bad files")
df.reset_index(inplace=True, drop=True)

# merge "rocpig1" and "rocpig"
df["_secondary_labels"] = list(
    map(
        lambda x: x.replace("rocpig1", "rocpig") if type(x) == str else x,
        list(df["_secondary_labels"]),
    )
)

# endregion

# region: geofilter

if args.geofilter is not None:
    # remove all primary, extra primary and secondary labels not in geofilter
    # drop rows that do not have any labels left

    # args.geofilter can have "geofilter.SITE" format
    args_geofilter = args.geofilter.split(".")
    geo_filter = geo_filters[args_geofilter[0]]
    if len(args_geofilter) > 1:
        geo_filter = {"_": geo_filter[args_geofilter[1]]}

    gf_classes = set(sum(geo_filter.values(), []))
    print(f"* Geofiltering by {args.geofilter} with {len(gf_classes)} classes...")

    ixs_to_drop = []
    cols = ["_primary_labels", "_extra_primary_labels", "_secondary_labels"]
    df[cols] = df[cols].fillna("")

    for ix, row in tqdm(df[cols].iterrows(), total=df.shape[0]):
        for col in cols:
            row_labels = set(row[col].split(" "))
            row_labels &= gf_classes
            df.at[ix, col] = " ".join(list(row_labels))  # type: ignore

        if set(list(df.loc[ix, cols])) == {""}:  # no labels left - drop row
            ixs_to_drop.append(ix)

    df = df.drop(ixs_to_drop).reset_index(drop=True)
    print(f"* Dropped {len(ixs_to_drop)} rows")

# endregion

# region: rectify class balance

if len(args.rectify_class_balance) == 2:
    mean = np.mean(df["_primary_labels"].value_counts())
    max_items = int(mean * args.rectify_class_balance[0])
    min_items = int(mean * args.rectify_class_balance[1])
    print(f"* Rectifying class balance with max={max_items}, min={min_items}...")

    df = rectify_class_counts(
        df,
        max_items=max_items,
        min_items=min_items,
        class_col="_primary_labels",
    )

    print(f"* Total {df.shape[0]:,} clips")
# endregion

# region: split into folds based on _primary_labels
df = add_folds(df, args.folds, "_primary_labels")
# endregion

# region: read and convert labels

# read and convert primary and secondary labels

classes = set()

for col in [
    "_primary_labels",
    "_secondary_labels",
    "_extra_primary_labels",
]:
    df[[col]] = df[[col]].fillna(value="")
    classes.update(set(" ".join(df[col].unique()).split(" ")))

if "" in classes:
    classes.remove("")

classes = sorted(list(classes))
print(f"* Total classes: {len(classes):,}")

# convert labels to one-hots
print("* Converting labels to one-hots...")

Y = []
class_ixs = list_indexes(classes)

for ix, row in tqdm(df.iterrows(), total=df.shape[0]):
    Y.append(np.zeros((len(classes)), dtype=np.float16))

    # _secondary_labels
    row_labels = row._secondary_labels.split(" ")
    for row_label in row_labels:
        if row_label != "":
            Y[-1][class_ixs[row_label]] = args.secondary_label_p

    # _primary_labels
    row_labels = row._primary_labels.split(" ")
    for row_label in row_labels:
        if row_label != "":
            Y[-1][class_ixs[row_label]] = 1.0

    # _extra_primary_labels - also assign ones there
    row_labels = row._extra_primary_labels.split(" ")
    for row_label in row_labels:
        if row_label != "":
            Y[-1][class_ixs[row_label]] = 1.0

df["_y"] = Y

# endregion

# save metadata
write_json(
    filename=args.out + ".json",
    data={
        "cmd": " ".join(sys.argv),
        "args": vars(args),
        "labels": classes,
    },
)

# save csv
print(f'* Saving to "{args.out}"...')
getattr(df, f'to_{args.out.split(".")[-1].lower()}')(args.out)
