#!/usr/bin/python

import argparse
import os
from multiprocessing import Pool, cpu_count
from pprint import pformat

import numpy as np
import pandas as pd
from lib.utils import list_indexes, fix_random_seed
from src.config import c
from src.data_utils import rectify_class_counts
from src.services import get_data_provider
from tqdm import tqdm

# see README.md for details on the dataset creation

# region: read arguments
parser = argparse.ArgumentParser(
    description="Create short audio dataset",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "--out_csv",
    type=str,
    default="short_dataset.csv",
    help="Output CSV file path",
)

parser.add_argument(
    "--min_rating",
    type=float,
    default=4,
    help="Filter short audio clips only with at least this rating",
)

parser.add_argument(
    "--no_rating_value",
    type=float,
    default=3,
    help="Default rating value for samples without one (=0)",
)

parser.add_argument(
    "--sample_with_stride",
    type=float,
    default=5,
    help="Produce samples by string with this interval [seconds]."
    + "Set to 0 to disable stride sampling.",
)

parser.add_argument(
    "--sample_with_detection",
    action="store_true",
    help="Produce samples from bird song presense detection.",
)

parser.add_argument(
    "--rectify_class_balance",
    type=float,
    default=[1.5, 0.25],
    nargs="+",
    help="Randomly drop rows with too many entries (> mean*[0]). "
    + "Repeat rows with too little entries (< mean*[1]).",
)

args = parser.parse_args()
print(f"* Arguments:\n{pformat(vars(args))}")
# endregion

# region: bootstrap

fix_random_seed(c["SEED"])
os.chdir(c["WORK_DIR"])


def _filter_by_rating(df, min_rating):
    """Filter df by min rating"""
    print(
        f"* Filtered out {df[df.rating < min_rating].shape[0]:,} "
        + f"rows where rating < {min_rating}"
    )

    return df[df.rating >= min_rating]


def _audio_file_path_to_duration(filename):
    return get_data_provider().get_audio_duration(filename)


def _get_audio_file_durations(filenames):
    """Calculate audio files durations"""

    with Pool(cpu_count()) as pool:
        res = list(
            tqdm(
                pool.imap(_audio_file_path_to_duration, filenames),
                total=len(filenames),
                smoothing=0,
            )
        )

    return res


# endregion

# region: read short clips csv
csv_path = os.path.join(c["DATA_DIR"], "competition_data", "train_metadata.csv")
df = pd.read_csv(csv_path)
print(f"* Total {df.shape[0]:,} rows in {csv_path}")
# endregion

# region: ratings

# assign default rating value
df.at[df.rating == 0, "rating"] = args.no_rating_value

# filter by min rating
df = _filter_by_rating(df, args.min_rating)

# endregion

# region: sample fragments

# calc audio files durations
print("* Calculating short files duration...")
df["_duration_s"] = durations = _get_audio_file_durations(list(df.filename))
print(f"* Total short clips time: {sum(durations):,.0f} seconds")

# add from/to cols
df["_from_s"] = [None] * df.shape[0]
df["_to_s"] = [None] * df.shape[0]

# placeholder for output data
out_df_rows = []
out_df_col_ixs = list_indexes(list(df.columns))

# sample with strides
if args.sample_with_stride > 0:

    clip_len_s = c["AUDIO_TARGET_LEN_S"]
    stride_s = args.sample_with_stride

    print(f"* Sampling {clip_len_s}s clips with stride={stride_s}s")

    for ix, row in tqdm(df.iterrows(), total=df.shape[0]):
        clip_duration_s = row["_duration_s"]

        # stride through one file
        for from_s in np.arange(0, clip_duration_s - clip_len_s, stride_s):
            out_df_row = list(row)
            out_df_row[out_df_col_ixs["_from_s"]] = from_s
            out_df_row[out_df_col_ixs["_to_s"]] = from_s + clip_len_s
            out_df_rows.append(out_df_row)


# create output df
out_df = pd.DataFrame(out_df_rows, columns=df.columns)
out_df = out_df.reset_index()
print(f"* Total {len(out_df_rows):,} clips")

# endregion

# region: rectify class balance
if len(args.rectify_class_balance) == 2:
    primary_label_counts = out_df["primary_label"].value_counts()
    mean = np.mean(primary_label_counts)
    max_items = int(mean * args.rectify_class_balance[0])
    min_items = int(mean * args.rectify_class_balance[1])
    print(f"* Rectifying class balance with max={max_items}, min={min_items}...")

    out_df = rectify_class_counts(
        out_df,
        max_items=max_items,
        min_items=min_items,
        class_col="primary_label",
    )

    print(f"* Total {out_df.shape[0]:,} clips")
# endregion

# region: save output df
out_df = out_df.drop(
    columns=[
        "url",
        "license",
        "scientific_name",
        "common_name",
        "author",
    ]
)

out_df.to_csv(args.out_csv, index=False)
print(f'* Saved CSV to "{args.out_csv}"')
# endregion
