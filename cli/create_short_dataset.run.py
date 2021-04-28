#!/usr/bin/python

import argparse
import os
from multiprocessing import Pool, cpu_count
from pprint import pformat

import numpy as np
import pandas as pd
from lib.utils import list_indexes
from src.config import c
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
    type=bool,
    default=0,
    help="Produce samples from bird song presense detection.)",
)


args = parser.parse_args()
print(f"* Arguments:\n{pformat(vars(args))}")
# endregion

# bootstrap
os.chdir(c["WORK_DIR"])


def _filter_by_rating(df, min_rating):
    """Filter df by min rating"""
    print(
        f"* Filtered out {df[df.rating < min_rating].shape[0]} "
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


# read short audios metadata csv
csv_path = os.path.join(c["DATA_DIR"], "competition_data", "train_metadata.csv")
df = pd.read_csv(csv_path)
print(f"* Total {df.shape[0]} rows in {csv_path}")

# assign default rating value
df.at[df.rating == 0, "rating"] = args.no_rating_value

# filter by min rating
df = _filter_by_rating(df, args.min_rating)

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

# sample with stride
if args.sample_with_stride > 0:

    clip_len_s = c["AUDIO_TARTGET_LEN_S"]
    stride_s = args.sample_with_stride

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

# balance: randomly drop too large classes


# save outoput df

out_df = out_df.drop(
    columns=[
        "url",
        "license",
        "scientific_name",
        "common_name",
        "author",
    ]
)

out_df.to_csv(os.path.join(c["WORK_DIR"], "short_dataset.csv"), index=False)
