#!/usr/bin/python

import argparse
import os
from multiprocessing import Pool, cpu_count
from pprint import pformat

import pandas as pd
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
    default="short.csv",
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
    help="Default rating value for no rating",
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
df = pd.read_csv(csv_path, index_col="filename")
print(f"* Total {df.shape[0]} rows in {csv_path}")

# assign default rating value
df.at[df.rating == 0, "rating"] = args.no_rating_value

# filter by min rating
df = _filter_by_rating(df, args.min_rating)

# calc audio files durations
print("* Calculating short files duration...")
durations = _get_audio_file_durations(list(df.index))
print(f"* Total short clips time: {sum(durations):,.0f} seconds")

#

# df_out = pd.DataFrame()
# df["filename"] = list(df.index)
# df["duration_s"] = durations
# df.set_index("filename")
# df.to_csv(os.path.join(c["WORK_DIR"], "_short.csv"), index=True)
