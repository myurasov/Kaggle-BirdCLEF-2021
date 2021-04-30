#!/usr/bin/python

import argparse
import datetime
import os
from multiprocessing import Pool, cpu_count
from pprint import pformat

import numpy as np
import pandas as pd
from lib.utils import coarsen_number, fix_random_seed, list_indexes
from src.config import c
from src.data_utils import rectify_class_counts
from src.services import get_data_provider
from tqdm import tqdm

# see README.md for details on the dataset creation

# region: read arguments
parser = argparse.ArgumentParser(
    description="Prepare short audio data",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "--out_csv",
    type=str,
    default="short.csv",
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
    to_drop = df[df.rating < min_rating]
    print(f"* Filtered out {to_drop.shape[0]:,} rows where rating < {min_rating}")
    return df.drop(index=to_drop.index)


def _filter_invalid_dates(df):
    """Filter-out invalid dates"""
    to_drop = df[df.date == "0000-00-00"]
    print(f"* Filtered out {to_drop.shape[0]:,} rows with invalid dates")
    return df.drop(index=to_drop.index)


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

# region filter out invalid/bad quality rows

# assign default rating value
df.at[df.rating == 0, "rating"] = args.no_rating_value

# filter by min rating
df = _filter_by_rating(df, args.min_rating)

# filter out invalid dates
df = _filter_invalid_dates(df)

# endregion

# region: convert labels
print("* Converting labels...")

for ix, row in tqdm(df.iterrows()):
    ...


# endregion

# region: add date coarsened to month
print("* Adding coarsened dates...")

months = []
years = []

for ix, row in df.iterrows():
    date_s = row.date

    # they have invented month zero and day zero...
    date_s = date_s.split("-")
    for i in [1, 2]:
        if date_s[i] == "00":
            date_s[i] = "01"
    date_s = "-".join(date_s)

    date = datetime.datetime.strptime(date_s, "%Y-%m-%d")
    months.append(date.month)
    years.append(date.year)

df["_year"] = years
df["_month"] = months
# endregion

# region: add coarsened coords
print("* Adding coarsened lat/lon...")

coarse_lats = []
coarse_lons = []

for ix, row in df.iterrows():
    lat = row.latitude
    lon = row.latitude
    coarse_lats.append(
        coarsen_number(lat, bins=c["GEO_COORDINATES_BINS"], min_val=-90, max_val=90)
    )
    coarse_lons.append(
        coarsen_number(lat, bins=c["GEO_COORDINATES_BINS"], min_val=-180, max_val=180)
    )

df["_lat_coarse"] = coarse_lats
df["_lon_coarse"] = coarse_lons
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
        "type",
        "time",
        "date",
        "author",
        "license",
        "latitude",
        "longitude",
        "common_name",
        "scientific_name",
    ]
)

out_df.to_csv(args.out_csv, index=False)
print(f'* Saved CSV to "{args.out_csv}"')
# endregion
