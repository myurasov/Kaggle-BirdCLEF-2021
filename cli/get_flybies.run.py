#!/usr/bin/python
import argparse
import os
from math import atan2, cos, radians, sin, sqrt
from pprint import pformat

import pandas as pd
from src.config import c
from src.data_utils import read_soundscapes_info

# region: read arguments
parser = argparse.ArgumentParser(
    description="Find bird classes that has flown over test sites in test dates",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "--dataset",
    type=str,
    default="dataset-all_m1_r3.pickle",
    help="Input DF file",
)

parser.add_argument(
    "--miles",
    type=float,
    default=500,
    help="Distance in miles",
)

parser.add_argument(
    "--time_tolerance_months",
    type=float,
    default=1,
    help="Extend date range by this much months",
)

parser.add_argument(
    "--only_months",
    type=int,
    default=0,
    help="Filter only by months, ignore years",
)

parser.add_argument(
    "--last_n_years",
    type=int,
    default=0,
    help="Include only last N years. 0 = all.",
)

args = parser.parse_args()
print(f"* Arguments:\n{pformat(vars(args))}")
# endregion


def _str_date_to_ym_date(x):
    x = str(x)
    return int(x[0:4]) * 12 + int(x[4:6])


def _geodist_miles(lat1, lon1, lat2, lon2):
    R = 6373.0 * 1.60934

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    x = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * x


def _add_distance(df, to_lat, to_lon, lat_col="latitude", lon_col="longitude"):

    res_df = df.copy()

    res_df["_distance_"] = list(
        map(
            lambda k: _geodist_miles(
                df.loc[k][lat_col], df.loc[k][lon_col], to_lat, to_lon
            ),
            df.index,
        )
    )

    return res_df


os.chdir(c["WORK_DIR"])

# read dataset
df = getattr(pd, f'read_{args.dataset.split(".")[-1].lower()}')(
    args.dataset
).reset_index(drop=True)

# filter out bad files
df = df[~df["filename"].isin(c["EXCLUDE_FILES"])].reset_index(drop=True)

# add ym dates
df["_ym_date_"] = list(
    map(lambda x: df.loc[x]["_year"] * 12 + df.loc[x]["_month"], df.index)
)

# read soundscapes info
ss_info = read_soundscapes_info(c["COMPETITION_DATA"] + "/test_soundscapes")

# read sites collection dates
dates_df = pd.read_csv(
    c["COMPETITION_DATA"] + "/test_soundscapes/test_set_recording_dates.csv"
)

# convert dates to ym dates (y*12+m)
dates_df["_ym_date_"] = list(map(_str_date_to_ym_date, dates_df["date"]))  # type: ignore

# convert dates to just months
dates_df["_month_"] = list(map(lambda x: (x // 100) % 100, dates_df["date"]))  # type: ignore

classes = set()

for k, v in ss_info.items():

    months = sorted(list(dates_df[dates_df["site"] == k]["_month_"]))  # type: ignore
    dates = sorted(list(dates_df[dates_df["site"] == k]["date"]))  # type: ignore
    ym_dates = sorted(list(dates_df[dates_df["site"] == k]["_ym_date_"]))  # type: ignore

    df = _add_distance(df, ss_info[k]["lat"], ss_info[k]["lon"])

    if args.only_months:
        start_year = 0

        range_desc = (
            f" {months[0]} and {months[-1]} (+-{args.time_tolerance_months}) months"
        )

        if args.last_n_years > 0:
            start_year = 2021 - args.last_n_years + 1
            range_desc += f" in {start_year}-2021 years"

        site_classes = set(
            df[
                (df["_distance_"] <= args.miles)
                & (df["_month"] >= months[0] - args.time_tolerance_months)
                & (df["_month"] <= months[-1] + args.time_tolerance_months)
                & (df["_year"] >= start_year)
            ]["_primary_labels"]
        )
    else:
        range_desc = (
            f" {dates[0]//100} and {dates[-1]//100}"
            + f" (+-{args.time_tolerance_months} months)"
        )

        site_classes = set(
            df[
                (df["_distance_"] <= args.miles)
                & (df["_ym_date_"] >= ym_dates[0] - args.time_tolerance_months)
                & (df["_ym_date_"] <= ym_dates[-1] + args.time_tolerance_months)
            ]["_primary_labels"]
        )

    print(
        f"\n* Within {args.miles} miles from {k} ({ss_info[k]['location']}"
        + f" @ {ss_info[k]['lat']}, {ss_info[k]['lon']}) between"
        + range_desc
        + ":"
    )

    print(
        f"{len(site_classes)} classes:",
        pformat(sorted(list(site_classes)), compact=True),
    )
    classes.update(site_classes)

print(
    f"\n* Total: {len(classes)} classes:",
    pformat(sorted(list(classes)), compact=True),
)
