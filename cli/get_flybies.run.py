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

# read soundscapes info
ss_info = read_soundscapes_info(c["COMPETITION_DATA"] + "/test_soundscapes")

# read sites collection dates
dates_df = pd.read_csv(
    c["COMPETITION_DATA"] + "/test_soundscapes/test_set_recording_dates.csv"
)

# convert dates to ym dates (y*12+m)
dates_df["date"] = list(map(_str_date_to_ym_date, dates_df["date"]))  # type: ignore

classes = set()

for k, v in ss_info.items():
    dates = sorted(list(dates_df[dates_df["site"] == k]["date"]))  # type: ignore
    ss_info[k]["ymdate_range"] = [dates[0], dates[-1]]

    print(
        f"* Within {args.miles} miles from {k} ({ss_info[k]['location']}"
        + f" @ {ss_info[k]['lat']}, {ss_info[k]['lon']}):"
    )

    df_dist = _add_distance(df, ss_info[k]["lat"], ss_info[k]["lon"])
    classes = set(df_dist[df_dist["_distance_"] <= args.miles]["_primary_labels"])
    classes.update(classes)

    print(f"{len(classes)} classes:", classes)

print(f"\nTotal: {len(classes)} classes:", classes)
