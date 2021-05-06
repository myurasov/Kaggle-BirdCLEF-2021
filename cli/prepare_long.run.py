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

args = parser.parse_args()
print(f"* Arguments:\n{pformat(vars(args))}")
# endregion

# region: bootstrap
fix_random_seed(c["SEED"])
os.makedirs(c["WORK_DIR"], exist_ok=True)
os.chdir(c["WORK_DIR"])
# endregion

# region: read sounscapes csv
csv_path = os.path.join(
    c["DATA_DIR"], "competition_data", "train_soundscape_labels.csv"
)
df = pd.read_csv(csv_path)
print(f"* Total {df.shape[0]:,} rows in {csv_path}")
# endregion

# region: read soundscape info (coords, rec dates)


def _read_soundscapes_info():
    info = {}
    info_dir = os.path.join(c["DATA_DIR"], "competition_data", "test_soundscapes")

    # read coordinates, location
    for p in glob(os.path.join(info_dir, "*.txt")):
        name = os.path.basename(p)[:3]
        with open(p, "r") as f:
            contents = f.read()
            lat = float(re.findall("Latitude: (.+)\\b", contents)[0])
            lon = float(re.findall("Longitude: (.+)\\b", contents)[0])
            location = re.findall("^.+\n.+", contents)[0].replace("\n", ", ")
        info[name] = {"lat": lat, "lon": lon, "location": location}

    return info


soundscapes_info = _read_soundscapes_info()
# endregion

# region: add info fields

newcols = defaultdict(list)
newcols["_secondary_labels"] = [None] * df.shape[0]
newcols["rating"] = [5.0] * df.shape[0]

for ix, row in tqdm(df.iterrows(), total=df.shape[0]):
    # audio file path/name

    file_glob = os.path.join(
        c["DATA_DIR"],
        "competition_data",
        "train_soundscapes",
        f"{row.audio_id}*.ogg",
    )

    file_path = glob(file_glob)
    assert len(file_path) == 1
    file_path = file_path[0]
    file_name = os.path.basename(file_path)
    newcols["filename"].append(file_name)

    # date
    date_s = re.findall("_(\\d+).ogg$", file_name)[0]
    date = datetime.datetime.strptime(date_s, "%Y%m%d")
    newcols["_month"].append(date.month)
    newcols["_year"].append(date.year)

    # from/to
    newcols["_from_s"].append(row.seconds - 5)
    newcols["_to_s"].append(row.seconds)

    # lat/lon
    newcols["latitude"].append(soundscapes_info[row.site]["lat"])
    newcols["longitude"].append(soundscapes_info[row.site]["lon"])

    # labels
    newcols["_primary_labels"].append(" ".join(row.birds.split(" ")))

for k, v in newcols.items():
    df[k] = v

# endregion

# region: save output df

df["_source"] = ["long"] * df.shape[0]

df = df[
    [
        "filename",
        "_primary_labels",
        "_secondary_labels",
        "_from_s",
        "_to_s",
        "_year",
        "_month",
        "latitude",
        "longitude",
        "rating",
        "_source",
        "row_id",
    ]
]

df.to_csv(args.out_csv, index=False)
print(f'* Saved CSV to "{args.out_csv}"')

# endregion
