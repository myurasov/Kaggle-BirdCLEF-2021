#!/usr/bin/python


import argparse
import datetime
import os
import re
from glob import glob
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

newcols = {
    "_to_s": [],
    "_year": [],
    "_month": [],
    "_from_s": [],
    "filename": [],
    "_duration_s": [],
    "_lat_coarse": [],
    "_lon_coarse": [],
}

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
    newcols["_year"].append(date.month)

    # duration
    # newcols["_duration_s"].append(get_data_provider().get_audio_duration(file_path))
    newcols["_duration_s"].append(600.0)  # !!!

    # from/to
    newcols["_from_s"].append(row.seconds - 5)
    newcols["_to_s"].append(row.seconds)

    # lat/lon
    newcols["_lat_coarse"].append(
        coarsen_number(
            soundscapes_info[row.site]["lat"],
            bins=c["GEO_COORDINATES_BINS"],
            min_val=-90,
            max_val=90,
        )
    )
    newcols["_lon_coarse"].append(
        coarsen_number(
            soundscapes_info[row.site]["lon"],
            bins=c["GEO_COORDINATES_BINS"],
            min_val=-180,
            max_val=180,
        )
    )

for k, v in newcols.items():
    df[k] = v

# endregion

# region: save output df
df = df.drop(columns=[])
df.to_csv(args.out_csv, index=False)
print(f'* Saved CSV to "{args.out_csv}"')
# endregion
