#!/usr/bin/python


import argparse
import datetime
from glob import glob
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
# TODO
# endregion

# region: add info fields

newcols = {
    "filename": [],
    "_duration_s": [],
    "_from_s": [],
    "_to_s": [],
}

for ix, row in tqdm(df.iterrows(), total=df.shape[0]):
    # audio file

    file_glob = os.path.join(
        c["DATA_DIR"],
        "competition_data",
        "train_soundscapes",
        f"{row.audio_id}*.ogg",
    )

    file_path = glob(file_glob)
    assert len(file_path) == 1
    file_path = file_path[0]
    newcols["filename"].append(os.path.basename(file_path))

    # duration
    newcols["_duration_s"].append(get_data_provider().get_audio_duration(file_path))

    # from/to
    newcols["_from_s"].append(row.seconds - 5)
    newcols["_to_s"].append(row.seconds)

for k, v in newcols.items():
    df[k] = v

# endregion

# region: save output df
df = df.drop(columns=[])
df.to_csv(args.out_csv, index=False)
print(f'* Saved CSV to "{args.out_csv}"')
# endregion
