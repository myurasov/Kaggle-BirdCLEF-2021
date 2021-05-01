#!/usr/bin/python

# merge datasets (short, long, external)
# add folds
# save class information

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

# region: read arguments
parser = argparse.ArgumentParser(
    description="Create dataset",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "--in_csvs",
    type=str,
    nargs="+",
    default=["short.csv", "long.csv"],
    help="Input CSV files",
)

parser.add_argument(
    "--out_csv",
    type=str,
    default="dataset.csv",
    help="Output CSV file",
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
    df = df.append(curr_df)  # type: ignore
    print(f'* Added "{csv}" with {curr_df.shape[0]} rows')

print(f"* Total {df.shape[0]} rows")

# save
df.to_csv(args.out_csv, index=False)
print(f'* Saved CSV to "{args.out_csv}"')
