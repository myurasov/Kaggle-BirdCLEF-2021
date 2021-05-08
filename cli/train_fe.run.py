#!/usr/bin/python

import argparse
import os
from pprint import pformat

import pandas as pd
from lib.utils import fix_random_seed
from src.config import c
from src.generator import Generator
from src.services import get_wave_provider
from tqdm import tqdm

# region: read arguments
parser = argparse.ArgumentParser(
    description="Train feature extractor model",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "--in_pickle",
    type=str,
    default="dataset.pickle",
    help="Input .pickle DF file",
)

args = parser.parse_args()
print(f"* Arguments:\n{pformat(vars(args))}")
# endregion

fix_random_seed(c["SEED"])
os.makedirs(c["WORK_DIR"], exist_ok=True)
os.chdir(c["WORK_DIR"])

#
