#!/usr/bin/python


import argparse
import os
import sys
import unittest
from multiprocessing import Pool, cpu_count
from pprint import pformat

import numpy as np
import pandas as pd
from lib.utils import fix_random_seed, list_indexes, write_json
from src.config import c
from src.generator import Generator
from src.services import get_msg_provider, get_wave_provider
from tqdm import tqdm

# region: read arguments
parser = argparse.ArgumentParser(
    description="Cache waves",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "--in_pickle",
    type=str,
    default="dataset.pickle",
    help="Input pickle DF file",
)

args = parser.parse_args()
print(f"* Arguments:\n{pformat(vars(args))}")
# endregion

fix_random_seed(c["SEED"])
os.makedirs(c["WORK_DIR"], exist_ok=True)
os.chdir(c["WORK_DIR"])

#

generator = Generator(
    df=pd.read_pickle(args.in_pickle),
    wave_provider=get_wave_provider(c),
    msg_provider=None,
    batch_size=1,
    shuffle=True,
    augmentation=None,
)


def _mapping(i):
    _, _, _ = generator.__getitem__(i)


with Pool(cpu_count()) as pool:
    list(
        tqdm(
            pool.imap(
                _mapping,
                range(generator.__len__()),
            ),
            total=generator.__len__(),
            smoothing=0,
        )
    )
