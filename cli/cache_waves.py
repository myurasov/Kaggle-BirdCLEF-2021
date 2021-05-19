#!/usr/bin/python

import argparse
import os
from multiprocessing import Pool, cpu_count
from pprint import pformat

import pandas as pd
from lib.utils import fix_random_seed
from src.config import c
from src.generator import Generator
from src.services import get_wave_provider
from tqdm import tqdm

# region: read arguments
parser = argparse.ArgumentParser(
    description="Cache waves",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "--dataset",
    type=str,
    default="dataset.pickle",
    help="Input DF file",
)

args = parser.parse_args()
print(f"* Arguments:\n{pformat(vars(args))}")
# endregion

fix_random_seed(c["SEED"])
os.makedirs(c["WORK_DIR"], exist_ok=True)
os.chdir(c["WORK_DIR"])

#


def _read_whole_file(i):
    wp.get_audio_fragment(
        file_name=filenames[i],
    )


def _read_fragment(i):
    wp.get_audio_fragment(
        file_name=fragments["filename"][i],
        range_seconds=[fragments["_from_s"][i], fragments["_to_s"][i]],
    )


#

df = getattr(pd, f'read_{args.dataset.split(".")[-1].lower()}')(args.dataset)
fragments = df[["filename", "_from_s", "_to_s"]].to_dict("list")
filenames = list(set(fragments["filename"]))

wp = get_wave_provider(c)


print("* Caching whole files...")
with Pool(cpu_count()) as pool:
    list(
        tqdm(
            pool.imap(
                _read_whole_file,
                range(len(filenames)),
            ),
            total=len(filenames),
            smoothing=0,
        )
    )


print("* Caching fragments...")
with Pool(cpu_count()) as pool:
    list(
        tqdm(
            pool.imap(
                _read_fragment,
                range(len(fragments["filename"])),
            ),
            total=len(fragments["filename"]),
            smoothing=0,
        )
    )
