import argparse
import os
from multiprocessing import Pool, cpu_count
from pprint import pformat

import pandas as pd
from tqdm import tqdm

from src.config import c
from src.services import get_data_provider, get_torch_melspectrogram

# region: read arguments
parser = argparse.ArgumentParser(
    description="Precache decoded and resized images",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "--sr",
    type=float,
    default=32000,
    help="Target sample rate",
)

args = parser.parse_args()
print(f"* Arguments:\n{pformat(vars(args))}")
# endregion

os.chdir(c["WORK_DIR"])

#

dp = get_data_provider()
print(dp.get_audio_fragment("XC6671.ogg"))

ms = get_torch_melspectrogram()
print(ms)
