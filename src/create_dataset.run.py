import argparse
import os
from multiprocessing import Pool, cpu_count
from pprint import pformat

import pandas as pd
from tqdm import tqdm

from src.config import c
from src.services import get_data_provider, get_torch_melspectrogram

# dataset creation:

# - read train_metadata.csv
# - select only labels with good rating
# - slice clips into fragments with <stride> and <length> /
#       cut fragments based on detection model
# - convert coordinates into orthogonal basis,
#       bin them to a coarse grid (10x10?)
# - add date coarsened up to season (month?, 1/8 of y?)
# - add secondary labels

# - read train_soundscape_labels.csv
# - add date coarsened date
# - add coarsened coordinates
# - assume rating is '5' (?)
# - assume all labels are primary (?)

# - add folds

# training:

# - use rating as sample weight (?)
# - use secondary labels with label value < 1 and linear activation (?)
# - do something about class imbalance
# - do augmentation with mixing of random fragments

# region: read arguments
parser = argparse.ArgumentParser(
    description="Create dataset",
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
