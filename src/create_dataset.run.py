import argparse
import os
from pprint import pformat

import pandas as pd

from src.config import c

# see README.md for details on the dataset creation

# region: read arguments
parser = argparse.ArgumentParser(
    description="Create dataset",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "--out_csv",
    type=str,
    default="dataset.csv",
)

parser.add_argument(
    "--short_min_rating",
    type=float,
    default=2.5,
)

args = parser.parse_args()
print(f"* Arguments:\n{pformat(vars(args))}")
# endregion

os.chdir(c["WORK_DIR"])

#

df = pd.read_csv(
    os.path.join(
        c["DATA_DIR"],
        "competition_data",
        "train_metadata.csv",
    )
)
