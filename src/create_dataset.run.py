import argparse
import os
from pprint import pformat

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

args = parser.parse_args()
print(f"* Arguments:\n{pformat(vars(args))}")
# endregion

os.chdir(c["WORK_DIR"])
