import argparse
import os
from multiprocessing import Pool, cpu_count
from pprint import pformat

import pandas as pd
from tqdm import tqdm

from src.config import c 

# region: read arguments
parser = argparse.ArgumentParser(
    description="Precache decoded and resized images",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "--src_dirs",
    type=str,
    default=c["SRC_AUDIO_DIRS"],
    nargs="+",
    help="List of input dirs/globs with audio files",
)

parser.add_argument(
    "--sr",
    type=float,
    default=22500,
    help="Target sample rate",
)



args = parser.parse_args()
print(f"* Arguments:\n{pformat(vars(args))}")
# endregion

os.chdir(c["WORK_DIR"])
df = pd.read_csv(args.in_csv)

g = Generator(
    df=df,
    batch_size=1,
    shuffle=False,
    zoom=args.zoom,
    augmentation_options=None,
    image_output_size=tuple(args.size),
)


def _mapping(i):
    g.__getitem__(i)


with Pool(cpu_count()) as pool:
    list(
        tqdm(
            pool.imap(
                _mapping,
                range(df.shape[0]),
            ),
            total=df.shape[0],
            smoothing=0,
        )
    )