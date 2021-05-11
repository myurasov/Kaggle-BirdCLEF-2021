import os

import pandas as pd
from src.config import c

# from src.kaggle_utils import c

# predict

csv_path = os.path.join(c["COMPETITION_DATA"], "train_soundscape_labels.csv")
df = pd.read_csv(csv_path)
print(df)


os.
