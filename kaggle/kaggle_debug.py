import os
from src.config import c

# from src.kaggle_utils import c

# predict

csv_path = os.path.join(c["COMPETITION_DATA"], "train_soundscape_labels.csv")

print(csv_path)
