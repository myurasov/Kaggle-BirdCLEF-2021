import numpy as np
import pandas as pd
from lib.utils import read_json
from src.data_utils import (
    normalize_soundscapes_df,
)

from src.config import c

IN_CSV = c["COMPETITION_DATA"] + "/train_soundscape_labels.csv"
MODEL = c["WORK_DIR"] + "/models/B1_nrsw_2/B1_nrsw_2.h5"

# metadata from model training run
meta = read_json(MODEL.replace(".h5", ".json"))

# prepare soundscapes df
df = pd.read_csv(IN_CSV)
df = normalize_soundscapes_df(df, quiet=True, seconds=5)

Y_pred = np.load("/app/sandbox/Y_pred.npy")
df["_y_pred"] = list(map(lambda x: x, Y_pred))


def boost_multiple_occurences(
    df,
    labels,
    pred_col="_y_pred",
    out_col="_y_pred",
    boost_coef=1.5,
    max_boost_coef=1.5 * 1.5,
    threshold=0.5,
):
    """
    Boost predictions in file:
        - if something occured once, multiply that class by boost_coef
        - if something occured more than once - keep multiplying until
            boost_coef reaches max_boost_coef
    """

    def _compute_boost_matrix(y_preds, labels, threshold, boost_coef, max_boost_coef):

        nocall_ix = labels.index("nocall")
        boost_matrix = np.ones((len(meta["labels"])), dtype=np.float64)

        for p in y_preds:
            boost_matrix = boost_matrix * np.where(p > threshold, boost_coef, 1.0)
            boost_matrix = np.clip(boost_matrix, 1.0, max_boost_coef)
            boost_matrix[nocall_ix] = 1.0

        return boost_matrix

    res_df = pd.DataFrame()

    for filename in set(df["filename"]):  # type: ignore

        file_df = df[df.filename == filename].sort_values("_from_s")
        file_y_preds = np.array(list(file_df[pred_col]), dtype=np.float64)

        bm = _compute_boost_matrix(
            file_y_preds,
            labels=labels,
            threshold=threshold,
            boost_coef=boost_coef,
            max_boost_coef=max_boost_coef,
        )

        file_y_preds = bm * file_y_preds

        file_df[out_col] = list(map(lambda x: x, file_y_preds))
        res_df = res_df.append(file_df)

    return res_df.reset_index(drop=True)


df = boost_multiple_occurences(
    df=df,
    labels=meta["labels"],
    pred_col="_y_pred",
    boost_coef=1.5,
    max_boost_coef=1.5 * 1.5,
    threshold=0.5,
)

for i in range(df.shape[0]):
    if 0 != np.mean(df.iloc[i]["_y_pred"] - df.iloc[i]["_y_pred_boosted"]):  # type: ignore
        print(i)
