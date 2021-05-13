import datetime
import math
import os
import re
from collections import defaultdict
from glob import glob

import numpy as np
from pandas import DataFrame
from tqdm import tqdm

from src.config import c


def rectify_class_counts(df, class_col, max_items, min_items):
    """
    - Randomly drop rows with too many entries (> max_items)
    - Repeat rows with too liuttle entries (< min_items).
    """

    value_counts = df[class_col].value_counts()
    res_df = df.copy()

    for class_name, count in tqdm(
        zip(value_counts.index, value_counts), total=len(value_counts)
    ):

        if count > max_items:

            # too much - drop samples over max_items
            to_drop = df[df[class_col] == class_name].sample(count - max_items)
            res_df = res_df.drop(to_drop.index)

        elif count < min_items:

            # too little - add samples to min_items

            # repeat existing rows
            repeats = int(math.ceil(min_items / count))
            rows_to_repeat = df[df[class_col] == class_name]

            for _ in range(repeats - 1):
                res_df = res_df.append(rows_to_repeat, ignore_index=True)

            # drop excessive items
            count = res_df[res_df[class_col] == class_name].shape[0]
            to_drop = res_df[res_df[class_col] == class_name].sample(count - min_items)
            assert to_drop.shape[0] == count - min_items
            res_df = res_df.drop(to_drop.index)

    res_df.reset_index(inplace=True, drop=True)
    return res_df


def add_folds(df: DataFrame, n_folds, labels_col) -> DataFrame:
    """
    Add "fold" column to DataFrame
    Data is equally sampled from bins with same label combination
    """

    print(f"* Adding {n_folds} folds...")

    labels = np.array(list(df[labels_col]))

    # split items into bins with the same label
    # bins is a list of indexes
    bins = []
    for label in set(labels):
        bins.append(np.where(labels == label)[0])
        np.random.shuffle(bins[-1])

    print(f"* Total {len(bins)} bins")

    folds = np.zeros((df.shape[0]), dtype=np.int32)

    for fold in range(1, 1 + n_folds):
        for bin in bins:
            bin_fold_len = int(math.ceil(len(bin) / n_folds))
            bin_fold_indexes = bin[bin_fold_len * (fold - 1) : bin_fold_len * fold]
            folds[bin_fold_indexes] = fold

    fold_lens = list(
        map(lambda x: np.where(folds == x + 1)[0].shape[0], range(n_folds))
    )
    print(f"* Fold sizes: {fold_lens}")

    df["_fold"] = folds
    return df


def read_soundscapes_info(info_dir):
    """Read soundscapes information from txt files"""

    info = {}

    # read coordinates, location
    for p in glob(os.path.join(info_dir, "*.txt")):
        name = os.path.basename(p)[:3]
        with open(p, "r") as f:
            contents = f.read()
            lat = float(re.findall("Latitude: (.+)\\b", contents)[0])
            lon = float(re.findall("Longitude: (.+)\\b", contents)[0])
            location = re.findall("^.+\n.+", contents)[0].replace("\n", ", ")
        info[name] = {"lat": lat, "lon": lon, "location": location}

    return info


def predictions_to_text_labels(
    predictions,
    labels,
    default_label=None,
    max_labels=None,
    priority_to_nocall=False,
):
    """
    Convert predictions to text labels

    ```python
    predictions_to_text_labels(
        [
            [0, 0.75, 1],
            [0.55, 0.75, 0.6],
        ],
        ["a", "b", "nocall"],
    )

    # ['nocall', 'a b']
    ```
    """

    res = []
    predictions = np.array(predictions)
    labels = np.array(labels)

    for i in range(len(predictions)):
        label = set(labels[np.nonzero(predictions[i] > 0.5)])

        if "nocall" in label and len(label) > 1:

            if priority_to_nocall or labels[np.argmax(predictions[i])] == "nocall":
                # leave "nocall" on multi-predictions only if it's the strongest one
                # or priority_to_nocall is True
                label = set(["nocall"])
            else:
                label.remove("nocall")

        label = " ".join(sorted(list(label)[:max_labels]))

        if "" == label and default_label is not None:
            label = default_label

        res.append(label)

    return res


def normalize_soundscapes_df(
    df,
    seconds=5,
    rating=5.0,
    source="long",
    quiet=False,
):
    """Prepare soundscapes df to contain c["DATASET_COLS"] cols + 'row_id' + 'site'"""

    soundscapes_info = read_soundscapes_info(
        os.path.join(
            c["COMPETITION_DATA"],
            "test_soundscapes",
        )
    )

    def tqdm_(x, **kwargs):
        if quiet:
            return x
        else:
            return tqdm(x, **kwargs)

    df["rating"] = rating
    df["_source"] = source
    newcols = defaultdict(list)

    for _, row in tqdm_(df.iterrows(), total=df.shape[0]):

        file_glob = os.path.join(
            c["COMPETITION_DATA"],
            "*_soundscapes",
            f"*{row.audio_id}*.ogg",
        )

        # filename
        file_path = glob(file_glob, recursive=True)
        assert len(file_path) > 0
        file_name = os.path.basename(file_path[0])
        newcols["filename"].append(file_name)

        # date
        date_s = re.findall("_(\\d+).ogg$", file_name)[0]
        date = datetime.datetime.strptime(date_s, "%Y%m%d")
        newcols["_month"].append(date.month)
        newcols["_year"].append(date.year)

        # from/to
        # TODO: more precise sliding window
        newcols["_from_s"].append(row.seconds - seconds)
        newcols["_to_s"].append(row.seconds)

        # lat/lon
        newcols["latitude"].append(soundscapes_info[row.site]["lat"])
        newcols["longitude"].append(soundscapes_info[row.site]["lon"])

        # labels
        row_birds = " ".join(row.birds.split(" ")) if "birds" in row else ""
        newcols["_primary_labels"].append(row_birds)
        newcols["_secondary_labels"].append("")

    for k, v in newcols.items():
        df[k] = v

    df = df[c["DATASET_COLS"] + ["row_id", "site"]]

    return df


def geofilter_predictions(df, Y_pred, site_labels, labels, downgrade_const=0.501):
    """
    Filter predictions by dot-multiplying on the matrix
    with 1 at classes occuring in specific site
    (based on site_labels={'<site>': [<labels>]} dict).

    df should contain 'site' column

    ```python
    from src.geo_filter import filters as geo_filters

    Y_pred_geofiltered = geofilter_predictions(
        df=df,
        Y_pred=Y_pred,
        site_labels=geo_filters["all-500mi-0mo_tolerance"],
        labels=meta["labels"],
        downgrade_const=0.0,
    )
    ```
    """

    res = np.copy(Y_pred)
    site_filters = {}

    for site, site_labels in site_labels.items():
        ones_at = list(map(lambda x: labels.index(x), site_labels))
        site_filter = np.repeat(downgrade_const, len(labels))
        site_filter[np.array(ones_at)] = 1.0
        site_filters[site] = site_filter

    for i, site in enumerate(list(df["site"])):
        res[i] *= site_filters[site]

    return res
