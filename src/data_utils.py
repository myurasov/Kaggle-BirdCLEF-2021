import math

from tqdm import tqdm


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

    return res_df
