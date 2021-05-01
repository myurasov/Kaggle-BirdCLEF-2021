import numpy as np
from pandas import DataFrame
from tensorflow import keras

from src.data_provider import DataProvider


class MSG_Generator(keras.utils.Sequence):
    """
    Generator for melspectrogram-based training
    """

    def __init__(
        self,
        df: DataFrame,
        data_provider: DataProvider,
        shuffle=True,
        batch_size=32,
        augmentations=set(["mix", "wave", "msg"]),
    ):
        self._df = df
        self._shuffle = shuffle
        self._batch_size = batch_size
        self._data_provider = data_provider
        self._augmentations = augmentations

    def __len__(self):
        return self._n_samples // self._batch_size

    def __getitem__(self, batch_ix):
        assert batch_ix < self._df.shape[0]
        b_X, b_Y = [], []

        for i in range(self._batch_size):
            x, y = self._get_one(i + self._batch_size * batch_ix)
            b_X.append(x)
            b_Y.append(y)

        return np.array(b_Y), np.array(b_Y)

    def on_epoch_end(self):
        if self._shuffle:
            self._shuffle_samples()

    def _get_one(self, ix):

        # TODO

        wave = self._data_provider.get_audio_fragment(
            file_name=self._df.loc[ix]["filename"],
            start_s=self._df.loc[ix]["_from_s"],
            end_s=self._df.loc[ix]["_to_s"],
        )

        return wave, np.array([0, 0, 0, 0, 1], dtype=np.float16)  # !!!

    def _shuffle_samples(self):
        self._df = self._df.sample(frac=1).reset_index(drop=True)
