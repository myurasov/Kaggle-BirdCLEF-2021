import numpy as np
from pandas import DataFrame
from tensorflow import keras

from src.wave_provider import WaveProvider
from src.msg_provider import MSG_Provider


class MSG_Generator(keras.utils.Sequence):
    """
    Generator for melspectrogram-based training
    """

    def __init__(
        self,
        df: DataFrame,
        wave_provider: WaveProvider,
        msg_provider: MSG_Provider,
        shuffle=True,
        batch_size=32,
        augmentation=None,
    ):
        self._df = df
        self._shuffle = shuffle
        self._batch_size = batch_size
        self._msg_provider = msg_provider
        self._augmentation = augmentation
        self._wave_provider = wave_provider

    def __len__(self):
        return self._n_samples // self._batch_size

    def __getitem__(self, batch_ix):
        assert batch_ix < self._df.shape[0]
        b_X, b_Y = [], []

        for i in range(self._batch_size):
            x, y = self._get_one(i + self._batch_size * batch_ix)
            b_X.append(x)
            b_Y.append(y)

        return b_X, b_Y

    def on_epoch_end(self):
        if self._shuffle:
            self._shuffle_samples()

    def _get_one(self, ix):
        wave = self._wave_provider.get_audio_fragment(
            file_name=self._df.loc[ix]["filename"],
            start_s=self._df.loc[ix]["_from_s"],
            end_s=self._df.loc[ix]["_to_s"],
        )

        x = self._msg_provider.msg(wave).astype(np.float16)
        y = np.array(self._df._Y_labels.loc[ix], dtype=np.float16)

        return x, y

    def _shuffle_samples(self):
        self._df = self._df.sample(frac=1).reset_index(drop=True)
