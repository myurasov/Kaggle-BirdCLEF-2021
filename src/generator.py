import numpy as np
from pandas import DataFrame
from tensorflow import keras

from src.wave_provider import WaveProvider
from src.msg_provider import MSG_Provider

from lib.utils import coarsen_number


class Generator(keras.utils.Sequence):
    """
    Generator for melspectrogram-based training
    """

    def __init__(
        self,
        df: DataFrame,
        wave_provider: WaveProvider,
        msg_provider: MSG_Provider = None,  # if not set, waves will be returned
        batch_size=32,
        shuffle=True,  # shuffle on each epoch
        augmentation=None,
        msg_as_rgb=True,  # return melspectrogram as rgb image
        rating_as_sw=True,  # use rating/5 as sample weight
        geo_coordinates_bins=None,  # number of bins for coarsening lat/lon
    ):
        self._df = df
        self._shuffle = shuffle
        self._batch_size = batch_size
        self._msg_as_rgb = msg_as_rgb
        self._msg_provider = msg_provider
        self._augmentation = augmentation
        self._rating_as_sw = rating_as_sw
        self._wave_provider = wave_provider
        self._geo_coordinates_bins = geo_coordinates_bins

        if self._shuffle:
            self._shuffle_samples()

    def __len__(self):
        return self._df.shape[0] // self._batch_size

    def __getitem__(self, batch_ix):
        assert batch_ix < self._df.shape[0]
        b_x, b_y, b_sw = [], [], []

        for i in range(self._batch_size):
            x, y, s = self._get_one(i + self._batch_size * batch_ix)
            b_x.append(x)
            b_y.append(y)
            b_sw.append(s)

        return b_x, b_y, b_sw

    def on_epoch_start(self):
        if self._shuffle:
            self._shuffle_samples()

    def _get_one(self, ix):
        x = {}

        # wave

        wave = self._wave_provider.get_audio_fragment(
            file_name=self._df.loc[ix]["filename"],
            start_s=self._df.loc[ix]["_from_s"],
            end_s=self._df.loc[ix]["_to_s"],
        )

        # msg

        if self._msg_provider is None:  # return waves

            x["input_wave"] = wave

        else:  # return melspectrograms

            msg = self._msg_provider.msg(wave).astype(np.float16)

            # return as rgb uint8 image
            if self._msg_as_rgb:
                # force normalization and convert to uint8 range
                msg = (msg - np.mean(msg)) / np.std(msg) * 128
                # duplicate across 3 channels
                msg = np.repeat(np.expand_dims(msg.astype(np.uint8), 2), 3, 2)

            x["input_msg"] = msg

        # lat/lon

        x["input_latitude"] = float(self._df.loc[ix].latitude)
        x["input_longitude"] = float(self._df.loc[ix].longitude)

        if self._geo_coordinates_bins is not None:
            x["input_latitude"] = coarsen_number(
                x["input_latitude"],
                bins=self._geo_coordinates_bins,
                val_range=[-90, 90],
            )
            x["input_longitude"] = coarsen_number(
                x["input_longitude"],
                bins=self._geo_coordinates_bins,
                val_range=[-180, 180],
            )

        # date
        x["input_year"] = int(self._df.loc[ix]._year)
        x["input_month"] = int(self._df.loc[ix]._year)

        y = np.array(self._df._Y_labels.loc[ix], dtype=np.float16)

        # sample weight
        sw = 1.0
        if self._rating_as_sw:
            sw = self._df.loc[ix]["rating"] / 5.0

        return x, y, sw

    def _shuffle_samples(self):
        self._df = self._df.sample(frac=1).reset_index(drop=True)
