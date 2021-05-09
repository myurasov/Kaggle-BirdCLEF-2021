import numpy as np
from lib.utils import coarsen_number
from pandas import DataFrame
from tensorflow import keras

from src.msg_provider import MSG_Provider
from src.wave_provider import WaveProvider


class Generator(keras.utils.Sequence):
    """
    Generator
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
        b_x_dict, b_y, b_sw = {}, [], []

        for i in range(self._batch_size):
            x_dict, y, s = self._get_one(i + self._batch_size * batch_ix)

            # single x is dictionary of <input>:<value>
            # but, batch needs to be a dictionaty of <input>:np.array(<values>)
            for k, v in x_dict.items():
                if k not in b_x_dict:
                    b_x_dict[k] = []
                b_x_dict[k].append(v)

            b_y.append(y)
            b_sw.append(s)

        for k, v in b_x_dict.items():
            b_x_dict[k] = np.array(v)

        b_y = np.array(b_y)
        b_sw = np.array(b_sw)

        return b_x_dict, b_y, b_sw

    def on_epoch_start(self):
        if self._shuffle:
            self._shuffle_samples()

    def _get_one(self, ix):

        x = {}

        # wave

        wave = self._wave_provider.get_audio_fragment(
            file_name=self._df.loc[ix]["filename"],  # type: ignore
            range_seconds=[
                self._df.loc[ix]["_from_s"],  # type: ignore
                self._df.loc[ix]["_to_s"],  # type: ignore
            ],
        )

        # msg

        if self._msg_provider is None:  # return waves

            x["i_wave"] = wave

        else:  # return melspectrograms

            msg = self._msg_provider.msg(wave).astype(np.float16)

            # return as rgb uint8 image
            if self._msg_as_rgb:
                # force normalization and convert to uint8 range
                msg = msg.astype(np.float32)
                msg = (msg - np.mean(msg)) / np.std(msg) * 128
                # duplicate across 3 channels
                msg = np.repeat(np.expand_dims(msg.astype(np.uint8), 2), 3, 2)

            x["i_msg"] = msg

        # lat/lon

        x["i_lat"] = float(self._df.loc[ix]["latitude"])  # type: ignore
        x["i_lon"] = float(self._df.loc[ix]["longitude"])  # type: ignore

        if self._geo_coordinates_bins is not None:
            x["i_lat"] = coarsen_number(
                x["i_lat"],
                bins=self._geo_coordinates_bins,
                val_range=[-90, 90],
            )
            x["i_lon"] = coarsen_number(
                x["i_lon"],
                bins=self._geo_coordinates_bins,
                val_range=[-180, 180],
            )

        x["i_lat"] = np.array(x["i_lat"], dtype=np.float32)
        x["i_lon"] = np.array(x["i_lon"], dtype=np.float32)

        # date
        x["i_year"] = np.array(self._df.loc[ix]["_year"], dtype=np.int32)  # type: ignore
        x["i_month"] = np.array(self._df.loc[ix]["_month"], dtype=np.int32)  # type: ignore

        # y
        y = np.array(self._df._y.loc[ix], dtype=np.float16)

        # sample weight
        sw = 1.0
        if self._rating_as_sw:
            sw = self._df.loc[ix]["rating"] / 5.0  # type: ignore

        return x, y, sw

    def _shuffle_samples(self):
        self._df = self._df.sample(frac=1).reset_index(drop=True)
