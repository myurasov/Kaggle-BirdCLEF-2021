import warnings

import numpy as np
from lib.utils import coarsen_number, float2d_to_rgb
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
        msg_output_size=(256, 256, 3),  # output size of a melspectrogram
        msg_power=2,
        batch_size=32,
        augmentation=None,
        shuffle=True,  # shuffle on each epoch
        rating_as_sw=True,  # use rating/5 as sample weight
        rareness_as_sw=True,  # use 1/<class_freq> as sw. multiplied by rating if both are set.
        geo_coordinates_bins=None,  # number of bins for coarsening lat/lon
    ):
        self._df = df.copy()
        self._shuffle = shuffle
        self._msg_power = msg_power
        self._batch_size = batch_size
        self._msg_provider = msg_provider
        self._rating_as_sw = rating_as_sw
        self._augmentation = augmentation
        self._wave_provider = wave_provider
        self._rareness_as_sw = rareness_as_sw
        self._msg_output_size = msg_output_size
        self._geo_coordinates_bins = geo_coordinates_bins

        if augmentation is None:
            self._augmentation = {}

        # compute rareness sample weighting coefficients
        if self._rareness_as_sw is not None:
            rareness_sws = np.array(
                list(
                    df["_primary_labels"].replace(
                        dict(df["_primary_labels"].value_counts())
                    )
                ),
                dtype=np.float64,
            )
            rareness_sws /= np.max(rareness_sws)
            self._df["_rareness_sw"] = rareness_sws

        # if we're in prediction mode, we have no Y column
        if "_y" not in self._df:
            self._df["_y"] = [0] * self._df.shape[0]

        # reset index - just in case
        self._df.reset_index(inplace=True, drop=True)

        # shuffle before first epoch for non-model.fit uses
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

        # sample weight
        sw = 1.0
        if self._rating_as_sw:
            sw *= self._df.loc[ix]["rating"] / 5.0  # type: ignore
        if self._rareness_as_sw:
            sw *= self._df.loc[ix]["_rareness_sw"]  # type: ignore

        # y
        y = np.array(self._df.loc[ix]["_y"])  # type: ignore

        # x

        x = {}

        # wave

        wave = self._wave_provider.get_audio_fragment(
            file_name=self._df.loc[ix]["filename"],  # type: ignore
            range_seconds=[
                self._df.loc[ix]["_from_s"],  # type: ignore
                self._df.loc[ix]["_to_s"],  # type: ignore
            ],
        )

        # augmentation: mix with same class
        wave, y = self._aug_same_class_mixing(
            default_wave=wave,
            default_y=y,
            current_ix=ix,
        )

        # augmentation: mix with nocall
        wave, y = self._aug_nocall_mixing(
            default_wave=wave,
            default_y=y,
            current_ix=ix,
        )

        # msg

        if self._msg_provider is None:  # return waves

            x["i_wave"] = wave

        else:  # return melspectrograms

            # augmentation: random melspectrogram power
            power = self._aug_msg_random_power(default=self._msg_power)

            msg = self._msg_provider.msg(
                wave,
                n_mels=self._msg_output_size[0],
                time_steps=self._msg_output_size[1],
                power=power,
            ).astype(np.float32)

            # return as rgb uint8 image
            if len(self._msg_output_size) == 3:
                if self._msg_output_size[2] == 3:
                    msg = float2d_to_rgb(msg)

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

        return x, y.astype(np.float16), sw

    def _shuffle_samples(self):
        self._df = self._df.sample(frac=1).reset_index(drop=True)

    def _aug_msg_random_power(self, default):
        key = "msg.random_power"

        if key in self._augmentation:
            opts = self._augmentation[key]

            if np.random.uniform(0, 1) <= opts["chance"]:
                return np.random.uniform(
                    opts["min_power"],
                    opts["max_power"],
                )

        return default

    def _aug_same_class_mixing(self, default_wave, default_y, current_ix):
        key = "wave.same_class_mixing"

        if key in self._augmentation:
            opts = self._augmentation[key]

            if np.random.uniform(0, 1) <= opts["chance"]:

                wave, y = self._aug_mix_wave(
                    default_wave=default_wave,
                    default_y=default_y,
                    current_ix=current_ix,
                    with_class=self._df.loc[current_ix]["_primary_labels"],  # type: ignore
                    coeffs=opts["coeffs"],
                )

                return wave, y if opts["labels"] else default_y

        return default_wave, default_y

    def _aug_nocall_mixing(self, default_wave, default_y, current_ix):
        key = "wave.nocall_mixing"

        if key in self._augmentation:
            opts = self._augmentation[key]

            if np.random.uniform(0, 1) <= opts["chance"]:

                wave, _ = self._aug_mix_wave(
                    default_wave=default_wave,
                    default_y=default_y,
                    current_ix=current_ix,
                    with_class="nocall",
                    coeffs=opts["coeffs"],
                )

                return wave, default_y

        return default_wave, default_y

    def _aug_mix_wave(self, default_wave, default_y, current_ix, with_class, coeffs):
        # coefficients for multiplying samples/labels to
        coeffs = [np.random.uniform(*x) for x in coeffs]
        coeffs /= sum(coeffs)

        # find more candidates of the same class to mix with
        extra_rows = self._df[
            (self._df["_primary_labels"] == with_class) & (self._df.index != current_ix)
        ].sample(n=len(coeffs) - 1)
        coeffs = coeffs[: 1 + extra_rows.shape[0]]
        extra_rows = extra_rows.to_dict("list")

        # original wave
        wave = default_wave.astype(np.float32) * coeffs[0]

        # original y
        y = default_y.astype(np.float32) * coeffs[0]

        for filename, from_s, to_s, row_y, coeff in zip(
            extra_rows["filename"],
            extra_rows["_from_s"],
            extra_rows["_to_s"],
            extra_rows["_y"],
            coeffs[1:],
        ):
            wave += (
                coeff
                * self._wave_provider.get_audio_fragment(
                    file_name=filename,
                    range_seconds=[from_s, to_s],
                ).astype(np.float32)
            )

            y += coeff * row_y.astype(np.float32)

        return wave, y
