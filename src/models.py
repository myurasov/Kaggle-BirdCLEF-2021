import tensorflow.keras.backend as K

from lib.float2d_to_rgb_layer import Float2DToFloatRGB, Float2DToRGB
from lib.melspectrogram_layer import MelSpectrogram
from lib.power_to_db_layer import PowerToDb
from lib.sin_cos_layer import SinCos
from tensorflow import keras

from src.config import c


def build_model(name, n_classes) -> keras.models.Model:
    """Name convetion mgs|wave_body_option1_option2_.."""
    input_type, body, *options = name.split("_")

    if input_type == "msg":

        mb = MSG_Model_Builder(
            n_classes=n_classes,
            body=body,
            imagenet_weights="imagenet" in options,
            extra_dense_layers=[1, 1024] if "xdense" in options else None,
            dropout=0.5 if "drops" in options else None,
        )

        return mb.build()

    elif input_type == "wave":

        # this model somehow only works without raising melspec to power other than one...
        assert c["MSG_POWER"] == 1

        mb = Wave_Model_Builder(
            n_classes=n_classes,
            body=body,
            imagenet_weights="imagenet" in options,
            extra_dense_layers=[1, 1024] if "xdense" in options else None,
            dropout=0.5 if "drops" in options else None,
            wave_len_samples=c["AUDIO_SR"] * c["AUDIO_TARGET_LEN_S"],
            wave_sample_rate=c["AUDIO_SR"],
            n_fft=c["MSG_N_FFT"],
            spec_power=c["MSG_POWER"],
        )

        return mb.build()

    raise ValueError(f'Model "{name}" can\'t be built')


class MSG_Model_Builder:
    """Builds spectrogram-based model"""

    def __init__(
        self,
        n_classes,
        body="enb0",
        imagenet_weights=False,
        extra_dense_layers=[
            1,
            1024,
        ],  # number and dimensions for extra dense layers in the head. None = no extra layers.
        dropout=None,
    ):
        self._body = body
        self._dropout = dropout
        self._n_classes = n_classes
        self._imagenet_weights = imagenet_weights
        self._extra_dense_layers = extra_dense_layers

    def build(self) -> keras.models.Model:

        if self._body[:3] == "enb":
            self._body = "EfficientNetB" + self._body[-1]
            msg_shape = [224, 240, 260, 300, 380, 456, 528, 600]
            msg_shape = msg_shape[int(self._body[-1])]
            msg_shape = (msg_shape, msg_shape, 3)
            msg_dtype = "uint8"
        elif self._body.lower() == "resnet50":
            self._body = "ResNet50"
            msg_shape = (224, 224, 3)
            msg_dtype = "float32"
        else:
            raise ValueError(f'Unsupported body type "{self._body}"')

        i_msg = keras.layers.Input(
            shape=msg_shape,
            dtype=msg_dtype,
            name="i_msg",
        )

        x = getattr(keras.applications, self._body)(
            include_top=False,
            weights="imagenet" if self._imagenet_weights else None,
        )(i_msg)

        f_msg = keras.layers.GlobalAveragePooling2D(name="f_msg")(x)

        # year
        i_year = keras.layers.Input(
            shape=(1,),
            dtype="int32",
            name="i_year",
        )

        # month

        i_month = keras.layers.Input(
            shape=(1,),
            dtype="int32",
            name="i_month",
        )

        f_month_sincos = SinCos(  # type: ignore
            val_range=[1, 12],
            name="f_month_sincos",
        )(i_month)

        # date
        f_date = YMToDate(name="f_date")([i_year, i_month])  # type: ignore

        # lat

        i_lat = keras.layers.Input(
            shape=(1,),
            dtype="float32",
            name="i_lat",
        )

        f_lat = Div(by=90, name="f_lat")(i_lat)  # type: ignore

        # lon

        i_lon = keras.layers.Input(
            shape=(1,),
            dtype="float32",
            name="i_lon",
        )

        f_lon_sincos = SinCos(  # type: ignore
            val_range=[-180, 180],
            name="f_lon_sincos",
        )(i_lon)

        # combine all the features

        features = keras.layers.Concatenate(axis=1, name="features",)(
            [
                f_msg,
                f_date,
                f_month_sincos,
                f_lat,
                f_lon_sincos,
            ]
        )

        # classifier head
        x = features

        if self._dropout is not None:
            x = keras.layers.Dropout(self._dropout)(x)

        if self._extra_dense_layers is not None:
            for _ in range(self._extra_dense_layers[0]):

                x = keras.layers.Dense(
                    self._extra_dense_layers[1],
                    activation="relu",
                )(x)

                if self._dropout is not None:
                    x = keras.layers.Dropout(self._dropout)(x)

        o_classes = keras.layers.Dense(
            self._n_classes,
            name="o_classes",
            activation="sigmoid",
        )(x)

        # model

        self.model = keras.models.Model(
            inputs=[
                i_msg,
                i_year,
                i_month,
                i_lat,
                i_lon,
            ],
            outputs=[o_classes],
        )

        return self.model


class Wave_Model_Builder:
    """Builds wave-based model with Karpe (https://github.com/keunwoochoi/kapre/) layers"""

    def __init__(
        self,
        n_classes,
        wave_len_samples,
        wave_sample_rate,
        n_fft=2048,
        spec_power=2,
        body="enb0",
        imagenet_weights=False,
        extra_dense_layers=None,
        dropout=None,
        freq_min=0,
        freq_max=None,
    ):
        self._body = body
        self._n_fft = n_fft
        self._dropout = dropout
        self._freq_min = freq_min
        self._n_classes = n_classes
        self._spec_power = spec_power
        self._imagenet_weights = imagenet_weights
        self._wave_len_samples = wave_len_samples
        self._wave_sample_rate = wave_sample_rate
        self._extra_dense_layers = extra_dense_layers
        self._freq_max = freq_max if freq_max is not None else wave_sample_rate // 2

    def build(self) -> keras.models.Model:

        # msg branch

        if self._body[:3] == "enb":
            self._body = "EfficientNetB" + self._body[-1]
            msg_shape = [224, 240, 260, 300, 380, 456, 528, 600]
            msg_shape = msg_shape[int(self._body[-1])]
            msg_shape = (msg_shape, msg_shape, 3)
            msg_dtype = "uint8"
        elif self._body.lower() == "resnet50":
            self._body = "ResNet50"
            msg_shape = (224, 224, 3)
            msg_dtype = "float"
        else:
            raise ValueError(f'Unsupported body type "{self._body}"')

        # wave

        i_wave = x = keras.layers.Input(
            shape=(self._wave_len_samples),
            dtype="float16",
            name="i_wave",
        )

        x = MelSpectrogram(
            sample_rate=self._wave_sample_rate,
            fft_size=self._n_fft,
            n_mels=msg_shape[1],
            hop_size=self._wave_len_samples // (msg_shape[0] - 1),
            power=self._spec_power,
            f_min=self._freq_min,
            f_max=self._freq_max,
        )(
            x
        )  # type: ignore

        x = PowerToDb()(x)  # type: ignore

        if len(msg_shape) == 3 and msg_shape[2] == 3:
            if msg_dtype == "uint8":
                x = Float2DToRGB()(x)  # type: ignore
            else:
                x = Float2DToFloatRGB()(x)  # type: ignore

        x = getattr(keras.applications, self._body)(
            input_shape=msg_shape,
            include_top=False,
            weights="imagenet" if self._imagenet_weights else None,
        )(x)

        f_msg = keras.layers.GlobalAveragePooling2D(name="f_msg")(x)

        # year
        i_year = keras.layers.Input(
            shape=(1,),
            dtype="int32",
            name="i_year",
        )

        # month

        i_month = keras.layers.Input(
            shape=(1,),
            dtype="int32",
            name="i_month",
        )

        f_month_sincos = SinCos(  # type: ignore
            val_range=[1, 12],
            name="f_month_sincos",
        )(i_month)

        # date
        f_date = YMToDate(name="f_date")([i_year, i_month])  # type: ignore

        # lat

        i_lat = keras.layers.Input(
            shape=(1,),
            dtype="float32",
            name="i_lat",
        )

        f_lat = Div(by=90, name="f_lat")(i_lat)  # type: ignore

        # lon

        i_lon = keras.layers.Input(
            shape=(1,),
            dtype="float32",
            name="i_lon",
        )

        f_lon_sincos = SinCos(  # type: ignore
            val_range=[-180, 180],
            name="f_lon_sincos",
        )(i_lon)

        # combine all the features

        features = keras.layers.Concatenate(axis=1, name="features")(
            [f_msg, f_date, f_month_sincos, f_lat, f_lon_sincos]
        )

        # classifier head
        x = features

        if self._dropout is not None:
            x = keras.layers.Dropout(self._dropout)(x)

        if self._extra_dense_layers is not None:
            for _ in range(self._extra_dense_layers[0]):

                x = keras.layers.Dense(
                    self._extra_dense_layers[1],
                    activation="relu",
                )(x)

                if self._dropout is not None:
                    x = keras.layers.Dropout(self._dropout)(x)

        o_classes = keras.layers.Dense(
            self._n_classes,
            name="o_classes",
            activation="sigmoid",
        )(x)

        # model

        self.model = keras.models.Model(
            inputs=[i_wave, i_year, i_month, i_lat, i_lon],
            outputs=[o_classes],
        )

        return self.model


class Div(keras.layers.Layer):
    def __init__(self, by, **kwargs):
        super(Div, self).__init__(**kwargs)
        self._by = by

    def call(self, inputs):
        return inputs / self._by

    def get_config(self):
        return {"by": self._by}


class YMToDate(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(YMToDate, self).__init__(**kwargs)

    def call(self, inputs):
        # inputs[1] is year, inputs[0] is month
        return K.cast(inputs[1] * 12 + inputs[0], "float32")


class Pow(keras.layers.Layer):
    def __init__(
        self,
        power=2,
        **kwargs,
    ):
        super(Pow, self).__init__(**kwargs)
        self._power = power

    def call(self, inputs):
        return K.pow(inputs, self._power)

    def get_config(self):
        return {"power": self._power}


class Cast(keras.layers.Layer):
    def __init__(
        self,
        dtype="float32",
        **kwargs,
    ):
        super(Cast, self).__init__(**kwargs)
        self._dtype = dtype

    def call(self, inputs):
        return K.cast(inputs, self._dtype)

    def get_config(self):
        return {"dtype": self._dtype}
