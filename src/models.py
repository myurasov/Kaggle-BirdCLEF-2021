import tensorflow.keras.backend as K
from lib.sin_cos_layer import SinCos
from tensorflow import keras


class MSG_Model_Builder:
    """Builds spectrogram-based model"""

    def __init__(
        self,
        n_classes,
        body="enb0",
        extra_dense_layers=[
            1,
            1024,
        ],  # number and dimensions for extra dense layers in the head. None = no extra layers.
    ):
        self._body = body
        self._n_classes = n_classes
        self._extra_dense_layers = extra_dense_layers

    def build(self):

        # msg branch

        if self._body[:3] == "enb":  # EfficientNetB#

            msg_i_size = [224, 240, 260, 300, 380, 456, 528, 600][int(self._body[-1])]
            msg_i_size = (msg_i_size, msg_i_size, 3)

            i_msg = keras.layers.Input(
                shape=msg_i_size,
                dtype="uint8",
                name="i_msg",
            )

            x = getattr(keras.applications, f"EfficientNetB{self._body[-1]}")(
                include_top=False,
                weights=None,
            )(i_msg)

            f_msg = keras.layers.GlobalAveragePooling2D(name="f_msg")(x)

        else:
            raise ValueError(f'Unsupported "body" parameter value "{self._body}"')

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

        f_date = keras.layers.Lambda(
            # x[1] is year, x[0] is month
            lambda x: K.cast(x[1] * 12 + x[0], "float32"),
            name="f_date",
        )([i_year, i_month])

        # lat

        i_lat = keras.layers.Input(
            shape=(1,),
            dtype="float32",
            name="i_lat",
        )

        f_lat = keras.layers.Lambda(lambda x: x / 90.0, name="f_lat",)(
            i_lat
        )  # normalize to -1..1

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

        if self._extra_dense_layers is not None:
            for _ in range(self._extra_dense_layers[0]):
                x = keras.layers.Dense(
                    self._extra_dense_layers[1],
                    activation="relu",
                )(x)

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
