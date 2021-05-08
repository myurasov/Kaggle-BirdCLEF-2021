from lib.sin_cos_encode import SinCosEncode
from tensorflow import keras


class MSG_Model_Builder:
    def __init__(
        self,
        n_classes,
        body="enb0",
    ):
        self._body = body
        self._n_classes = n_classes

    def build(self):

        # msg branch

        if self._body[:3] == "enb":  # EfficientNetB#

            msg_input_size = [224, 240, 260, 300, 380, 456, 528, 600][
                int(self._body[-1])
            ]
            msg_input_size = (msg_input_size, msg_input_size, 3)

            input_msg = keras.layers.Input(
                shape=msg_input_size,
                dtype="uint8",
                name="input_msg",
            )

            x = getattr(keras.applications, f"EfficientNetB{int(self._body[-1])}")(
                include_top=False,
                weights=None,
            )(input_msg)

            feature_msg = keras.layers.GlobalAveragePooling2D(name="feature_msg")(x)

        else:
            raise ValueError(f'Unsupported "body" value "{self._body}"')

        # year
        input_year = feature_year = keras.layers.Input(
            shape=(1,),
            dtype="float32",
            name="input_year",
        )

        # month

        input_month = keras.layers.Input(
            shape=(1,),
            dtype="float32",
            name="input_month",
        )

        feature_month_sincos = SinCosEncode(  # type: ignore
            val_range=[1, 12],
            name="feature_month_sincos",
        )(input_month)

        # latitude

        input_latitude = feature_latitude = keras.layers.Input(
            shape=(1,),
            dtype="float32",
            name="input_latitude",
        )

        # longitude

        input_longitude = keras.layers.Input(
            shape=(1,),
            dtype="float32",
            name="input_longitude",
        )

        feature_longitude_sincos = SinCosEncode(  # type: ignore
            val_range=[-180, 180],
            name="feature_longitude_sincos",
        )(input_longitude)

        # combine all the features

        features = keras.layers.Concatenate(axis=1, name="features",)(
            [
                feature_msg,
                feature_year,
                feature_month_sincos,
                feature_latitude,
                feature_longitude_sincos,
            ]
        )

        # classifier head

        output_classes = keras.layers.Dense(
            self._n_classes, name="output_classes", activation="sigmoid"
        )(features)

        # model

        self.model = keras.models.Model(
            inputs=[
                input_msg,
                input_year,
                input_month,
                input_latitude,
                input_longitude,
            ],
            outputs=[output_classes],
        )
