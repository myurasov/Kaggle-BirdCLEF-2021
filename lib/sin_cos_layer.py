import numpy as np
import tensorflow.keras.backend as K
from tensorflow import keras


class SinCos(keras.layers.Layer):
    """
    Encodes cyclical value as pair of sin/cos

    Example:

    ```python
        i_month = keras.layers.Input(shape=(1,), dtype="float32")
        x = SinCosEncode(val_range=[1, 12])(i_month)
        m = keras.Model(inputs=[i_month], outputs=[x])
        print(np.array(m.predict(np.arange(1, 13))))
    ```

    """

    def __init__(self, val_range=[0, 1], **kwargs):
        super(SinCos, self).__init__(**kwargs)
        self.val_range = val_range

    def call(self, inputs):

        inputs_nomalized = (
            (inputs - self.val_range[0])
            / (self.val_range[1] - self.val_range[0])
            * 2
            * np.pi
        )

        return K.concatenate(
            [K.sin(inputs_nomalized), K.cos(inputs_nomalized)],
        )

    def get_config(self):
        # config = super(SinCos, self).get_config()
        # config.update({"val_range": self.val_range})
        # return config

        return {"val_range": self.val_range}
