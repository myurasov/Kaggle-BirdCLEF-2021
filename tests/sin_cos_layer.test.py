import unittest

import numpy as np
from lib.sin_cos_layer import SinCos
from tensorflow import keras


class Test_SinCos_Layer(unittest.TestCase):
    def test_encode_month(self):
        inputs = np.expand_dims(np.arange(1, 13), axis=1)

        i_month = keras.layers.Input(shape=(1,), dtype="int32")
        x = SinCos(val_range=[1, 12])(i_month)  # type: ignore
        m = keras.Model(inputs=[i_month], outputs=[x])
        results = np.array(m.predict(inputs))

        for input, result in zip(inputs, results):
            self.assertAlmostEqual(
                np.sin(2 * np.pi * (input[0] - 1) / 11), result[0], delta=1e-6
            )
            self.assertAlmostEqual(
                np.cos(2 * np.pi * (input[0] - 1) / 11), result[1], delta=1e-6
            )


if __name__ == "__main__":
    unittest.main(warnings="ignore")
