from tensorflow import keras

from src.config import c


class MSG_Model(keras.Model):
    def __init__(
        self,
        n_classes,
        body="enb0",
    ):
        super(MSG_Model, self).__init__()
        self._n_classes = n_classes

        self.dense1 = keras.layers.Dense(4, activation="relu")
        self.dense2 = keras.layers.Dense(self._n_classes, activation="sigmoid")

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)
