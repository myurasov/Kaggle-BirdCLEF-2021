import os
import unittest

import numpy as np
import pandas as pd
from src.config import c
from src.msg_generator import MSG_Generator
from src.services import get_wave_provider, get_msg_provider


class Test_MSG_Generator(unittest.TestCase):
    def setUp(self):

        self._generator = MSG_Generator(
            df=pd.read_pickle(os.path.join(c["WORK_DIR"], "dataset.pickle")),
            wave_provider=get_wave_provider(c),
            msg_maker=get_msg_provider(c),
            batch_size=1,
            shuffle=False,
            augmentation=None,
        )

    def test_1(self):
        b_x, b_y = self._generator.__getitem__(100)

        self.assertEqual(b_x[0].dtype, np.float16)

        self.assertEqual(
            b_x[0].shape,
            (
                c["MSG_TARGET_SIZE"]["freqs"],
                c["MSG_TARGET_SIZE"]["time"],
            ),
        )

        self.assertGreater(
            len(np.where(b_y[0] == 0.0)[0]),
            len(np.where(b_y[0] == 1.0)[0]),
        )

        self.assertEqual(b_y[0].dtype, np.float16)


if __name__ == "__main__":
    unittest.main(warnings="ignore")
