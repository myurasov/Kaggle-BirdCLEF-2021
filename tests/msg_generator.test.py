import os
import unittest

import numpy as np
import pandas as pd
from src.config import c
from src.generator import Generator
from src.services import get_msg_provider, get_wave_provider


class Test_MSG_Generator(unittest.TestCase):
    def setUp(self):

        self._msg_generator_rgb = Generator(
            df=pd.read_pickle(os.path.join(c["WORK_DIR"], "dataset.pickle")),
            wave_provider=get_wave_provider(c),
            msg_provider=get_msg_provider(c),
            msg_output_size=(128, 256, 3),
            msg_power=c["MSG_POWER"],
            batch_size=10,
            shuffle=False,
            augmentation=None,
        )

        self._msg_generator_non_rgb = Generator(
            df=pd.read_pickle(os.path.join(c["WORK_DIR"], "dataset.pickle")),
            wave_provider=get_wave_provider(c),
            msg_provider=get_msg_provider(c),
            msg_output_size=(128, 256),
            msg_power=c["MSG_POWER"],
            batch_size=10,
            shuffle=False,
            augmentation=None,
        )

    def test_1(self):
        b_x, b_y, _ = self._msg_generator_rgb.__getitem__(100)

        self.assertEqual(b_x["i_msg"].dtype, np.uint8)

        self.assertEqual(
            b_x["i_msg"].shape[1:],
            (128, 256, 3),
        )

        self.assertEqual(b_x["i_lat"].dtype, np.float32)
        self.assertEqual(b_x["i_lon"].dtype, np.float32)
        self.assertEqual(b_x["i_month"].dtype, np.int32)
        self.assertEqual(b_x["i_year"].dtype, np.int32)

        self.assertGreater(
            len(np.where(b_y[0] == 0.0)[0]),
            len(np.where(b_y[0] == 1.0)[0]),
        )

        self.assertEqual(b_y[0].dtype, np.float16)

    def test_2(self):
        b_x, _, _ = self._msg_generator_non_rgb.__getitem__(100)

        self.assertEqual(b_x["i_msg"].dtype, np.float32)

        self.assertEqual(
            b_x["i_msg"].shape[1:],
            (128, 256),
        )


if __name__ == "__main__":
    unittest.main(warnings="ignore")
