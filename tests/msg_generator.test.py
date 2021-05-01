import os
import unittest

import pandas as pd
from src.config import c
from src.msg_generator import MSG_Generator
from src.services import get_data_provider, get_msg_maker


class Test_MSG_Generator(unittest.TestCase):
    def setUp(self):

        self._generator = MSG_Generator(
            df=pd.read_pickle(os.path.join(c["WORK_DIR"], "dataset.pickle")),
            data_provider=get_data_provider(c),
            msg_maker=get_msg_maker(c),
            batch_size=1,
            shuffle=False,
            augmentation=None,
        )

    def test_1(self):
        b_x, b_y = self._generator.__getitem__(0)
        print(b_x, b_y)


if __name__ == "__main__":
    unittest.main(warnings="ignore")
