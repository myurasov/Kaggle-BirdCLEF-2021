import unittest

import pandas as pd
from src.config_test import c
from src.msg_generator import MSG_Generator
from src.services import get_data_provider


class Test_MSG_Generator(unittest.TestCase):
    def setUp(self):
        df = pd.DataFrame()
        dp = get_data_provider(c)
        self._generator = MSG_Generator(df=df, data_provider=dp)


if __name__ == "__main__":
    unittest.main(warnings="ignore")
