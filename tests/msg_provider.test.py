import unittest

import numpy as np
from src.config_test import c
from src.services import get_msg_provider, get_wave_provider


class Test_MSG_Provider(unittest.TestCase):
    def setUp(self):
        self._wave_provider = get_wave_provider(c)

    def test_1(self):
        wave = self._wave_provider.get_audio_fragment(
            "XC11209.ogg", range_seconds=[0, 5]
        )

        msg_p = get_msg_provider(c, key="_test_1_")
        msg = msg_p.msg(wave, n_mels=128, time_steps=256, power=2)
        self.assertEqual(msg.shape, (128, 256))

    def test_2(self):
        wave = self._wave_provider.get_audio_fragment(
            "XC602886.ogg", range_seconds=[0, 5]
        )

        c["MSG_NORMALIZE"] = True
        msg_p = get_msg_provider(c, key="_test_2_")
        msg = msg_p.msg(wave, n_mels=256, time_steps=256, power=3)
        self.assertEqual(msg.dtype, np.float32)
        self.assertAlmostEqual(np.std(msg), 1.0, delta=1e-7)  # type: ignore

    def test_3(self):
        wave = self._wave_provider.get_audio_fragment(
            "XC602886.ogg", range_seconds=[0, 5]
        )

        c["MSG_NORMALIZE"] = False
        # produced message about empty filter banks with these settings
        msg_p = get_msg_provider(c, key="_test_3_")
        msg = msg_p.msg(wave, n_mels=380, time_steps=380, power=2)
        self.assertEqual(msg.dtype, np.float32)
        self.assertAlmostEqual(np.std(msg), 9.416054, delta=1e-5)  # type: ignore


if __name__ == "__main__":
    unittest.main()
