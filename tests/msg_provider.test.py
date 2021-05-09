import unittest

from src.config_test import c
from src.services import get_msg_provider, get_wave_provider


class Test_MSG_Provider(unittest.TestCase):
    def setUp(self):
        self._wave_provider = get_wave_provider(c)
        self._msg_provider = get_msg_provider(c)

    def test_1(self):
        wave = self._wave_provider.get_audio_fragment(
            "XC11209.ogg", range_seconds=[0, 5]
        )
        msg = self._msg_provider.msg(wave)

        self.assertEqual(
            msg.shape,
            (
                c["MSG_TARGET_SIZE"]["freqs"],
                c["MSG_TARGET_SIZE"]["time"],
            ),
        )

    def test_2(self):
        wave = self._wave_provider.get_audio_fragment(
            "XC602886.ogg", range_seconds=[0, 5]
        )

        msg = self._msg_provider.msg(wave)

        print(msg)


if __name__ == "__main__":
    unittest.main(warnings="ignore")
