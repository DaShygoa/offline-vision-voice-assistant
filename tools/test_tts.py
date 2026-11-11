"""TTS mode and fallback regressions without network or audio devices."""
import unittest
from unittest.mock import patch
from voice.tts import TtsClient

class TtsTests(unittest.TestCase):
    def test_edge_failure_uses_offline_speech(self):
        client = TtsClient({'mode': 'edge'})
        with patch.object(client, '_edge_synth', side_effect=RuntimeError('offline')), \
             patch.object(client, '_offline_synth', return_value='local.wav') as offline, \
             patch.object(client, '_mock_synth') as mock, \
             patch.object(client, '_wav_duration', return_value=1):
            self.assertEqual(client.synthesize('你好'), 'local.wav')
            offline.assert_called_once_with('你好')
            mock.assert_not_called()

    def test_missing_offline_model_reports_failure_not_beep(self):
        client = TtsClient({'mode': 'offline', 'offline_model_dir': 'missing-tts-model'})
        with patch.object(client, '_mock_synth') as mock:
            self.assertIsNone(client.synthesize('你好'))
            self.assertEqual(client.health(), 'down:offline')
            mock.assert_not_called()

if __name__ == '__main__':
    unittest.main()
