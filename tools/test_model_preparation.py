"""Model preparation leaves only the runtime copy of the ASR model."""

import shutil
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import prepare_models


class AsrPreparationTests(unittest.TestCase):
    def test_download_extract_and_speech_sample_use_temporary_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source" / "paraformer"
            source.mkdir(parents=True)
            (source / "tokens.txt").write_text("token", encoding="utf-8")
            (source / "model.int8.onnx").write_bytes(b"model")
            (source / "am.mvn").write_bytes(b"cmvn")
            (source / "test_wavs").mkdir()
            (source / "test_wavs" / "sample.wav").write_bytes(b"audio")
            archive = root / "source.tar.bz2"
            with tarfile.open(archive, "w:bz2") as tar:
                tar.add(source, arcname="paraformer")
            models = root / "models"
            models.mkdir()
            downloaded_to = []

            def download(_url, dest, _desc):
                downloaded_to.append(Path(dest))
                shutil.copyfile(archive, dest)

            with patch.object(prepare_models, "MODELS", str(models)), \
                 patch.object(prepare_models, "ROOT", str(root)), \
                 patch.object(prepare_models, "_download", side_effect=download), \
                 patch.object(prepare_models, "_prepare_vad"):
                result = Path(prepare_models.prepare_asr())

            self.assertEqual((result / "model.int8.onnx").read_bytes(), b"model")
            self.assertEqual((root / "data" / "asr_sample.wav").read_bytes(), b"audio")
            self.assertFalse(downloaded_to[0].exists())
            self.assertFalse((models / "_paraformer_extract").exists())
            self.assertFalse((models / "paraformer_zh.tar.bz2").exists())

    def test_existing_asr_still_prepares_vad(self):
        with tempfile.TemporaryDirectory() as tmp:
            asr = Path(tmp) / "sherpa-onnx-paraformer-zh-int8"
            asr.mkdir()
            (asr / "tokens.txt").write_text("token", encoding="utf-8")
            (asr / "model.int8.onnx").write_bytes(b"model")
            with patch.object(prepare_models, "MODELS", tmp), \
                 patch.object(prepare_models, "_prepare_vad") as vad:
                prepare_models.prepare_asr()
            vad.assert_called_once()

    def test_existing_int8_is_not_overwritten_by_float_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "model.int8.onnx").write_bytes(b"int8")
            (root / "model.onnx").write_bytes(b"float")
            prepare_models._alias_int8(tmp)
            self.assertEqual((root / "model.int8.onnx").read_bytes(), b"int8")

    def test_main_reports_missing_model_as_failure(self):
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(prepare_models, "MODELS", tmp), \
             patch.object(prepare_models, "prepare_resnet18", return_value=""):
            self.assertEqual(prepare_models.main(["--resnet18"]), 1)


if __name__ == "__main__":
    unittest.main()
