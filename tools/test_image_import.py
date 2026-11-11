"""Image upload enrollment integration tests (no camera or model required)."""
import io
import os
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from core.app import AssistantApp
from vision.memory import VisualMemory
from vision.object_cropper import ObjectCropper
from web import server

class ImageImportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = self.tmp.name
        self.assistant = object.__new__(AssistantApp)
        self.assistant.cfg = {'memory': {'min_samples': 3, 'max_samples': 4, 'samples_dir': os.path.join(root, 'samples')}}
        self.assistant.memory = VisualMemory({'db_path': os.path.join(root, 'bank.db'), 'embeddings_path': os.path.join(root, 'bank.npz'), 'samples_dir': os.path.join(root, 'samples'), 'embed_dim': 4})
        self.addCleanup(self.assistant.memory.close)
        self.assistant.cropper = ObjectCropper(min_crop_px=64)
        self.assistant.resnet = SimpleNamespace(ready=True, device_used='test', embed=lambda img: np.array([1, 0, 0, 0], dtype=np.float32))
        self.assistant.stats = SimpleNamespace(mem_objects=0, mem_info='')
        self.client = server.app.test_client()
        self.patcher = patch.object(server, '_assistant', self.assistant)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        image = np.random.default_rng(12).integers(0, 256, (80, 80, 3), dtype=np.uint8)
        self.image = cv2.imencode('.png', image)[1].tobytes()

    def upload(self, name, images):
        return self.client.post('/objects/import', data={'name': name, 'images': [(io.BytesIO(data), f'{i}.png') for i, data in enumerate(images)]}, content_type='multipart/form-data')

    def test_success_and_duplicate_preserves_existing_samples(self):
        response = self.upload('我的杯子', [self.image] * 3)
        self.assertEqual(response.status_code, 200, response.get_json())
        paths = self.assistant.memory.sample_images('我的杯子')
        self.assertEqual(len(paths), 3)
        originals = [Path(p).read_bytes() for p in paths]
        response = self.upload('我的杯子', [self.image] * 3)
        self.assertEqual(response.status_code, 409)
        self.assertEqual(self.assistant.memory.sample_images('我的杯子'), paths)
        self.assertEqual([Path(p).read_bytes() for p in paths], originals)
        self.assertEqual(len(os.listdir(self.assistant.cfg['memory']['samples_dir'])), 1)
        self.assertEqual(self.client.get('/objects/sample?name=我的杯子').status_code, 200)

    def test_invalid_images_and_count_do_not_create_object(self):
        self.assertEqual(self.upload('杯子', [self.image] * 2).status_code, 400)
        self.assertEqual(self.upload('杯子', [self.image, b'bad', self.image]).status_code, 400)
        self.assertEqual(self.assistant.memory.stats()['objects'], 0)
        self.assertEqual(os.listdir(self.assistant.cfg['memory']['samples_dir']), [])

    def test_filename_is_not_used_as_server_path(self):
        response = self.client.post('/objects/import', data={'name': '../物品', 'images': [(io.BytesIO(self.image), '../../unsafe.png') for _ in range(3)]}, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 200)
        for path in self.assistant.memory.sample_images('../物品'):
            self.assertTrue(os.path.commonpath([path, self.assistant.cfg['memory']['samples_dir']]) == self.assistant.cfg['memory']['samples_dir'])

if __name__ == '__main__':
    unittest.main()
