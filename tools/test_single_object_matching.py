"""Regression checks for single-object rejection and comparable displayed scores."""
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from core.app import AssistantApp
from vision.memory import VisualMemory

class SingleObjectMatchingTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        path=self.tmp.name
        self.memory=VisualMemory({'db_path':os.path.join(path,'db.sqlite'),
            'embeddings_path':os.path.join(path,'bank.npz'),'samples_dir':os.path.join(path,'samples'),
            'embed_dim':4,'center_features':True,'match_threshold':0.7,'single_match_threshold':0.85,
            'match_threshold_centered':0.5,'margin_threshold_centered':0.15})
        self.addCleanup(self.memory.close)
        self.memory.add_object('手机',None,[np.array([1,0,0,0],np.float32)]*3,[])

    def test_one_object_uses_conservative_threshold_even_with_multiple_samples(self):
        self.assertFalse(self.memory.centering_active)
        self.assertEqual(self.memory.threshold,0.85)
        unknown=np.array([0.8,0.6,0,0],np.float32)
        known=np.array([0.95,np.sqrt(1-0.95**2),0,0],np.float32)
        self.assertLess(self.memory.best_match(unknown).score,self.memory.threshold)
        self.assertGreater(self.memory.best_match(known).score,self.memory.threshold)

    def test_raw_score_stable_when_other_object_is_added_and_removed(self):
        query=np.array([0.9,np.sqrt(1-0.9**2),0,0],np.float32)
        original=self.memory.best_match(query)
        self.memory.add_object('鼠标',None,[np.array([0,1,0,0],np.float32)]*3,[])
        self.assertTrue(self.memory.centering_active)
        self.assertEqual(self.memory.threshold,0.5)
        after=next(m for m in self.memory.top_k(query,2) if m.user_name=='手机')
        self.assertTrue(after.centered)
        self.assertAlmostEqual(original.raw_score,after.raw_score,places=6)
        self.assertNotAlmostEqual(after.score,after.raw_score,places=3)
        self.memory.delete_object('鼠标')
        self.assertEqual(self.memory.threshold,0.85)
        self.assertAlmostEqual(self.memory.best_match(query).raw_score,original.raw_score,places=6)

    def test_rotations_cannot_bypass_single_object_threshold(self):
        app=object.__new__(AssistantApp); app.memory=self.memory; app._embed_crop=lambda crop: np.array([0.8,0.6,0,0],np.float32)
        _,match=app._match_crop(np.zeros((40,40,3),np.uint8))
        self.assertLess(match.score,self.memory.threshold)

    def test_centered_unknown_explains_both_scales(self):
        app=object.__new__(AssistantApp); app.stats=SimpleNamespace(last_similarity=0); app.memory=self.memory
        self.memory.add_object('鼠标',None,[np.array([0,1,0,0],np.float32)],[])
        match=SimpleNamespace(user_name='手机',score=0.42,raw_score=0.88,centered=True)
        text=app._describe_unknown(match)
        self.assertIn('原始相似度88%',text); self.assertIn('中心化判定分数42%',text); self.assertIn('阈值50%',text)

    def test_accepted_result_does_not_claim_same_physical_object(self):
        app=object.__new__(AssistantApp); app.stats=SimpleNamespace(last_similarity=0)
        match=SimpleNamespace(user_name='鼠标',score=0.8,raw_score=0.9,centered=True)
        text=app._describe_known(match)
        self.assertIn('匹配到已录入的「鼠标」',text)
        self.assertNotIn('这是你的',text)

    def test_local_file_does_not_accept_ambiguous_multi_object_result(self):
        app=object.__new__(AssistantApp); app.stats=SimpleNamespace(embed_latency_ms=0)
        app.resnet=SimpleNamespace(ready=True,timed_embed=lambda image:(np.ones(4),1.0))
        app.memory=SimpleNamespace(threshold=0.5,margin_threshold=0.15,top_k=lambda *a,**k:[SimpleNamespace(score=0.8),SimpleNamespace(score=0.75)])
        with patch('core.app.imread_unicode',return_value=np.zeros((80,80,3),np.uint8)):
            self.assertIn('不太确定',app.recognize_image_file('test.png'))

if __name__=='__main__': unittest.main()
