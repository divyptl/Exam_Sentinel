"""tests/test_all.py — 35-test suite for ExamSentinel."""
import sys, time, json, unittest
from pathlib import Path
import torch, numpy as np
sys.path.append(str(Path(__file__).parent.parent))
Path("logs").mkdir(exist_ok=True)


class TestBayarConv(unittest.TestCase):
    def setUp(self):
        from core.forensics_filters import BayarConv2d
        self.layer = BayarConv2d(3, 3, 5)
    def test_output_shape(self):
        out = self.layer(torch.randn(2,3,64,64))
        self.assertEqual(out.shape, (2,3,64,64))
    def test_gradients(self):
        x = torch.randn(1,3,32,32,requires_grad=True)
        self.layer(x).mean().backward()
        self.assertIsNotNone(x.grad)
    def test_learnable(self):
        n = sum(p.numel() for p in self.layer.parameters() if p.requires_grad)
        self.assertGreater(n, 0)

class TestSRMFilterBank(unittest.TestCase):
    def setUp(self):
        from core.forensics_filters import SRMFilterBank
        self.srm = SRMFilterBank()
    def test_output_shape(self):
        out = self.srm(torch.randn(2,3,64,64))
        self.assertEqual(out.shape, (2,90,64,64))
    def test_not_learnable(self):
        params = list(self.srm.parameters())
        for p in params:
            self.assertFalse(p.requires_grad)
    def test_30_filters(self):
        self.assertEqual(self.srm.srm_weight.shape[0], 30)

class TestMoireDetector(unittest.TestCase):
    def setUp(self):
        from core.forensics_filters import MoireDetector
        self.det = MoireDetector(img_size=64)
    def test_output_shape(self):
        out = self.det(torch.randn(2,3,64,64))
        self.assertEqual(out.shape, (2,2,8,8))
    def test_moire_vs_clean(self):
        clean = torch.zeros(1,3,64,64)
        x = torch.arange(64).float()
        moire = clean.clone()
        moire[0,0] += torch.sin(2*3.14159*x/4).unsqueeze(0).expand(64,64)
        self.det.eval()
        with torch.no_grad():
            oc = self.det(clean).abs().mean()
            om = self.det(moire).abs().mean()
        self.assertGreater(om.item(), oc.item())

class TestForensicsExtractor(unittest.TestCase):
    def setUp(self):
        from core.forensics_filters import ForensicsFeatureExtractor
        self.ext = ForensicsFeatureExtractor(img_size=64)
    def test_shapes(self):
        f, m = self.ext(torch.randn(2,3,64,64))
        self.assertEqual(f.shape[1], 64)
        self.assertEqual(m.shape, (2,2,8,8))
    def test_small_param_count(self):
        n = sum(p.numel() for p in self.ext.parameters() if p.requires_grad)
        self.assertLess(n, 500_000)

class TestDataset(unittest.TestCase):
    def test_no_crash_empty(self):
        import warnings; warnings.filterwarnings("ignore")
        from core.dataset import ExamSentinelDataset
        ds = ExamSentinelDataset("data", split="train")
        self.assertEqual(len(ds), 0)
    def test_val_no_crash(self):
        import warnings; warnings.filterwarnings("ignore")
        from core.dataset import ExamSentinelDataset
        ds = ExamSentinelDataset("data", split="val")
        self.assertEqual(len(ds), 0)
    def test_train_transforms(self):
        from core.dataset import get_train_transforms
        import warnings; warnings.filterwarnings("ignore")
        img = np.random.randint(0,255,(128,128,3),dtype=np.uint8)
        out = get_train_transforms(64)(image=img)["image"]
        self.assertEqual(out.shape, (3,64,64))
    def test_val_transforms(self):
        from core.dataset import get_val_transforms
        img = np.random.randint(0,255,(128,128,3),dtype=np.uint8)
        out = get_val_transforms(64)(image=img)["image"]
        self.assertEqual(out.shape, (3,64,64))
    def test_webcam_dataset(self):
        from core.dataset import WebcamFrameDataset
        ds = WebcamFrameDataset(img_size=64)
        ds.add_frame(np.random.randint(0,255,(480,640,3),dtype=np.uint8))
        self.assertEqual(len(ds), 1)
        self.assertEqual(ds[0].shape, (3,64,64))

class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from core.model import ExamSentinelNet
        cls.model = ExamSentinelNet(img_size=128, pretrained=False)
        cls.model.eval()
    def test_all_heads_present(self):
        x = torch.randn(1,3,128,128)
        with torch.no_grad():
            out = self.model(x)
        for h in ["deepfake","recapture","splicing","forgery","combined"]:
            self.assertIn(h, out)
    def test_output_shapes(self):
        x = torch.randn(2,3,128,128)
        with torch.no_grad():
            out = self.model(x)
        for k,v in out.items():
            self.assertEqual(v.shape, (2,1), f"Bad shape for {k}: {v.shape}")
    def test_probabilities_in_range(self):
        x = torch.randn(2,3,128,128)
        with torch.no_grad():
            probs = self.model.get_probabilities(x)
        for k,p in probs.items():
            self.assertTrue((p>=0).all() and (p<=1).all(), f"Out of [0,1]: {k}")
    def test_different_inputs_differ(self):
        x1 = torch.zeros(1,3,128,128)
        x2 = torch.ones(1,3,128,128)
        with torch.no_grad():
            o1 = self.model(x1)["deepfake"]
            o2 = self.model(x2)["deepfake"]
        self.assertFalse(torch.allclose(o1,o2))
    def test_no_channel_mismatch(self):
        try:
            with torch.no_grad():
                _ = self.model(torch.randn(1,3,128,128))
            ok = True
        except RuntimeError as e:
            ok = False; self.fail(f"RuntimeError: {e}")
        self.assertTrue(ok)
    def test_param_count_reasonable(self):
        n = self.model.count_parameters()
        self.assertGreater(n, 1_000_000)
        self.assertLess(n, 30_000_000)

class TestLoss(unittest.TestCase):
    def setUp(self):
        from core.model import ExamSentinelNet, ExamSentinelLoss
        self.model   = ExamSentinelNet(img_size=128, pretrained=False)
        self.loss_fn = ExamSentinelLoss()
    def test_loss_positive(self):
        x = torch.randn(2,3,128,128)
        labels = torch.zeros(2,4); labels[0,0]=1
        with torch.no_grad():
            preds = self.model(x)
        losses = self.loss_fn(preds, labels)
        self.assertGreater(losses["loss_total"].item(), 0)
    def test_correct_pred_lower_loss(self):
        labels = torch.zeros(1,4); labels[0,0]=1
        good = {"deepfake":torch.tensor([[5.0]]),"recapture":torch.tensor([[0.0]]),
                "splicing":torch.tensor([[0.0]]),"forgery":torch.tensor([[0.0]]),
                "combined":torch.tensor([[4.0]])}
        bad  = {"deepfake":torch.tensor([[-5.0]]),"recapture":torch.tensor([[0.0]]),
                "splicing":torch.tensor([[0.0]]),"forgery":torch.tensor([[0.0]]),
                "combined":torch.tensor([[-4.0]])}
        self.assertLess(self.loss_fn(good,labels)["loss_total"].item(),
                        self.loss_fn(bad, labels)["loss_total"].item())

class TestInferenceResult(unittest.TestCase):
    def test_defaults(self):
        from core.inference_engine import InferenceResult
        r = InferenceResult()
        self.assertEqual(r.score_deepfake, 0.0)
        self.assertEqual(r.alert_tier, "GREEN")
        self.assertTrue(r.face_detected)
    def test_serialisation(self):
        from core.inference_engine import InferenceResult
        r = InferenceResult(score_deepfake=0.85, alert_tier="ORANGE")
        d = json.loads(r.to_json())
        self.assertAlmostEqual(d["score_deepfake"], 0.85)
        self.assertIn("timestamp_iso", d)

class TestMockEngine(unittest.TestCase):
    def test_all_scenarios(self):
        from core.inference_engine import MockInferenceEngine
        for s in ["normal","deepfake_attack","recapture_attempt","gaze_cheat"]:
            r = MockInferenceEngine(scenario=s).get_latest()
            self.assertIsNotNone(r)
            self.assertGreaterEqual(r.score_deepfake, 0)
    def test_non_blocking(self):
        from core.inference_engine import MockInferenceEngine
        eng = MockInferenceEngine("normal")
        t0 = time.time()
        for _ in range(20): eng.get_latest()
        self.assertLess(time.time()-t0, 1.0)

class TestDecisionEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import yaml
        from agents.decision_engine import AgenticDecisionEngine
        with open("configs/config.yaml") as f:
            cfg = yaml.safe_load(f)
        cls.cfg = cfg
        cls.engine = AgenticDecisionEngine(cfg, log_path="logs/test_dec.jsonl")

    def _r(self, **kw):
        from core.inference_engine import InferenceResult
        d = dict(score_deepfake=0.05,score_recapture=0.03,score_splicing=0.02,
                 score_forgery=0.02,score_combined=0.05,face_detected=True,
                 is_off_screen=False,sustained_offscreen_sec=0.0,
                 is_mouth_open=False,blink_rate=0.3)
        d.update(kw); return InferenceResult(**d)

    def test_green_on_clean(self):
        rec = self.engine.decide(self._r())
        self.assertEqual(rec.tier, "GREEN")
    def test_orange_on_deepfake(self):
        rec = self.engine.decide(self._r(score_deepfake=0.95,score_combined=0.88))
        self.assertIn(rec.tier, ["ORANGE","RED"])
    def test_orange_on_recapture(self):
        rec = self.engine.decide(self._r(score_recapture=0.93,score_combined=0.88))
        self.assertIn(rec.tier, ["ORANGE","RED"])
    def test_gaze_triggers_alert(self):
        import yaml
        from agents.decision_engine import AgenticDecisionEngine
        with open("configs/config.yaml") as f: cfg = yaml.safe_load(f)
        e = AgenticDecisionEngine(cfg, log_path="logs/test_gaze.jsonl")
        rec = e.decide(self._r(is_off_screen=True, sustained_offscreen_sec=5.5,
                                score_combined=0.15))
        self.assertIn(rec.tier, ["YELLOW","ORANGE","RED"])
    def test_auto_escalation_to_red(self):
        import yaml
        from agents.decision_engine import AgenticDecisionEngine
        with open("configs/config.yaml") as f: cfg = yaml.safe_load(f)
        e = AgenticDecisionEngine(cfg, log_path="logs/test_esc.jsonl")
        e.consecutive_orange = 3
        rec = e.decide(self._r(score_deepfake=0.95,score_combined=0.9))
        self.assertEqual(rec.tier, "RED")
    def test_qtable_updates_on_feedback(self):
        before = self.engine.q_table.copy()
        self.engine.decide(self._r(score_deepfake=0.85,score_combined=0.75))
        self.engine.receive_feedback(len(self.engine.decision_log)-1,"confirmed")
        self.assertFalse(np.allclose(self.engine.q_table, before))
    def test_session_stats_keys(self):
        stats = self.engine.get_session_stats()
        for k in ["total_decisions","tier_counts","current_tier","total_flags"]:
            self.assertIn(k, stats)
    def test_tier_ordering(self):
        from agents.decision_engine import AlertTier
        self.assertLess(AlertTier.GREEN.level,  AlertTier.YELLOW.level)
        self.assertLess(AlertTier.YELLOW.level, AlertTier.ORANGE.level)
        self.assertLess(AlertTier.ORANGE.level, AlertTier.RED.level)

class TestEndToEnd(unittest.TestCase):
    def _setup(self, scenario):
        import yaml
        from core.inference_engine import MockInferenceEngine
        from agents.decision_engine import AgenticDecisionEngine
        with open("configs/config.yaml") as f: cfg = yaml.safe_load(f)
        return (MockInferenceEngine(scenario=scenario),
                AgenticDecisionEngine(cfg, log_path=f"logs/e2e_{scenario}.jsonl"))

    def test_deepfake_triggers_alert(self):
        eng, dec = self._setup("deepfake_attack")
        tiers = set()
        for _ in range(25):
            tiers.add(dec.decide(eng.get_latest()).tier)
            time.sleep(0.02)
        self.assertGreater(len(tiers - {"GREEN"}), 0)

    def test_recapture_triggers_alert(self):
        eng, dec = self._setup("recapture_attempt")
        order = {"GREEN":0,"YELLOW":1,"ORANGE":2,"RED":3}
        max_t = "GREEN"
        for _ in range(30):
            t = dec.decide(eng.get_latest()).tier
            if order[t] > order[max_t]: max_t = t
            time.sleep(0.02)
        self.assertIn(max_t, ["YELLOW","ORANGE","RED"])

    def test_normal_mostly_green(self):
        eng, dec = self._setup("normal")
        highs = 0
        for _ in range(15):
            t = dec.decide(eng.get_latest()).tier
            if t in ["ORANGE","RED"]: highs += 1
            time.sleep(0.02)
        self.assertLessEqual(highs, 4)


if __name__ == "__main__":
    Path("logs").mkdir(exist_ok=True)
    loader = unittest.TestLoader()
    suite  = loader.loadTestsFromModule(__import__(__name__))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    passed = result.testsRun - len(result.failures) - len(result.errors)
    print(f"\n{'='*50}")
    if result.wasSuccessful():
        print(f"✓ All {result.testsRun} tests passed")
    else:
        print(f"✗ {len(result.failures)} failures, {len(result.errors)} errors")
        print(f"  Passed: {passed}/{result.testsRun}")
    print("="*50)
    sys.exit(0 if result.wasSuccessful() else 1)
