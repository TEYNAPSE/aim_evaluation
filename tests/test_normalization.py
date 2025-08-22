import unittest

from evaluation import ImageEvaluator


class TestNormalization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.eval = ImageEvaluator.__new__(ImageEvaluator)
        # Enable flags for weight logic
        cls.eval.enable_clip = True
        cls.eval.enable_lpips = True
        cls.eval.enable_blip2 = True

    def test_normalize_clip_readme(self):
        self.assertEqual(self.eval.normalize_clip(0.4), 1.0)
        self.assertAlmostEqual(self.eval.normalize_clip(0.0), (0.0 + 1.0) / 1.4, places=6)
        self.assertAlmostEqual(self.eval.normalize_clip(-1.0), 0.0, places=6)

    def test_normalize_blip2_readme(self):
        self.assertEqual(self.eval.normalize_blip2(0.7), 1.0)
        self.assertAlmostEqual(self.eval.normalize_blip2(0.35), 0.5, places=6)
        self.assertAlmostEqual(self.eval.normalize_blip2(0.0), 0.0, places=6)

    def test_normalize_lpips_readme(self):
        self.assertEqual(self.eval.normalize_lpips(0.6), 1.0)
        self.assertEqual(self.eval.normalize_lpips(0.65), 0.9)
        self.assertAlmostEqual(self.eval.normalize_lpips(0.85), 0.89 - ((0.85 - 0.75) / 0.2) * (0.89 - 0.1), places=6)
        self.assertEqual(self.eval.normalize_lpips(0.96), 0.0)

    def test_calculate_final_score_weights(self):
        # with all enabled, weights 0.2/0.3/0.5 should produce 100 when all norms=1
        self.assertEqual(self.eval.calculate_final_score(1.0, 1.0, 1.0), 100.0)


if __name__ == '__main__':
    unittest.main()
