import unittest

import baseline_report


class BaselineReportTests(unittest.TestCase):
    def sample(self, seed, value, peak=700.0, train_hash="train"):
        return {
            "seed": seed,
            "metrics": {"val_bpb": value, "peak_vram_mb": peak},
            "source": {
                "files": {
                    "train.py": train_hash,
                    "prepare.py": "prepare",
                    "uv.lock": "lock",
                }
            },
            "_manifest_path": f"manifest-{seed}.json",
        }

    def test_provisional_summary(self):
        report = baseline_report.summarize([self.sample(42, 2.7)])
        self.assertEqual(report["status"], "provisional")
        self.assertEqual(report["val_bpb"]["median"], 2.7)

    def test_established_summary(self):
        report = baseline_report.summarize(
            [self.sample(41, 2.8), self.sample(42, 2.7), self.sample(43, 2.6)]
        )
        self.assertEqual(report["status"], "established")
        self.assertEqual(report["val_bpb"]["median"], 2.7)

    def test_rejects_mixed_sources(self):
        with self.assertRaises(ValueError):
            baseline_report.summarize(
                [self.sample(42, 2.7), self.sample(43, 2.6, train_hash="other")]
            )


if __name__ == "__main__":
    unittest.main()
