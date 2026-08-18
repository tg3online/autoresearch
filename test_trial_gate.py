import tempfile
import unittest
from pathlib import Path

import trial_gate


class TrialGateTests(unittest.TestCase):
    def test_parse_metrics(self):
        text = """noise
val_bpb:          2.749264
training_seconds: 10.1
peak_vram_mb:     659.8
num_steps:        49
"""
        self.assertEqual(trial_gate.parse_metrics(text)["val_bpb"], 2.749264)
        self.assertEqual(trial_gate.parse_metrics(text)["peak_vram_mb"], 659.8)
        self.assertEqual(trial_gate.parse_metrics(text)["num_steps"], 49.0)

    def test_parse_metrics_missing_is_not_invented(self):
        self.assertEqual(trial_gate.parse_metrics("training failed"), {})

    def test_parse_metrics_accepts_scientific_notation(self):
        metrics = trial_gate.parse_metrics("val_bpb: 2.5e+00\npeak_vram_mb: 6.6E2\n")
        self.assertEqual(metrics, {"val_bpb": 2.5, "peak_vram_mb": 660.0})

    def test_sha256_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample"
            path.write_bytes(b"abc")
            self.assertEqual(
                trial_gate.sha256_file(path),
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            )


if __name__ == "__main__":
    unittest.main()
