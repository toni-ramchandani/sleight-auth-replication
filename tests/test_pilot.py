import tempfile
import unittest
from pathlib import Path

from run_case import CASE, PROMPTS, build_plan
from summarize import summarize


class TestPilot(unittest.TestCase):
    def test_plan_has_both_labels_and_prompts(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            folder = root / "attacks" / "authorization_confusion" / CASE
            folder.mkdir(parents=True)
            (folder / "transcript.jsonl").write_text("test")
            (folder / "benign.jsonl").write_text("test")
            (root / "prompts").mkdir()
            for prompt in PROMPTS:
                (root / "prompts" / f"{prompt}.md").write_text("test")
            plan = build_plan(root, trials=2)
            self.assertEqual(8, len(plan))
            self.assertEqual(4, sum(not p["is_benign"] for p in plan))
            self.assertEqual(4, sum(p["is_benign"] for p in plan))

    def test_plan_missing_data_is_explicit(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FileNotFoundError):
                build_plan(Path(td), trials=1)

    def test_summary_does_not_invent_missing_scores(self):
        rows = [
            {"prompt_name": "base", "is_benign": False, "numeric_score": 80},
            {"prompt_name": "base", "is_benign": True, "numeric_score": 20},
            {"prompt_name": "base", "is_benign": False, "numeric_score": None},
        ]
        result = summarize(rows)["base"]
        self.assertEqual(60, result["mean_gap"])
        self.assertEqual(1, result["invalid_or_refusal"])
        self.assertEqual(1, result["attack_n"])


if __name__ == "__main__":
    unittest.main()
