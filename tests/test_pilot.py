import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from run_case import (BENCHMARK_COMMIT, CASE, PROMPTS, build_plan,
                      call_monitor, default_output, load_completed, sha256)
from summarize import summarize


class TestPilot(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.folder = self.root / "attacks" / "authorization_confusion" / CASE
        self.folder.mkdir(parents=True)
        (self.folder / "transcript.jsonl").write_text("attack")
        (self.folder / "benign.jsonl").write_text("benign")
        (self.root / "prompts").mkdir()
        for prompt in PROMPTS:
            (self.root / "prompts" / f"{prompt}.md").write_text(f"prompt:{prompt}")

    def test_plan_has_both_labels_and_prompts(self):
        plan = build_plan(self.root, trials=2)
        self.assertEqual(8, len(plan))
        self.assertEqual(4, sum(not p["is_benign"] for p in plan))
        self.assertEqual(4, sum(p["is_benign"] for p in plan))

    def test_plan_missing_data_is_explicit(self):
        with self.assertRaises(FileNotFoundError):
            build_plan(Path(self.tmp.name) / "empty", trials=1)

    def test_openai_request_params_and_no_temperature(self):
        response = SimpleNamespace(
            output_text="<score>84</score>", status="completed",
            usage=SimpleNamespace(input_tokens=250, output_tokens=40,
                                  output_tokens_details=SimpleNamespace(reasoning_tokens=20)),
        )
        client = SimpleNamespace(responses=SimpleNamespace(create=Mock(return_value=response)))
        answer, refused, reason, usage = call_monitor(
            "openai", "gpt-6.1-sol", "medium", "system instructions", "transcript", client,
        )
        self.assertEqual("<score>84</score>", answer)
        self.assertFalse(refused)
        self.assertEqual("completed", reason)
        self.assertEqual(20, usage["reasoning_tokens"])
        params = client.responses.create.call_args.kwargs
        self.assertEqual("gpt-6.1-sol", params["model"])
        self.assertEqual({"effort": "medium"}, params["reasoning"])
        self.assertFalse(params["store"])
        self.assertNotIn("temperature", params)
        self.assertEqual("system instructions", params["instructions"])

    def test_openai_incomplete_is_not_scored(self):
        response = SimpleNamespace(status="incomplete", output_text="<score>90</score>", usage=None)
        client = SimpleNamespace(responses=SimpleNamespace(create=Mock(return_value=response)))
        answer, _, stop_reason, _ = call_monitor("openai", "gpt-6.1-sol", "medium", None, "text", client)
        self.assertEqual("", answer)
        self.assertEqual("response_incomplete", stop_reason)

    def test_output_paths_separate_models_and_providers(self):
        self.assertNotEqual(default_output("openai", "gpt-6.1-sol", 1),
                            default_output("anthropic", "claude-opus-4-6", 1))

    def test_resume_rejects_incompatible_results(self):
        path = self.root / "results.jsonl"
        item = build_plan(self.root, 1)[0]
        row = {
            "provider": "anthropic", "model": "claude-opus-4-6",
            "reasoning_effort": None, "benchmark_commit": BENCHMARK_COMMIT,
            "case": CASE, "prompt_name": item["prompt_name"],
            "is_benign": item["is_benign"], "trial": item["trial"],
            "transcript_sha256": sha256(item["transcript_path"]),
            "prompt_sha256": sha256(self.root / "prompts" / f"{item['prompt_name']}.md"),
        }
        path.write_text(json.dumps(row) + "\n")
        with self.assertRaises(RuntimeError):
            load_completed(path, provider="openai", model="gpt-6.1-sol",
                           revision=BENCHMARK_COMMIT, plan=build_plan(self.root, 1),
                           root=self.root, resume=True, reasoning_effort="medium")

    def test_resume_skips_matching_results(self):
        path = self.root / "results.jsonl"
        item = build_plan(self.root, 1)[0]
        row = {
            "provider": "openai", "model": "gpt-6.1-sol",
            "reasoning_effort": "medium", "benchmark_commit": BENCHMARK_COMMIT,
            "case": CASE, "prompt_name": item["prompt_name"],
            "is_benign": item["is_benign"], "trial": item["trial"],
            "transcript_sha256": sha256(item["transcript_path"]),
            "prompt_sha256": sha256(self.root / "prompts" / f"{item['prompt_name']}.md"),
        }
        path.write_text(json.dumps(row) + "\n")
        done = load_completed(path, provider="openai", model="gpt-6.1-sol",
                              revision=BENCHMARK_COMMIT, plan=build_plan(self.root, 1),
                              root=self.root, resume=True, reasoning_effort="medium")
        self.assertEqual(1, len(done))

    def test_summary_separates_providers_and_does_not_invent_scores(self):
        rows = [
            {"provider": "openai", "model": "gpt-6.1-sol", "prompt_name": "base", "is_benign": False, "numeric_score": 80},
            {"provider": "openai", "model": "gpt-6.1-sol", "prompt_name": "base", "is_benign": True, "numeric_score": 20},
            {"provider": "openai", "model": "gpt-6.1-sol", "prompt_name": "base", "is_benign": False, "numeric_score": None},
            {"provider": "anthropic", "model": "claude-opus-4-6", "prompt_name": "base", "is_benign": False, "numeric_score": 70},
        ]
        result = summarize(rows)
        self.assertEqual(60, result["openai/gpt-6.1-sol/base"]["mean_gap"])
        self.assertEqual(1, result["openai/gpt-6.1-sol/base"]["invalid_or_refusal"])
        self.assertIsNone(result["anthropic/claude-opus-4-6/base"]["mean_gap"])


if __name__ == "__main__":
    unittest.main()
