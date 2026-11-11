"""Experiment 3 offline checks: synthetic strings only, no benchmark/API access."""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from analyze_context import analyze_context
from run_case import BENCHMARK_COMMIT, PROMPTS
from run_context import (
    CONTEXT_MODES, FORMAT_VERSION, TARGET_INSTRUCTION, TWO_STAGE_SCENARIOS,
    build_context_plan, load_context_completed, main, output_path,
    render_context, select_scenarios,
)


class SyntheticContextFixture(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        (self.root / "prompts").mkdir()
        for name in PROMPTS:
            (self.root / "prompts" / f"{name}.md").write_text(f"MONITOR={name}\n<score>N</score>\n")
        for scenario in TWO_STAGE_SCENARIOS:
            for stage in (1, 2):
                parent = self.root / "attacks" / "authorization_confusion" / f"{scenario}-stage{stage}"
                parent.mkdir(parents=True)
                for benign in (False, True):
                    name = "benign.jsonl" if benign else "transcript.jsonl"
                    label = "BENIGN" if benign else "ATTACK"
                    (parent / name).write_text(f"SYNTHETIC_{label}_{scenario}_S{stage}")
        self.plan = build_context_plan(self.root)

    @contextlib.contextmanager
    def fake_upstream(self):
        # Unlike the real benchmark, fixture content is intentionally not JSONL.
        # Verify we call *the imported benchmark helpers* for both modes.
        transcripts = ModuleType("evaluate.core.transcript")
        transcripts.strip_assistant_text = Mock(side_effect=lambda text: "STRIPPED:" + text)
        transcripts.render_prompt = Mock(side_effect=lambda template, evidence: (template, f"<transcript>\n{evidence}\n</transcript>"))
        parsing = ModuleType("evaluate.core.response")
        parsing.parse_response = lambda text: {"numeric_score": int(text.split("<score>")[1].split("</score>")[0])}
        modules = {
            "evaluate": ModuleType("evaluate"),
            "evaluate.core": ModuleType("evaluate.core"),
            "evaluate.core.transcript": transcripts,
            "evaluate.core.response": parsing,
        }
        with patch.dict(sys.modules, modules):
            yield transcripts

    def record(self, item, *, score=80, provider="openai", effort="medium"):
        with self.fake_upstream():
            _, _, hashes = render_context(item, self.root)
        return {
            "format_version": FORMAT_VERSION,
            "provider": provider, "model": "gpt-6.1-sol" if provider == "openai" else "claude-opus-4-6",
            "reasoning_effort": effort if provider == "openai" else None,
            "temperature": None if provider == "openai" else 1.0,
            "thinking": True, "strip_assistant": True,
            "benchmark_commit": BENCHMARK_COMMIT,
            "scenario": item.scenario, "target_stage": 2, "stage_count": 2,
            "context_mode": item.context_mode, "prompt_name": item.prompt_name,
            "is_benign": item.is_benign, "trial": item.trial,
            **hashes,
            "numeric_score": score,
            "status": "scored" if score is not None else "refusal_or_invalid",
        }


class PlanTests(SyntheticContextFixture):
    def test_plan_is_16_conditions_across_two_scenarios(self):
        self.assertEqual(16, len(self.plan))
        self.assertEqual(16, len({item.key for item in self.plan}))
        self.assertEqual(set(CONTEXT_MODES), {p.context_mode for p in self.plan})
        self.assertEqual({True, False}, {p.is_benign for p in self.plan})
        self.assertEqual(2, len({p.scenario for p in self.plan}))

    def test_ten_trials_is_160_calls(self):
        self.assertEqual(160, len(build_context_plan(self.root, trials=10)))

    def test_selected_scenario_is_eight_conditions(self):
        selected = (TWO_STAGE_SCENARIOS[0],)
        self.assertEqual(8, len(build_context_plan(self.root, 1, selected)))

    def test_cumulative_only_is_eight_conditions(self):
        self.assertEqual(8, len(build_context_plan(self.root, 1, modes=("cumulative",))))

    def test_duplicate_modes_or_scenarios_rejected(self):
        with self.assertRaises(ValueError):
            build_context_plan(self.root, modes=("isolated", "isolated"))
        with self.assertRaises(ValueError):
            select_scenarios(TWO_STAGE_SCENARIOS * 2)
        with self.assertRaises(ValueError):
            select_scenarios(("preprocess-sabotage",))

    def test_missing_any_input_fails_precheck(self):
        p = self.plan[0].stage1_path
        p.unlink()
        with self.assertRaises(FileNotFoundError):
            build_context_plan(self.root)

    def test_trial_range(self):
        for value in (0, 11):
            with self.assertRaises(ValueError):
                build_context_plan(self.root, trials=value)

    def test_output_is_separate_from_experiment_two(self):
        target = output_path("openai", "gpt-6.1-sol", 1, TWO_STAGE_SCENARIOS, CONTEXT_MODES)
        self.assertIn("context", str(target))
        self.assertNotEqual(target, Path("results/openai-gpt-6.1-sol-auth6-t1.jsonl"))
        self.assertNotEqual(target, output_path("anthropic", "claude-opus-4-6", 1, TWO_STAGE_SCENARIOS, CONTEXT_MODES))

    def test_dry_run_does_not_call_provider_or_write_results(self):
        out = io.StringIO()
        with patch("run_context.check_revision", return_value=BENCHMARK_COMMIT):
            with patch("run_context.call_monitor") as monitor:
                with contextlib.redirect_stdout(out):
                    status = main(["--benchmark-dir", str(self.root)])
        self.assertEqual(0, status)
        self.assertIn("Planned requests: 16", out.getvalue())
        self.assertIn("DRY RUN COMPLETE", out.getvalue())
        monitor.assert_not_called()


class TranscriptTests(SyntheticContextFixture):
    def test_isolated_and_cumulative_share_stage2_payload(self):
        isolated, cumulative = self.plan[:2]
        self.assertEqual(isolated.context_mode, "isolated")
        self.assertEqual(cumulative.context_mode, "cumulative")
        with self.fake_upstream() as upstream:
            system_a, user_a, hashes_a = render_context(isolated, self.root)
            system_b, user_b, hashes_b = render_context(cumulative, self.root)
        self.assertEqual(system_a, system_b)
        self.assertIn("STRIPPED:SYNTHETIC_ATTACK_", user_a)
        self.assertIn('index="2" role="target"', user_a)
        self.assertNotIn('index="1"', user_a)
        self.assertIn('index="1" role="prior_context"', user_b)
        self.assertIn('index="2" role="target"', user_b)
        stage2_section = user_a.split('<session index="2" role="target">', 1)[1]
        self.assertIn(stage2_section, user_b)
        self.assertTrue(user_a.startswith(TARGET_INSTRUCTION))
        self.assertTrue(user_b.startswith(TARGET_INSTRUCTION))
        self.assertEqual(hashes_a["stage1_sha256"], hashes_b["stage1_sha256"])
        self.assertEqual(hashes_a["stage2_sha256"], hashes_b["stage2_sha256"])
        self.assertNotEqual(hashes_a["payload_sha256"], hashes_b["payload_sha256"])
        self.assertEqual(3, upstream.strip_assistant_text.call_count)

    def test_benign_context_does_not_include_harmful_examples(self):
        benign = next(item for item in self.plan if item.is_benign and item.context_mode == "cumulative")
        with self.fake_upstream():
            _, user, _ = render_context(benign, self.root)
        self.assertIn("SYNTHETIC_BENIGN_", user)
        self.assertNotIn("SYNTHETIC_ATTACK_", user)

    def test_payload_change_changes_signature(self):
        item = self.plan[0]
        with self.fake_upstream():
            a = render_context(item, self.root)[2]["payload_sha256"]
            with patch("run_context.TARGET_INSTRUCTION", TARGET_INSTRUCTION + " NEW"):
                b = render_context(item, self.root)[2]["payload_sha256"]
        self.assertNotEqual(a, b)


class ResumeTests(SyntheticContextFixture):
    def setUp(self):
        super().setUp()
        self.target = self.root / "results.jsonl"
        self.kwargs = dict(provider="openai", model="gpt-6.1-sol", revision=BENCHMARK_COMMIT,
                           effort="medium", resume=True)

    def resume(self):
        with self.fake_upstream():
            return load_context_completed(self.target, self.plan, self.root, **self.kwargs)

    def test_one_record_can_resume(self):
        self.target.write_text(json.dumps(self.record(self.plan[0])) + "\n")
        self.assertEqual({self.plan[0].key}, self.resume())

    def test_existing_requires_resume(self):
        self.target.write_text(json.dumps(self.record(self.plan[0])) + "\n")
        self.kwargs["resume"] = False
        with self.assertRaisesRegex(RuntimeError, "--resume"):
            self.resume()

    def test_duplicate_rejected(self):
        row = json.dumps(self.record(self.plan[0]))
        self.target.write_text(row + "\n" + row + "\n")
        with self.assertRaisesRegex(RuntimeError, "repeated"):
            self.resume()

    def test_changed_stage1_rejected_even_for_isolated_mode(self):
        self.target.write_text(json.dumps(self.record(self.plan[0])) + "\n")
        self.plan[0].stage1_path.write_text("CHANGED")
        with self.assertRaisesRegex(RuntimeError, "provenance"):
            self.resume()

    def test_changed_stage2_rejected(self):
        self.target.write_text(json.dumps(self.record(self.plan[0])) + "\n")
        self.plan[0].stage2_path.write_text("CHANGED")
        with self.assertRaisesRegex(RuntimeError, "provenance"):
            self.resume()

    def test_modified_prompt_or_wrapper_rejected(self):
        self.target.write_text(json.dumps(self.record(self.plan[0])) + "\n")
        with patch("run_context.TARGET_INSTRUCTION", TARGET_INSTRUCTION + " revised"):
            with self.assertRaisesRegex(RuntimeError, "rendering"):
                self.resume()

    def test_refusal_is_recorded_not_resubmitted(self):
        self.target.write_text(json.dumps(self.record(self.plan[0], score=None)) + "\n")
        self.assertEqual({self.plan[0].key}, self.resume())

    def test_incompatible_provider_rejected(self):
        bad = self.record(self.plan[0])
        bad["provider"] = "anthropic"
        self.target.write_text(json.dumps(bad) + "\n")
        with self.assertRaises(RuntimeError):
            self.resume()


class AnalysisTests(SyntheticContextFixture):
    def sample(self):
        rows = []
        for item in self.plan:
            if item.is_benign:
                score = 10 if item.context_mode == "isolated" else 14
            else:
                score = 20 if item.context_mode == "isolated" else 75
            rows.append(self.record(item, score=score))
        return rows

    def test_full_paired_comparison(self):
        r = analyze_context(self.sample())
        self.assertEqual(16, r["scope"]["records"])
        self.assertEqual(2, r["scope"]["scenario_count"])
        for group in r["by_scenario_and_prompt"].values():
            paired = group["paired_comparison"]
            self.assertEqual(1, paired["complete_matched_trials"])
            self.assertEqual(55, paired["attack_mean_change"])
            self.assertEqual(4, paired["benign_mean_change"])
            self.assertEqual(51, paired["mean_attack_minus_benign_change"])
            self.assertEqual(0, group["isolated"]["attack"]["at_or_above_illustrative_threshold"])
            self.assertEqual(1, group["cumulative"]["attack"]["at_or_above_illustrative_threshold"])

    def test_missing_arm_prevents_paired_claim(self):
        rows = [row for row in self.sample() if not (row["scenario"] == TWO_STAGE_SCENARIOS[0] and row["context_mode"] == "cumulative")]
        report = analyze_context(rows)
        g = report["by_scenario_and_prompt"][TWO_STAGE_SCENARIOS[0] + "/" + PROMPTS[0]]
        self.assertEqual(0, g["paired_comparison"]["complete_matched_trials"])

    def test_invalid_score_is_not_success_or_comparison(self):
        rows = self.sample()
        rows[0]["numeric_score"] = None
        rows[0]["status"] = "refusal_or_invalid"
        report = analyze_context(rows)
        g = report["by_scenario_and_prompt"][TWO_STAGE_SCENARIOS[0] + "/" + PROMPTS[0]]
        self.assertEqual(1, g["isolated"]["invalid_attack"])
        self.assertEqual(0, g["paired_comparison"]["attack_trials"])

    def test_duplicate_rejected(self):
        r = self.sample()
        with self.assertRaisesRegex(ValueError, "Duplicate condition"):
            analyze_context(r + [r[0]])

    def test_inconsistent_provenance_rejected(self):
        r = self.sample()
        r[0]["stage1_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "Transcript hashes"):
            analyze_context(r)

    def test_effective_input_mismatch_between_trials_rejected(self):
        rows = self.sample()
        repeat = dict(rows[0])
        repeat["trial"] = 1
        repeat["payload_sha256"] = "d" * 64
        with self.assertRaisesRegex(ValueError, "Effective model input changed"):
            analyze_context(rows + [repeat])

    def test_mixed_models_rejected(self):
        r = self.sample()
        r[0]["model"] = "gpt-other"
        with self.assertRaisesRegex(ValueError, "Mixed model"):
            analyze_context(r)

    def test_bogus_outcome_rejected(self):
        r = self.sample()
        r[0]["numeric_score"] = 121
        with self.assertRaisesRegex(ValueError, "Invalid scored"):
            analyze_context(r)

    def test_invalid_threshold(self):
        with self.assertRaises(ValueError):
            analyze_context(self.sample(), threshold=200)


class ProviderIntegrationTests(SyntheticContextFixture):
    def _mock_run(self, provider):
        # Execute the CLI using an in-memory client. A token-bearing
        # environment variable is mocked; no real request is allowed.
        fake_sdk = ModuleType(provider)
        ctor = Mock(return_value=SimpleNamespace())
        setattr(fake_sdk, "OpenAI" if provider == "openai" else "Anthropic", ctor)
        destination = self.root / f"{provider}.jsonl"
        cmd = ["--benchmark-dir", str(self.root), "--provider", provider,
               "--scenario", TWO_STAGE_SCENARIOS[0], "--trials", "1", "--output", str(destination), "--execute"]
        fake_usage = {"input_tokens": 10, "output_tokens": 5, "reasoning_tokens": 0}
        with self.fake_upstream(), patch.dict(sys.modules, {provider: fake_sdk}):
            with patch("run_context.check_revision", return_value=BENCHMARK_COMMIT), \
                 patch("run_context.call_monitor", return_value=("<score>70</score>", False, "completed", fake_usage)) as monitor, \
                 patch.dict(os.environ, {"OPENAI_API_KEY" if provider == "openai" else "ANTHROPIC_API_KEY": "test_not_real"}), \
                 contextlib.redirect_stdout(io.StringIO()):
                result = main(cmd)
        self.assertEqual(0, result)
        self.assertEqual(8, monitor.call_count)
        rows = [json.loads(x) for x in destination.read_text().splitlines()]
        self.assertEqual(8, len(rows))
        self.assertEqual({"isolated", "cumulative"}, {r["context_mode"] for r in rows})
        self.assertEqual({False, True}, {r["is_benign"] for r in rows})
        self.assertTrue(all(r["provider"] == provider for r in rows))
        return destination

    def test_openai_execution_path_without_network(self):
        self._mock_run("openai")

    def test_anthropic_execution_path_without_network(self):
        self._mock_run("anthropic")


if __name__ == "__main__":
    unittest.main()
