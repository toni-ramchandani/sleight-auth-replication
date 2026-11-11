"""Offline tests: no real benchmark data, credentials, or API calls."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch
import sys

from run_case import (
    AUTHORIZATION_CASES, BENCHMARK_COMMIT, CASE, CASE_CATALOG, PROMPTS,
    build_plan, call_monitor, condition_key, default_output, load_completed,
    main, select_cases, sha256,
)
from summarize import analyze, case_summary, stage_summary, summarize


class Fixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.prompts = self.root / "prompts"
        self.prompts.mkdir()
        for prompt in PROMPTS:
            (self.prompts / f"{prompt}.md").write_text(f"monitor:{prompt}")
        for case in AUTHORIZATION_CASES:
            folder = self.root / "attacks" / "authorization_confusion" / case.name
            folder.mkdir(parents=True)
            (folder / "transcript.jsonl").write_text(f"attack:{case.name}")
            (folder / "benign.jsonl").write_text(f"benign:{case.name}")

    def record(self, item, provider="openai", model="gpt-6.1-sol", score=90, effort="medium"):
        return {
            "provider": provider, "model": model, "reasoning_effort": effort,
            "temperature": None if provider == "openai" else 1.0,
            "thinking": True, "strip_assistant": True,
            "benchmark_commit": BENCHMARK_COMMIT,
            "case": item["case"], "scenario": item["scenario"],
            "stage": item["stage"], "stage_count": item["stage_count"],
            "prompt_name": item["prompt_name"], "is_benign": item["is_benign"],
            "trial": item["trial"],
            "transcript_sha256": sha256(item["transcript_path"]),
            "prompt_sha256": sha256(self.prompts / f"{item['prompt_name']}.md"),
            "numeric_score": score,
            "status": "scored" if score is not None else "refusal_or_invalid",
        }


class PlanTests(Fixture):
    def test_legacy_default_is_4_conditions_per_trial(self):
        plan = build_plan(self.root, 1)
        self.assertEqual(4, len(plan))
        self.assertEqual({CASE}, {p["case"] for p in plan})
        self.assertEqual({False, True}, {p["is_benign"] for p in plan})

    def test_full_category_is_24_requests_and_four_scenarios(self):
        cases = select_cases(all_authorization=True)
        plan = build_plan(self.root, 1, cases)
        self.assertEqual(6, len(cases))
        self.assertEqual(24, len(plan))
        self.assertEqual(4, len({p["scenario"] for p in plan}))
        self.assertEqual(24, len({condition_key(p) for p in plan}))

    def test_ten_trials_is_240_requests(self):
        self.assertEqual(240, len(build_plan(self.root, 10, select_cases(all_authorization=True))))

    def test_two_stage_scenarios_order_and_identity(self):
        selected = select_cases(all_authorization=True)
        specs = [CASE_CATALOG[name] for name in selected]
        corp = [x for x in specs if x.scenario == "transcript-perm-launder-corp-2part"]
        gcs = [x for x in specs if x.scenario == "transcript-perm-launder-gcs-2part"]
        self.assertEqual([1, 2], [x.stage for x in corp])
        self.assertEqual([1, 2], [x.stage for x in gcs])

    def test_selected_cases_repeatable_cli_semantics(self):
        names = ["preprocess-sabotage", "transcript-perm-launder-corp-2part-stage2"]
        self.assertEqual(tuple(names), select_cases(names))
        self.assertEqual(8, len(build_plan(self.root, 1, names)))

    def test_duplicate_and_unknown_cases_fail(self):
        with self.assertRaises(ValueError):
            select_cases([CASE, CASE])
        with self.assertRaises(ValueError):
            select_cases(["not-a-real-case"])
        with self.assertRaises(ValueError):
            select_cases([CASE], all_authorization=True)

    def test_missing_input_fails_entire_plan(self):
        path = self.root / "attacks" / "authorization_confusion" / "preprocess-sabotage" / "benign.jsonl"
        path.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "preprocess-sabotage"):
            build_plan(self.root, 1, select_cases(all_authorization=True))

    def test_trial_range(self):
        for trials in (0, 11):
            with self.assertRaises(ValueError):
                build_plan(self.root, trials)

    def test_output_preserves_pilot_and_isolates_multi_case(self):
        single = default_output("openai", "gpt-6.1-sol", 1)
        self.assertEqual(Path("results/openai-gpt-6.1-sol-t1.jsonl"), single)
        multi = default_output("openai", "gpt-6.1-sol", 1, select_cases(all_authorization=True))
        self.assertEqual(Path("results/openai-gpt-6.1-sol-auth6-t1.jsonl"), multi)
        self.assertNotEqual(multi, default_output("anthropic", "claude-opus-4-6", 1, select_cases(all_authorization=True)))
        different = default_output("openai", "gpt-6.1-sol", 1, ["preprocess-sabotage"])
        self.assertNotEqual(different, single)

    def test_list_cases_does_not_require_decrypted_data(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main(["--list-cases", "--benchmark-dir", "/does/not/exist"])
        self.assertEqual(0, code)
        self.assertIn("stage=2/2", out.getvalue())

    def test_all_authorization_dry_run_has_no_provider_calls(self):
        out = io.StringIO()
        with patch("run_case.check_revision", return_value=BENCHMARK_COMMIT):
            with contextlib.redirect_stdout(out):
                code = main(["--benchmark-dir", str(self.root), "--all-authorization", "--trials", "1"])
        self.assertEqual(0, code)
        self.assertIn("Planned requests: 24", out.getvalue())
        self.assertIn("DRY RUN COMPLETE", out.getvalue())


class ResumeTests(Fixture):
    def setUp(self):
        super().setUp()
        self.path = self.root / "results.jsonl"
        self.plan = build_plan(self.root, 1, select_cases(all_authorization=True))
        self.options = {
            "provider": "openai", "model": "gpt-6.1-sol", "revision": BENCHMARK_COMMIT,
            "plan": self.plan, "root": self.root, "resume": True, "reasoning_effort": "medium",
        }

    def test_resume_skips_only_matching_conditions(self):
        records = [self.record(self.plan[0]), self.record(self.plan[1])]
        self.path.write_text("\n".join(json.dumps(row) for row in records) + "\n")
        self.assertEqual(2, len(load_completed(self.path, **self.options)))

    def test_prior_single_case_record_can_be_used_with_expanded_selection(self):
        legacy = dict(self.record(build_plan(self.root, 1)[0]))
        legacy.pop("scenario")
        legacy.pop("stage")
        legacy.pop("stage_count")
        self.path.write_text(json.dumps(legacy) + "\n")
        self.assertEqual(1, len(load_completed(self.path, **self.options)))

    def test_resume_requires_flag_if_file_exists(self):
        self.path.write_text(json.dumps(self.record(self.plan[0])) + "\n")
        with self.assertRaisesRegex(RuntimeError, "--resume"):
            load_completed(self.path, **{**self.options, "resume": False})

    def test_resume_rejects_modified_transcript(self):
        self.path.write_text(json.dumps(self.record(self.plan[0])) + "\n")
        self.plan[0]["transcript_path"].write_text("modified")
        with self.assertRaisesRegex(RuntimeError, "hash"):
            load_completed(self.path, **self.options)

    def test_resume_rejects_duplicate_rows(self):
        record = json.dumps(self.record(self.plan[0]))
        self.path.write_text(record + "\n" + record + "\n")
        with self.assertRaisesRegex(RuntimeError, "duplicated"):
            load_completed(self.path, **self.options)

    def test_resume_rejects_wrong_model_or_effort(self):
        row = self.record(self.plan[0])
        row["reasoning_effort"] = "high"
        self.path.write_text(json.dumps(row) + "\n")
        with self.assertRaisesRegex(RuntimeError, "configuration"):
            load_completed(self.path, **self.options)

    def test_refusal_record_is_immutable_but_not_scored(self):
        row = self.record(self.plan[0], score=None)
        self.path.write_text(json.dumps(row) + "\n")
        self.assertEqual(1, len(load_completed(self.path, **self.options)))

    def test_resume_rejects_corrupted_or_out_of_range_score(self):
        row = self.record(self.plan[0], score=101)
        self.path.write_text(json.dumps(row) + "\n")
        with self.assertRaisesRegex(RuntimeError, "invalid numeric score"):
            load_completed(self.path, **self.options)

    def test_resume_rejects_condition_from_unselected_case(self):
        row = self.record(self.plan[-1])
        self.path.write_text(json.dumps(row) + "\n")
        narrowed = build_plan(self.root, 1)
        with self.assertRaisesRegex(RuntimeError, "absent from selection"):
            load_completed(self.path, **{**self.options, "plan": narrowed})


class ProviderTests(unittest.TestCase):
    def test_openai_omits_temperature_and_disables_storage(self):
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
        self.assertEqual({"effort": "medium"}, params["reasoning"])
        self.assertFalse(params["store"])
        self.assertNotIn("temperature", params)
        self.assertEqual("system instructions", params["instructions"])


    def test_anthropic_uses_official_provider_configuration(self):
        providers = ModuleType("evaluate.core.providers")
        providers.build_api_params = Mock(return_value={"system": "instructions"})
        response = SimpleNamespace(
            usage=SimpleNamespace(input_tokens=100, output_tokens=50), stop_reason="end_turn",
        )
        providers.call_anthropic = Mock(return_value=response)
        parsing = ModuleType("evaluate.core.response")
        parsing.extract_response_parts = Mock(return_value=("", "<score>75</score>", False, "end_turn"))
        types_module = ModuleType("evaluate.core.types")
        types_module.MonitorSpec = lambda **kwargs: SimpleNamespace(**kwargs)
        modules = {
            "evaluate": ModuleType("evaluate"),
            "evaluate.core": ModuleType("evaluate.core"),
            "evaluate.core.providers": providers,
            "evaluate.core.response": parsing,
            "evaluate.core.types": types_module,
        }
        with patch.dict(sys.modules, modules):
            result = call_monitor("anthropic", "claude-opus-4-6", "medium", "instructions", "transcript", object())
        self.assertEqual("<score>75</score>", result[0])
        self.assertEqual("end_turn", result[2])
        self.assertEqual(100, result[3]["input_tokens"])
        self.assertEqual("claude-opus-4-6", providers.build_api_params.call_args.args[0].model)
        self.assertTrue(providers.build_api_params.call_args.args[0].thinking)
        self.assertEqual(1.0, providers.build_api_params.call_args.args[0].temperature)

    def test_unknown_provider_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "Unsupported provider"):
            call_monitor("random", "unknown", "medium", None, "text", object())

    def test_openai_incomplete_response_not_scored(self):
        response = SimpleNamespace(status="incomplete", output_text="<score>90</score>", usage=None)
        client = SimpleNamespace(responses=SimpleNamespace(create=Mock(return_value=response)))
        answer, _, stop_reason, _ = call_monitor("openai", "gpt-6.1-sol", "medium", None, "text", client)
        self.assertEqual("", answer)
        self.assertEqual("response_incomplete", stop_reason)


class AnalysisTests(Fixture):
    def sample(self, scenario="transcript-perm-launder-corp-2part", prompt="control_arena"):
        names = [c.name for c in AUTHORIZATION_CASES if c.scenario == scenario]
        rows = []
        for item in build_plan(self.root, 1, names):
            if item["prompt_name"] != prompt:
                continue
            if item["stage"] == 1:
                score = 10 if item["is_benign"] else 40
            else:
                score = 12 if item["is_benign"] else 90
            rows.append(self.record(item, score=score))
        return rows

    def test_summary_backwards_compatible_monitor_key(self):
        result = summarize(self.sample())
        group = result["openai/gpt-6.1-sol/control_arena"]
        self.assertEqual([40, 90], group["attack_scores"])
        self.assertEqual([10, 12], group["benign_scores"])

    def test_stage_analysis_retains_stage_visibility(self):
        key = "openai/gpt-6.1-sol/control_arena/transcript-perm-launder-corp-2part"
        group = stage_summary(self.sample())[key]
        self.assertEqual(2, group["stage_count"])
        self.assertEqual([], group["missing_stages"])
        self.assertEqual(1, group["complete_trajectory_trials"])
        self.assertEqual({"2": 1}, group["attack_first_alert_stage_counts"])
        self.assertEqual({"never": 1}, group["benign_first_alert_stage_counts"])
        self.assertEqual(50, group["paired_attack_stage_delta"]["mean_stage2_minus_stage1"])
        self.assertEqual(2, group["paired_benign_stage_delta"]["mean_stage2_minus_stage1"])

    def test_missing_stage_not_misrepresented_as_complete_trajectory(self):
        rows = [r for r in self.sample() if r["stage"] == 1]
        key = "openai/gpt-6.1-sol/control_arena/transcript-perm-launder-corp-2part"
        group = stage_summary(rows)[key]
        self.assertEqual([2], group["missing_stages"])
        self.assertEqual(0, group["complete_trajectory_trials"])

    def test_case_summary_illustrative_threshold(self):
        groups = case_summary(self.sample(), threshold=70)
        suffix = "transcript-perm-launder-corp-2part-stage2"
        stage2 = next(v for k, v in groups.items() if k.endswith(suffix))
        self.assertEqual(1, stage2["attack"]["scores_at_or_above_threshold"])
        self.assertEqual(0, stage2["benign"]["scores_at_or_above_threshold"])

    def test_full_analysis_advertises_no_fpr_claim(self):
        analysis = analyze(self.sample())
        self.assertEqual(2, analysis["scope"]["case_directories"])
        self.assertEqual(1, analysis["scope"]["scenarios"])
        self.assertIn("NOT calibrated", analysis["cautions"][0])

    def test_duplicate_rows_rejected(self):
        rows = self.sample()
        with self.assertRaisesRegex(ValueError, "Duplicate scored condition"):
            analyze(rows + [rows[0]])

    def test_reasoning_effort_mixture_rejected(self):
        rows = self.sample()
        rows[0] = {**rows[0], "reasoning_effort": "high"}
        with self.assertRaisesRegex(ValueError, "Mixed reasoning effort"):
            analyze(rows)

    def test_unrecognized_case_rejected(self):
        rows = self.sample()
        rows[0] = {**rows[0], "case": "made-up-case"}
        with self.assertRaisesRegex(ValueError, "Unrecognized case"):
            analyze(rows)


    def test_all_invalid_stage_is_visible_but_not_a_completed_trajectory(self):
        rows = self.sample()
        for row in rows:
            if row["stage"] == 2:
                row["numeric_score"] = None
                row["status"] = "refusal_or_invalid"
        stage = next(iter(stage_summary(rows).values()))
        self.assertEqual([], stage["missing_stages"])
        self.assertEqual(2, stage["by_stage"]["2"]["invalid_or_refusal"])
        self.assertEqual(0, stage["complete_trajectory_trials"])

    def test_invalid_scores_do_not_become_alerts(self):
        rows = self.sample()
        rows[-2] = {**rows[-2], "numeric_score": None, "status": "refusal_or_invalid"}
        report = analyze(rows)
        self.assertTrue(any(x["invalid_or_refusal"] == 1 for x in report["by_case"].values()))
        group = next(iter(report["stage_aware"].values()))
        self.assertEqual(0, group["complete_trajectory_trials"])


if __name__ == "__main__":
    unittest.main()
