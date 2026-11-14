"""Offline score-only tests with invented payload and source hashes."""
import hashlib
import unittest

from analyze_ablation import (
    CONFIG_FIELDS,
    EVIDENCE_FIELDS,
    SOURCE_FIELDS,
    analyze_ablation,
)
from authorization_evidence import POLICY_VERSION, canonical, digest
from representation_ablation import REPRESENTATION_POLICY, TOKEN_PROXY_POLICY
from run_ablation import FORMAT_VERSION, MODES, OFFICIAL_ENDPOINTS, ORDER_SEED
from run_case import BENCHMARK_COMMIT, PROMPTS
from run_context import FORMAT_VERSION as CONTEXT_VERSION, TWO_STAGE_SCENARIOS
from run_evidence import FORMAT_VERSION as EVIDENCE_VERSION


def fingerprint(value):
    return hashlib.sha256(value.encode()).hexdigest()


SCENARIO, SECOND_SCENARIO = TWO_STAGE_SCENARIOS
PROMPT = next(iter(PROMPTS))


def current_row(mode, benign=False, trial=0, score=50, **updates):
    settings = {
        "model": "gpt-example", "api": "responses.create", "reasoning": {"effort": "medium"},
        "max_output_tokens": 8192, "timeout": 180.0, "store": False, "sdk_version": "0.0.0",
        "max_retries": 0, "base_url": OFFICIAL_ENDPOINTS["openai"],
    }
    return {
        "format_version": FORMAT_VERSION,
        "benchmark_commit": BENCHMARK_COMMIT,
        "provider": "openai",
        "model": "gpt-example",
        "reasoning_effort": "medium",
        "temperature": None,
        "thinking": True,
        "strip_assistant": True,
        "target_stage": 2,
        "stage_count": 2,
        "planned_scenarios": list(TWO_STAGE_SCENARIOS),
        "planned_trials": 2,
        "order_seed": ORDER_SEED,
        "scenario": SCENARIO,
        "prompt_name": PROMPT,
        "context_mode": mode,
        "is_benign": benign,
        "trial": trial,
        "stage1_sha256": fingerprint(f"source stage 1 {benign}"),
        "stage2_sha256": fingerprint(f"source stage 2 {benign}"),
        "stripped_stage1_sha256": fingerprint(f"stripped stage 1 {benign}"),
        "stripped_stage2_sha256": fingerprint(f"stripped stage 2 {benign}"),
        "prompt_sha256": fingerprint("prompt"),
        "payload_sha256": fingerprint(f"payload {mode} {benign}"),
        "representation_sha256": fingerprint(f"representation {mode} {benign}"),
        "request_settings": settings,
        "request_settings_sha256": digest(canonical(settings)),
        "runner_sources_sha256": fingerprint("runner sources"),
        **{f"{arm}_payload_sha256": fingerprint(f"payload {arm} {benign}")
           for arm in ("isolated", "cumulative", "evidence")},
        "evidence_sha256": fingerprint(f"representation evidence {benign}"),
        "evidence_source_digests_sha256": fingerprint(f"candidate sources {benign}"),
        "evidence_event_count": 1,
        "evidence_source_record_count": 2,
        "evidence_policy": POLICY_VERSION,
        "evidence_selection_mode": "tool_operations_fallback",
        "evidence_tool_block_count": 1,
        "input_length_proxy_bytes": 1800,
        "representation_token_proxy": 600,
        "representation_token_proxy_policy": TOKEN_PROXY_POLICY,
        "representation_policy": REPRESENTATION_POLICY,
        "representation_source_anchors_sha256": fingerprint(f"anchors {benign}"),
        "evidence_token_proxy": 600,
        "representation_token_proxy_delta": 0,
        "representation_bytes": 1200,
        "representation_entry_count": 1,
        "neutral_padding_units": 100 if mode == "neutral_context" else 0,
        "input_tokens": 420,
        "output_tokens": 10,
        "reasoning_tokens": None,
        "numeric_score": score,
        "status": "scored" if score is not None else "refusal_or_invalid",
        **updates,
    }


def historical_row(mode, benign=False, trial=0, score=50, **updates):
    current = current_row(mode, benign, trial, score)
    historical_fields = (
        *CONFIG_FIELDS, *SOURCE_FIELDS, "scenario", "prompt_name", "context_mode",
        "is_benign", "trial", "payload_sha256", "numeric_score", "status", "input_tokens",
    )
    history = {field: current[field] for field in historical_fields}
    if mode == "evidence":
        for field in (*EVIDENCE_FIELDS, "isolated_payload_sha256", "cumulative_payload_sha256"):
            history[field] = current[field]
    return {
        **history,
        "format_version": EVIDENCE_VERSION if mode == "evidence" else CONTEXT_VERSION,
        **updates,
    }


class AblationAnalysisTest(unittest.TestCase):
    def test_matched_contrasts_and_usage_while_missing_trials_remain_visible(self):
        rows = [
            current_row("evidence", score=85),
            current_row("evidence", benign=True, score=20),
            current_row("formatted_full", score=74),
            current_row("formatted_full", benign=True, score=25),
            current_row("neutral_context", score=60, input_tokens=417),
            current_row("neutral_context", benign=True, score=None),
            current_row("reduced_metadata", score=80),
            current_row("reduced_metadata", benign=True, score=19),
        ]
        report = analyze_ablation(rows, threshold=70)
        section = report["by_scenario_and_prompt"][f"{SCENARIO}/{PROMPT}"]
        self.assertEqual(set(section["arms"]), set(MODES))
        self.assertEqual(section["arms"]["evidence"]["mean_paired_separation"], 65)
        self.assertEqual(section["arms"]["evidence"]["attack"]["at_or_above_illustrative_threshold"], 1)
        self.assertEqual(section["arms"]["neutral_context"]["benign"]["invalid_or_refusal_trials"], [0])
        self.assertEqual(section["arms"]["neutral_context"]["benign"]["missing_trials"], [1])
        self.assertEqual(section["arms"]["neutral_context"]["benign"]["reported_usage"]["input_tokens"]["total_tokens"], 420)
        self.assertIsNone(section["arms"]["neutral_context"]["benign"]["reported_usage"]["reasoning_tokens"]["mean_tokens"])
        self.assertEqual(section["arms"]["neutral_context"]["attack"]["representation_by_trial"][0]["neutral_padding_units"], 100)
        full = section["contemporaneous_contrasts"]["formatted_full_minus_evidence"]
        self.assertEqual(full["attack_delta_by_trial"], {0: -11})
        self.assertEqual(full["benign_delta_by_trial"], {0: 5})
        self.assertEqual(full["separation_delta_by_trial"], {0: -16})
        neutral = section["contemporaneous_contrasts"]["neutral_context_minus_evidence"]
        self.assertEqual(neutral["attack_delta_by_trial"], {0: -25})
        self.assertEqual(neutral["provider_input_token_delta_by_label_and_trial"]["attack"], {0: -3})
        self.assertEqual(neutral["complete_matched_trials"], 0)
        absent_group = report["by_scenario_and_prompt"][f"{SECOND_SCENARIO}/{PROMPT}"]
        self.assertEqual(absent_group["arms"]["evidence"]["attack"]["missing_trials"], [0, 1])
        self.assertEqual(report["scope"]["recorded_attempts"], 8)
        self.assertEqual(report["scope"]["planned_attempts"], 64)

    def test_historical_pairs_use_verified_inputs_and_report_evidence_drift(self):
        rows = [current_row("evidence", score=87), current_row("evidence", benign=True, score=15)]
        context = [historical_row(mode, benign) for mode in ("isolated", "cumulative") for benign in (False, True)]
        evidence = [historical_row("evidence", score=80), historical_row("evidence", benign=True, score=20)]
        report = analyze_ablation(rows, context, evidence)
        section = report["by_scenario_and_prompt"][f"{SCENARIO}/{PROMPT}"]
        drift = section["historical_contrasts"]["evidence_minus_prior_evidence"]
        self.assertEqual(drift["separation_delta_by_trial"], {0: 12})
        self.assertEqual(section["historical_arms"]["cumulative"]["attack"]["missing_trials"], [1])

    def test_historical_provenance_is_checked_across_nonoverlapping_trials(self):
        rows = [current_row("reduced_metadata", trial=1)]
        context = [historical_row("isolated"), historical_row("cumulative")]
        evidence = [historical_row("evidence", payload_sha256=fingerprint("changed"))]
        with self.assertRaisesRegex(ValueError, "Historical evidence payload hash differs"):
            analyze_ablation(rows, context, evidence)

    def test_unverified_historical_inputs_are_omitted(self):
        rows = [current_row("reduced_metadata")]
        context = [historical_row("isolated"), historical_row("isolated", benign=True)]
        evidence = [historical_row("evidence"), historical_row("evidence", benign=True)]
        report = analyze_ablation(rows, context, evidence)
        self.assertEqual(report["scope"]["historical_records_supplied"], 4)
        self.assertEqual(report["scope"]["historical_records_matching_current_inputs"], 2)

    def test_mismatched_historical_model_rejected(self):
        rows = [current_row("evidence")]
        with self.assertRaisesRegex(ValueError, "Historical provider, model"):
            analyze_ablation(
                rows,
                [historical_row("isolated", model="gpt-other")],
                [historical_row("evidence", model="gpt-other")],
            )

    def test_mutated_reference_and_duplicate_are_rejected(self):
        first = current_row("evidence")
        with self.assertRaisesRegex(ValueError, "Evidence reference differs"):
            analyze_ablation([{**first, "payload_sha256": fingerprint("changed")}])
        with self.assertRaisesRegex(ValueError, "Duplicate condition"):
            analyze_ablation([first, first])

    def test_changed_model_input_between_trials_is_rejected(self):
        rows = [
            current_row("neutral_context"),
            current_row("neutral_context", trial=1, representation_sha256=fingerprint("changed")),
        ]
        with self.assertRaisesRegex(ValueError, "Inconsistent matched provenance"):
            analyze_ablation(rows)

    def test_changed_request_settings_and_seed_are_rejected(self):
        first = current_row("evidence")
        settings = {**first["request_settings"], "store": True}
        with self.assertRaisesRegex(ValueError, "Invalid request settings hash"):
            analyze_ablation([{**first, "request_settings": settings}])
        with self.assertRaisesRegex(ValueError, "Unexpected OpenAI request settings"):
            analyze_ablation([{**first, "request_settings": settings, "request_settings_sha256": digest(canonical(settings))}])
        with self.assertRaisesRegex(ValueError, "Unexpected order seed"):
            analyze_ablation([{**first, "order_seed": ORDER_SEED + 1}])

    def test_reduced_metadata_must_share_evidence_source_anchors(self):
        rows = [
            current_row("evidence"),
            current_row("reduced_metadata", representation_source_anchors_sha256=fingerprint("changed")),
        ]
        with self.assertRaisesRegex(ValueError, "Inconsistent matched provenance: representation_source_anchors_sha256"):
            analyze_ablation(rows)

    def test_historical_inputs_are_required_together(self):
        with self.assertRaisesRegex(ValueError, "Supply both"):
            analyze_ablation([current_row("evidence")], context_rows=[])


if __name__ == "__main__":
    unittest.main()
