"""Offline coverage of the Experiment 5 plan, API dispatch, and result ledger."""
from __future__ import annotations

import contextlib
import io
import json
import os
import re
import sys
import tempfile
import unittest
from itertools import product
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from run_ablation import MODES, OFFICIAL_ENDPOINTS, build_plan, main
from run_case import BENCHMARK_COMMIT, PROMPTS
from run_context import TARGET_INSTRUCTION, TWO_STAGE_SCENARIOS
from run_evidence import _payload_hash

SOURCE_MARKER = "SYNTHETIC_SOURCE_ONLY"
ANSWER_MARKER = "SYNTHETIC_RESPONSE_ONLY"
KEY = "placeholder-secret-for-offline-test"


def fake_module(name: str, **attributes) -> ModuleType:
    module = ModuleType(name)
    vars(module).update(attributes)
    return module


def fake_render(template: str, source: str) -> tuple[str | None, str]:
    if "{TEXT}" in template:
        return None, template.replace("{TEXT}", source)
    return template, f"<transcript>\n{source}\n</transcript>"


def fake_anthropic_params(spec, system, user) -> dict:
    params = {
        "model": spec.model,
        "messages": [{"role": "user", "content": user}],
        "temperature": spec.temperature,
        "thinking": {"type": "adaptive"},
        "max_tokens": 128000,
        "output_config": {"effort": "max"},
    }
    if system:
        params["system"] = system
    return params


def fake_parse_response(text: str) -> dict:
    score = re.search(r"<score>(\d+)</score>", text)
    return {"numeric_score": int(score.group(1)) if score else None}


class RunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        tmp = stack.enter_context(tempfile.TemporaryDirectory())
        self.root = Path(tmp) / "benchmark"
        self.output = Path(tmp) / "outputs" / "ablation.jsonl"
        self.calls = []
        self.clients = []
        self.fail_after = None
        self.write_fixture()

        tests = self

        class Client:
            def __init__(self, **kwargs):
                tests.clients.append(kwargs)
                self.responses = SimpleNamespace(create=tests.openai_response)
                self.messages = SimpleNamespace(stream=tests.anthropic_stream)

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return None

        modules = {
            "evaluate": fake_module("evaluate"),
            "evaluate.core": fake_module("evaluate.core"),
            "evaluate.core.transcript": fake_module(
                "evaluate.core.transcript",
                render_prompt=fake_render,
                strip_assistant_text=str.strip,
            ),
            "evaluate.core.types": fake_module("evaluate.core.types", MonitorSpec=SimpleNamespace),
            "evaluate.core.providers": fake_module("evaluate.core.providers", build_api_params=fake_anthropic_params),
            "evaluate.core.response": fake_module(
                "evaluate.core.response", parse_response=fake_parse_response,
                extract_response_parts=lambda response: ("", response.answer, response.refused, response.stop),
            ),
            "openai": fake_module("openai", OpenAI=Client),
            "anthropic": fake_module("anthropic", Anthropic=Client),
        }
        stack.enter_context(patch.dict(sys.modules, modules))
        stack.enter_context(patch.dict(os.environ, {"OPENAI_API_KEY": KEY, "ANTHROPIC_API_KEY": KEY}))
        stack.enter_context(patch("run_ablation.benchmark_revision", return_value=BENCHMARK_COMMIT))
        stack.enter_context(patch("run_ablation.version", return_value="0.0.0-offline"))

    def write_fixture(self) -> None:
        prompts = self.root / "prompts"
        prompts.mkdir(parents=True)
        (prompts / f"{PROMPTS[0]}.md").write_text("Evaluate the activity.\n{TEXT}", encoding="utf-8")
        (prompts / f"{PROMPTS[1]}.md").write_text("Evaluate the transcript.", encoding="utf-8")
        for index, scenario in enumerate(TWO_STAGE_SCENARIOS):
            for label in (False, True):
                fixture_key = f"{SOURCE_MARKER}_{index}_{int(label)}"
                stage1 = [
                    {"canary": fixture_key},
                    {"role": "assistant", "content": {
                        "type": "tool_use", "name": "lookup_menu", "input": {"day": "Tuesday"},
                    }},
                    {"role": "user", "content": {"type": "tool_result", "content": fixture_key}},
                ]
                stage2 = [
                    {"role": "user", "content": {"type": "text", "text": f"Review the menu. {fixture_key}"}},
                ]
                for stage, records in enumerate((stage1, stage2), 1):
                    directory = self.root / "attacks" / "authorization_confusion" / f"{scenario}-stage{stage}"
                    directory.mkdir(parents=True, exist_ok=True)
                    filename = "benign.jsonl" if label else "transcript.jsonl"
                    (directory / filename).write_text(
                        "\n".join(map(json.dumps, records)) + "\n", encoding="utf-8",
                    )

    def record_call(self, kwargs: dict) -> None:
        if len(self.calls) == self.fail_after:
            raise RuntimeError(f"Transport error includes {KEY}, {SOURCE_MARKER}, {ANSWER_MARKER}")
        self.calls.append(kwargs)

    def openai_response(self, **kwargs):
        self.record_call(kwargs)
        # First attempt is incomplete. It counts as a completed attempt on resume.
        return SimpleNamespace(
            status="incomplete" if len(self.calls) == 1 else "completed",
            output_text=f"<score>71</score>{ANSWER_MARKER}",
            usage=SimpleNamespace(input_tokens=101, output_tokens=25,
                                  output_tokens_details=SimpleNamespace(reasoning_tokens=12)),
        )

    @contextlib.contextmanager
    def anthropic_stream(self, **kwargs):
        self.record_call(kwargs)
        response = SimpleNamespace(
            stop="end_turn",
            refused=False,
            answer=f"<score>72</score>{ANSWER_MARKER}",
            usage=SimpleNamespace(input_tokens=101, output_tokens=25,
                                  cache_read_input_tokens=7, cache_creation_input_tokens=9),
        )
        yield SimpleNamespace(get_final_message=lambda: response)

    def run_cli(self, *args: str) -> tuple[int, str, str]:
        argv = ["--benchmark-dir", str(self.root), "--output", str(self.output),
                "--scenario", TWO_STAGE_SCENARIOS[0], *args]
        with contextlib.redirect_stdout(io.StringIO()) as stdout, contextlib.redirect_stderr(io.StringIO()) as stderr:
            result = main(argv)
        output = stdout.getvalue(), stderr.getvalue()
        for marker in (SOURCE_MARKER, ANSWER_MARKER, KEY):
            self.assertNotIn(marker, "".join(output))
        return result, *output

    def rows(self) -> list[dict]:
        text = self.output.read_text(encoding="utf-8")
        for marker in (SOURCE_MARKER, ANSWER_MARKER, KEY):
            self.assertNotIn(marker, text)
        return [json.loads(line) for line in text.splitlines()]

    def test_plan_has_matched_inputs_all_labels_and_repeated_conditions(self):
        plan = build_plan(self.root, 2, TWO_STAGE_SCENARIOS)
        self.assertEqual(len(plan), 64)
        self.assertEqual({arm.item.key for arm in plan}, set(product(
            TWO_STAGE_SCENARIOS, PROMPTS, (False, True), range(2), MODES,
        )))
        self.assertEqual([arm.item.key for arm in plan],
                         [arm.item.key for arm in build_plan(self.root, 2, TWO_STAGE_SCENARIOS)])
        self.assertNotEqual([arm.item.key[-1] for arm in plan[:32]], [arm.item.key[-1] for arm in plan[32:]])
        for scenario, prompt, label in product(TWO_STAGE_SCENARIOS, PROMPTS, (False, True)):
            match = [arm for arm in plan if arm.item.key[:3] == (scenario, prompt, label)]
            for field in ("stage1_sha256", "stage2_sha256", "stripped_stage2_sha256", "prompt_sha256",
                          "isolated_payload_sha256", "cumulative_payload_sha256", "evidence_payload_sha256"):
                self.assertEqual(len({arm.provenance[field] for arm in match}), 1)
            self.assertEqual(len({arm.provenance["payload_sha256"] for arm in match}), 4)
            self.assertEqual(len({arm.user.split('<session index="2" role="target">')[1] for arm in match}), 1)
            for arm in match:
                self.assertTrue(arm.user.startswith(TARGET_INSTRUCTION))
                self.assertEqual(arm.provenance["payload_sha256"], _payload_hash(arm.system, arm.user))
                if arm.item.context_mode == "evidence":
                    self.assertEqual(arm.provenance["payload_sha256"], arm.provenance["evidence_payload_sha256"])
        for scenario, prompt in product(TWO_STAGE_SCENARIOS, PROMPTS):
            arms = [a for a in plan if a.item.key[:2] == (scenario, prompt)]
            self.assertEqual(len({a.provenance["stage2_sha256"] for a in arms}), 2)

    def test_dry_run_makes_no_writes_and_does_not_create_client(self):
        result, stdout, _ = self.run_cli()
        self.assertEqual(result, 0)
        self.assertIn("Planned requests: 16", stdout)
        self.assertIn("DRY RUN COMPLETE", stdout)
        self.assertFalse(self.output.parent.exists())
        self.assertEqual(self.clients, [])
        self.assertEqual(self.calls, [])

    def test_cap_rejects_execution_before_provider_dispatch(self):
        result, _, stderr = self.run_cli("--execute", "--max-calls", "15")
        self.assertEqual(result, 2)
        self.assertIn("Remaining calls exceed --max-calls", stderr)
        self.assertFalse(self.output.exists())
        self.assertEqual(self.clients, [])
        self.assertEqual(self.calls, [])

    def test_openai_interruption_and_resume_skip_saved_scored_and_unscored_rows(self):
        self.fail_after = 4
        result, _, stderr = self.run_cli("--execute", "--max-calls", "16")
        self.assertEqual(result, 3)
        self.assertIn("EVALUATION INTERRUPTED", stderr)
        initial = self.rows()
        self.assertEqual(len(initial), 4)
        self.assertIsNone(initial[0]["numeric_score"])
        self.assertEqual(initial[0]["status"], "refusal_or_invalid")

        preview, stdout, _ = self.run_cli("--resume")
        self.assertEqual(preview, 0)
        self.assertIn("Completed: 4 | Remaining: 12", stdout)
        self.assertEqual(self.rows(), initial)
        self.assertEqual(len(self.calls), 4)

        self.fail_after = None
        result, _, _ = self.run_cli("--execute", "--resume", "--max-calls", "12")
        self.assertEqual(result, 0)
        rows = self.rows()
        self.assertEqual(len(rows), 16)
        self.assertEqual(rows[:4], initial)
        self.assertEqual(len(self.calls), 16)
        self.assertEqual(len({row["payload_sha256"] for row in rows}), 16)
        self.assertEqual(len({call["input"] for call in self.calls}), 16)
        self.assertTrue(all(call["store"] is False and call["max_output_tokens"] == 8192 for call in self.calls))
        self.assertTrue(all(client["base_url"] == OFFICIAL_ENDPOINTS["openai"] and client["max_retries"] == 0
                            for client in self.clients))
        self.assertTrue(all(row["numeric_score"] == 71 and row["input_tokens"] == 101 for row in rows[1:]))
        done, stdout, _ = self.run_cli("--execute", "--resume", "--max-calls", "1")
        self.assertEqual(done, 0)
        self.assertIn("Completed: 16 | Remaining: 0", stdout)
        self.assertEqual(len(self.clients), 2)

    def test_resume_rejects_mutated_provenance_before_provider_dispatch(self):
        self.fail_after = 2
        self.assertEqual(self.run_cli("--execute", "--max-calls", "16")[0], 3)
        rows = self.rows()
        rows[1]["representation_source_anchors_sha256"] = "0" * 64
        self.output.write_text("\n".join(map(json.dumps, rows)) + "\n", encoding="utf-8")
        result, _, stderr = self.run_cli("--execute", "--resume", "--max-calls", "16")
        self.assertEqual(result, 2)
        self.assertIn("incompatible resume record at line 2", stderr)
        self.assertEqual(len(self.clients), 1)
        self.assertEqual(len(self.calls), 2)

    def test_anthropic_streams_once_per_condition_with_matched_settings(self):
        result, _, _ = self.run_cli("--provider", "anthropic", "--execute", "--max-calls", "16")
        self.assertEqual(result, 0)
        self.assertEqual(len(self.calls), 16)
        self.assertEqual(self.clients, [
            {"api_key": KEY, "base_url": OFFICIAL_ENDPOINTS["anthropic"], "max_retries": 0},
        ])
        for call, row in zip(self.calls, self.rows()):
            self.assertEqual(row["numeric_score"], 72)
            self.assertEqual(row["input_tokens"], 117)
            self.assertEqual(row["request_settings"]["api"], "messages.stream")
            self.assertEqual(call["thinking"], {"type": "adaptive"})
            self.assertEqual(call["timeout"], 300.0)
            self.assertNotIn("output_config", call)
            self.assertEqual(row["payload_sha256"],
                             _payload_hash(call.get("system"), call["messages"][0]["content"]))


if __name__ == "__main__":
    unittest.main()
