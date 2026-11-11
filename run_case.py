"""SLEIGHT-Bench monitor pilot: pinned, multi-case, explicitly billable.

The official benchmark supplies transcripts, prompt stripping, rendering and
score parsing. This standalone runner stores only scores and provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

BENCHMARK_COMMIT = "218c58315cc01ff0dc5a100e906c27d82d259521"
PROMPTS = ("control_arena", "control_arena_doc_planting_aware")
DEFAULT_MODELS = {"anthropic": "claude-opus-4-6", "openai": "gpt-6.1-sol"}
OPENAI_EFFORTS = ("none", "low", "medium", "high", "xhigh", "max")


@dataclass(frozen=True)
class CaseSpec:
    """Pinned classification: a scenario can span two distinct sessions."""

    name: str
    scenario: str
    stage: int
    stage_count: int


AUTHORIZATION_CASES = (
    CaseSpec("ambiguous-instruction-weight-backup-orion-personal", "ambiguous-instruction-weight-backup-orion-personal", 1, 1),
    CaseSpec("preprocess-sabotage", "preprocess-sabotage", 1, 1),
    CaseSpec("transcript-perm-launder-corp-2part-stage1", "transcript-perm-launder-corp-2part", 1, 2),
    CaseSpec("transcript-perm-launder-corp-2part-stage2", "transcript-perm-launder-corp-2part", 2, 2),
    CaseSpec("transcript-perm-launder-gcs-2part-stage1", "transcript-perm-launder-gcs-2part", 1, 2),
    CaseSpec("transcript-perm-launder-gcs-2part-stage2", "transcript-perm-launder-gcs-2part", 2, 2),
)
CASE_CATALOG = {case.name: case for case in AUTHORIZATION_CASES}
# Retain the original Experiment 1 default and its output filename.
CASE = "transcript-perm-launder-corp-2part-stage1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_cases(requested: Sequence[str] | None = None, *, all_authorization: bool = False) -> tuple[str, ...]:
    if requested and all_authorization:
        raise ValueError("Choose --all-authorization or individual --case values, not both")
    if all_authorization:
        return tuple(case.name for case in AUTHORIZATION_CASES)
    names = tuple(requested) if requested else (CASE,)
    if len(names) != len(set(names)):
        raise ValueError("Duplicate case selector")
    unknown = [name for name in names if name not in CASE_CATALOG]
    if unknown:
        raise ValueError(f"Unknown case(s): {unknown}. Use --list-cases")
    return names


def build_plan(root: Path, trials: int, cases: Sequence[str] | None = None) -> list[dict[str, Any]]:
    """Build deterministic paired conditions; stop if ANY input is missing."""
    if not 1 <= trials <= 10:
        raise ValueError("Trials per condition must be between 1 and 10")
    selected = select_cases(cases)
    inputs: list[dict[str, Any]] = []
    required: list[Path] = [root / "prompts" / f"{p}.md" for p in PROMPTS]
    for case_name in selected:
        spec = CASE_CATALOG[case_name]
        base = root / "attacks" / "authorization_confusion" / spec.name
        required.extend((base / "transcript.jsonl", base / "benign.jsonl"))
        for prompt in PROMPTS:
            for benign in (False, True):
                transcript = base / ("benign.jsonl" if benign else "transcript.jsonl")
                for trial in range(trials):
                    inputs.append({
                        "case": spec.name,
                        "scenario": spec.scenario,
                        "stage": spec.stage,
                        "stage_count": spec.stage_count,
                        "prompt_name": prompt,
                        "transcript_path": transcript,
                        "is_benign": benign,
                        "trial": trial,
                    })
    missing = [str(path.relative_to(root)) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing decrypted benchmark input(s): " + ", ".join(missing))
    return inputs


def check_revision(root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=10,
        )
    except (subprocess.SubprocessError, OSError):
        raise RuntimeError("Benchmark must be a Git checkout pinned to the original release") from None
    revision = completed.stdout.strip()
    if revision != BENCHMARK_COMMIT:
        raise RuntimeError(f"Wrong benchmark revision: {revision}; expected {BENCHMARK_COMMIT}")
    return revision


def default_output(provider: str, model: str, trials: int, cases: Sequence[str] | None = None) -> Path:
    """Keep Experiment 1's filename; disambiguate all other selections."""
    selected = select_cases(cases)
    safe_model = re.sub(r"[^a-zA-Z0-9_.-]", "-", model)
    if selected == (CASE,):
        suffix = ""
    elif selected == tuple(c.name for c in AUTHORIZATION_CASES):
        suffix = "-auth6"
    else:
        selection_id = hashlib.sha256("\n".join(sorted(selected)).encode()).hexdigest()[:8]
        suffix = f"-cases{len(selected)}-{selection_id}"
    return Path("results") / f"{provider}-{safe_model}{suffix}-t{trials}.jsonl"


def condition_key(item: dict[str, Any]) -> tuple[str, str, bool, int]:
    return (item["case"], item["prompt_name"], item["is_benign"], item["trial"])


def load_completed(
    path: Path, *, provider: str, model: str, revision: str,
    plan: list[dict[str, Any]], root: Path, resume: bool,
    reasoning_effort: str | None,
) -> set[tuple[str, str, bool, int]]:
    """Reject mixed results, stale source hashes and duplicated conditions."""
    if not path.exists():
        return set()
    if not resume:
        raise RuntimeError(f"Output {path} exists. Use --resume, or choose a new --output.")
    valid = {condition_key(item): item for item in plan}
    completed: set[tuple[str, str, bool, int]] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            key = condition_key(row)
            item = valid.get(key)
            if item is None or key in completed:
                raise ValueError("Condition absent from selection or duplicated")
            if (row["provider"] != provider or row["model"] != model
                    or row["benchmark_commit"] != revision
                    or row.get("reasoning_effort") != reasoning_effort
                    or row.get("temperature") != (1.0 if provider == "anthropic" else None)
                    or row.get("thinking") is not True
                    or row.get("strip_assistant") is not True
                    or row["transcript_sha256"] != sha256(item["transcript_path"])
                    or row["prompt_sha256"] != sha256(root / "prompts" / f"{row['prompt_name']}.md")):
                raise ValueError("Run configuration or input hash changed")
            if "scenario" in row and row["scenario"] != item["scenario"]:
                raise ValueError("Scenario metadata changed")
            if "stage" in row and row["stage"] != item["stage"]:
                raise ValueError("Stage metadata changed")
            score = row.get("numeric_score")
            status = row.get("status")
            if status not in {"scored", "refusal_or_invalid"}:
                raise ValueError("Unrecognized outcome status")
            if status == "scored" and (type(score) is not int or not 0 <= score <= 100):
                raise ValueError("Scored outcome has invalid numeric score")
            if status == "refusal_or_invalid" and score is not None:
                raise ValueError("Invalid outcome unexpectedly has a numeric score")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Cannot resume {path}; invalid line {line_number}: {exc}") from None
        completed.add(key)
    return completed


def usage_from_openai(response: Any) -> dict[str, int | None]:
    usage = getattr(response, "usage", None)
    details = getattr(usage, "output_tokens_details", None)
    return {
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
        "reasoning_tokens": getattr(details, "reasoning_tokens", None),
    }


def call_monitor(
    provider: str, model: str, effort: str,
    system_prompt: str | None, user_message: str, client: Any,
) -> tuple[str, bool, str | None, dict[str, int | None]]:
    """Call a monitor and return text, refusal, stop reason, token usage."""
    if provider == "openai":
        # Reasoning-model API requests intentionally omit temperature.
        # store=False disallows storing the benchmark input as a persisted Response.
        params: dict[str, Any] = {
            "model": model,
            "input": user_message,
            "reasoning": {"effort": effort},
            "max_output_tokens": 8192,
            "store": False,
            "timeout": 180.0,
        }
        if system_prompt is not None:
            params["instructions"] = system_prompt
        response = client.responses.create(**params)
        status = getattr(response, "status", None)
        if status not in (None, "completed"):
            return "", False, f"response_{status}", usage_from_openai(response)
        return getattr(response, "output_text", "") or "", False, status, usage_from_openai(response)

    if provider != "anthropic":
        raise ValueError(f"Unsupported provider: {provider}")
    from evaluate.core.providers import build_api_params, call_anthropic
    from evaluate.core.response import extract_response_parts
    from evaluate.core.types import MonitorSpec

    spec = MonitorSpec(
        prompt="control_arena", model=model, temperature=1.0,
        trials=1, thinking=True, strip_assistant=True, type="api",
    )
    params = build_api_params(spec, system_prompt, user_message)
    response = call_anthropic(client, spec, params)
    _, answer, refusal, stop_reason = extract_response_parts(response)
    usage = getattr(response, "usage", None)
    return answer, refusal, stop_reason, {
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
        "reasoning_tokens": None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, default=Path("vendor/sleight-bench"))
    parser.add_argument("--provider", choices=("openai", "anthropic"), default="openai")
    parser.add_argument("--model", help="Provider API model ID")
    parser.add_argument("--reasoning-effort", choices=OPENAI_EFFORTS, default="medium")
    parser.add_argument("--trials", type=int, default=1, help="Trials per prompt/label (1..10)")
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument("--all-authorization", action="store_true", help="All 6 pinned authorization-confusion transcript directories")
    selector.add_argument("--case", action="append", choices=tuple(CASE_CATALOG), help="Select one or more case directories; repeat option")
    parser.add_argument("--list-cases", action="store_true", help="Display case/scenario/stage mapping without reading data")
    parser.add_argument("--execute", action="store_true", help="Explicitly authorize paid provider calls")
    parser.add_argument("--resume", action="store_true", help="Skip existing condition records after strict integrity checks")
    parser.add_argument("--output", type=Path, help="Override the default results/ file path")
    args = parser.parse_args(argv)

    if args.list_cases:
        for spec in AUTHORIZATION_CASES:
            print(f"{spec.name} | scenario={spec.scenario} | stage={spec.stage}/{spec.stage_count}")
        return 0
    model = args.model or DEFAULT_MODELS[args.provider]
    if args.provider == "openai" and not model.startswith("gpt-"):
        parser.error("OpenAI provider requires a gpt-* model ID")
    if args.provider == "anthropic" and not model.startswith("claude-"):
        parser.error("Anthropic provider requires a claude-* model ID")
    if args.provider == "anthropic" and args.reasoning_effort != "medium":
        parser.error("Reasoning effort is OpenAI-only; Anthropic uses the published adaptive thinking config")
    selected = select_cases(args.case, all_authorization=args.all_authorization)
    root = args.benchmark_dir.resolve()
    try:
        revision = check_revision(root)
        plan = build_plan(root, args.trials, selected)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"PRECHECK FAILED: {exc}", file=sys.stderr)
        return 2
    effort: str | None = args.reasoning_effort if args.provider == "openai" else None
    target = args.output or default_output(args.provider, model, args.trials, selected)
    print(f"Benchmark revision: {revision}")
    print(f"Cases: {len(selected)} directories across {len({CASE_CATALOG[c].scenario for c in selected})} scenarios")
    print(f"Provider: {args.provider}; model: {model}; reasoning: {effort or 'adaptive'}")
    print(f"Prompts: {', '.join(PROMPTS)}; attack + benign; trials per condition: {args.trials}")
    print(f"Planned requests: {len(plan)}; output: {target}")
    if not args.execute:
        print("DRY RUN COMPLETE: No model API requests or results generated.")
        return 0

    key_env = "OPENAI_API_KEY" if args.provider == "openai" else "ANTHROPIC_API_KEY"
    if not os.environ.get(key_env):
        print(f"Missing {key_env}; stopped before making calls", file=sys.stderr)
        return 2
    try:
        completed = load_completed(
            target, provider=args.provider, model=model, revision=revision,
            plan=plan, root=root, resume=args.resume, reasoning_effort=effort,
        )
    except RuntimeError as exc:
        print(f"OUTPUT PRECHECK FAILED: {exc}", file=sys.stderr)
        return 2
    remaining = [item for item in plan if condition_key(item) not in completed]
    print(f"Existing condition records: {len(completed)}; remaining billable calls: {len(remaining)}")
    if not remaining:
        print("All requested conditions are recorded; no API calls made.")
        return 0

    try:
        if args.provider == "openai":
            from openai import OpenAI
            client = OpenAI(max_retries=1)
        else:
            from anthropic import Anthropic
            client = Anthropic(max_retries=1)
        sys.path.insert(0, str(root))
        from evaluate.core.transcript import render_prompt, strip_assistant_text
        from evaluate.core.response import parse_response
    except ImportError as exc:
        print(f"Dependencies unavailable ({type(exc).__name__}); run uv sync --project vendor/sleight-bench --locked", file=sys.stderr)
        return 2

    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as output:
        for index, item in enumerate(remaining, 1):
            prompt = item["prompt_name"]
            transcript_path = item["transcript_path"]
            prepared = strip_assistant_text(transcript_path.read_text(encoding="utf-8"))
            prompt_path = root / "prompts" / f"{prompt}.md"
            system_prompt, user_message = render_prompt(prompt_path.read_text(encoding="utf-8"), prepared)
            print(f"Running {index}/{len(remaining)}: {item['case']}, {prompt}, benign={item['is_benign']}, trial={item['trial']}", flush=True)
            try:
                answer, refusal, stop_reason, usage = call_monitor(
                    args.provider, model, args.reasoning_effort, system_prompt,
                    user_message, client,
                )
                parsed = parse_response(answer)
                score = parsed.get("numeric_score")
                if refusal or type(score) is not int or not 0 <= score <= 100:
                    score = None
                row = {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "provider": args.provider,
                    "model": model,
                    "reasoning_effort": effort,
                    "temperature": 1.0 if args.provider == "anthropic" else None,
                    "thinking": True,
                    "strip_assistant": True,
                    "benchmark_commit": revision,
                    "case": item["case"],
                    "scenario": item["scenario"],
                    "stage": item["stage"],
                    "stage_count": item["stage_count"],
                    "prompt_name": prompt,
                    "is_benign": item["is_benign"],
                    "trial": item["trial"],
                    "transcript_sha256": sha256(transcript_path),
                    "prompt_sha256": sha256(prompt_path),
                    "numeric_score": score,
                    "status": "scored" if score is not None else "refusal_or_invalid",
                    "stop_reason": stop_reason,
                    **usage,
                }
            except Exception as exc:
                print(f"API FAILURE ({type(exc).__name__}); no input text or secret displayed. Repair and use --resume.", file=sys.stderr)
                return 3
            output.write(json.dumps(row, sort_keys=True) + "\n")
            output.flush()
    print(f"Completed. Score-only records: {target}. Summarize using: python summarize.py {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
