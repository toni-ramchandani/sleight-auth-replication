"""Run one SLEIGHT-Bench case with either an Anthropic or an OpenAI monitor.

Uses the official benchmark's transcript, prompt rendering, and score parser.
Never uploads decrypted benchmark data to GitHub or persists monitor reasoning.
Calls are billable ONLY with the explicit --execute flag.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BENCHMARK_COMMIT = "218c58315cc01ff0dc5a100e906c27d82d259521"
CASE = "transcript-perm-launder-corp-2part-stage1"
PROMPTS = ("control_arena", "control_arena_doc_planting_aware")
DEFAULT_MODELS = {"anthropic": "claude-opus-4-6", "openai": "gpt-6.1-sol"}
OPENAI_EFFORTS = ("none", "low", "medium", "high", "xhigh", "max")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_plan(root: Path, trials: int) -> list[dict[str, Any]]:
    if not 1 <= trials <= 10:
        raise ValueError("pilot trials must be 1..10")
    base = root / "attacks" / "authorization_confusion" / CASE
    required = [base / "transcript.jsonl", base / "benign.jsonl"]
    required += [root / "prompts" / f"{p}.md" for p in PROMPTS]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("Benchmark not decrypted or incomplete; missing: " + ", ".join(missing))
    return [
        {
            "prompt_name": prompt,
            "transcript_path": base / ("benign.jsonl" if benign else "transcript.jsonl"),
            "is_benign": benign,
            "trial": i,
        }
        for prompt in PROMPTS
        for benign in (False, True)
        for i in range(trials)
    ]


def check_revision(root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=10,
        )
    except (subprocess.SubprocessError, OSError):
        raise RuntimeError("Benchmark must be a Git checkout pinned to the published commit") from None
    revision = completed.stdout.strip()
    if revision != BENCHMARK_COMMIT:
        raise RuntimeError(f"Wrong benchmark revision: {revision}; expected {BENCHMARK_COMMIT}")
    return revision


def default_output(provider: str, model: str, trials: int) -> Path:
    safe_model = re.sub(r"[^a-zA-Z0-9_.-]", "-", model)
    return Path("results") / f"{provider}-{safe_model}-t{trials}.jsonl"


def condition_key(item: dict[str, Any]) -> tuple[str, bool, int]:
    return (item["prompt_name"], item["is_benign"], item["trial"])


def load_completed(
    path: Path, *, provider: str, model: str, revision: str,
    plan: list[dict[str, Any]], root: Path, resume: bool,
    reasoning_effort: str | None,
) -> set[tuple[str, bool, int]]:
    if not path.exists():
        return set()
    if not resume:
        raise RuntimeError(
            f"Output {path} already exists. To complete the existing experiment use --resume; "
            "otherwise choose a different --output path."
        )
    valid = {condition_key(item): item for item in plan}
    completed: set[tuple[str, bool, int]] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            key = (row["prompt_name"], row["is_benign"], row["trial"])
            if (key not in valid or key in completed or row["provider"] != provider
                    or row["model"] != model or row["benchmark_commit"] != revision
                    or row["case"] != CASE or row.get("reasoning_effort") != reasoning_effort
                    or row["transcript_sha256"] != sha256(valid[key]["transcript_path"])
                    or row["prompt_sha256"] != sha256(root / "prompts" / f"{row['prompt_name']}.md")):
                raise ValueError("metadata mismatch, duplicate, or unexpected condition")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Cannot resume {path}: invalid record at line {line_number}: {exc}") from None
        completed.add(key)
    return completed


def call_monitor(
    provider: str, model: str, effort: str,
    system_prompt: str | None, user_message: str,
    client: Any,
) -> tuple[str, bool, str | None, dict[str, int | None]]:
    """Return response text, refusal flag, stop reason, and token usage."""
    if provider == "openai":
        # Reasoning models may reject temperature; intentionally omit it.
        # store=False avoids persisting this benchmark transcript as a stored Response.
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
        content = getattr(response, "output_text", "") or ""
        return content, False, status, usage_from_openai(response)

    # Preserve upstream model-matched Anthropic API configuration and parser.
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


def usage_from_openai(response: Any) -> dict[str, int | None]:
    usage = getattr(response, "usage", None)
    details = getattr(usage, "output_tokens_details", None)
    return {
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
        "reasoning_tokens": getattr(details, "reasoning_tokens", None),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, default=Path("vendor/sleight-bench"))
    parser.add_argument("--provider", choices=("openai", "anthropic"), default="openai")
    parser.add_argument("--model", help="API model ID; defaults to gpt-6.1-sol or claude-opus-4-6")
    parser.add_argument("--reasoning-effort", choices=OPENAI_EFFORTS, default="medium",
                        help="OpenAI reasoning effort (ignored by the published Anthropic configuration)")
    parser.add_argument("--trials", type=int, default=1, help="Trials per prompt/label condition (1..10)")
    parser.add_argument("--execute", action="store_true", help="Explicitly authorize paid API calls")
    parser.add_argument("--resume", action="store_true", help="Skip already-scored records in existing output")
    parser.add_argument("--output", type=Path, help="Output JSONL; default is provider/model-specific")
    args = parser.parse_args(argv)
    model = args.model or DEFAULT_MODELS[args.provider]
    if args.provider == "openai" and not model.startswith("gpt-"):
        parser.error("OpenAI provider requires a GPT model ID")
    if args.provider == "anthropic" and not model.startswith("claude-"):
        parser.error("Anthropic provider requires a claude-* model ID")
    if args.provider == "anthropic" and args.reasoning_effort != "medium":
        parser.error("--reasoning-effort only controls the OpenAI path; Anthropic uses published adaptive thinking")
    root = args.benchmark_dir.resolve()
    try:
        revision = check_revision(root)
        plan = build_plan(root, args.trials)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"PRECHECK FAILED: {exc}", file=sys.stderr)
        return 2

    effort: str | None = args.reasoning_effort if args.provider == "openai" else None
    target = args.output or default_output(args.provider, model, args.trials)
    print(f"Benchmark revision: {revision}")
    print(f"Case: {CASE}")
    print(f"Provider: {args.provider}; model: {model}; reasoning: {effort or 'adaptive'}")
    print("Conditions: control_arena + control_arena_doc_planting_aware; harmful + benign")
    print(f"Planned requests: {len(plan)}; output: {target}")
    if not args.execute:
        print("DRY RUN COMPLETE: No API calls or research results generated.")
        return 0

    key_env = "OPENAI_API_KEY" if args.provider == "openai" else "ANTHROPIC_API_KEY"
    if not os.environ.get(key_env):
        print(f"Missing {key_env}; stop before making any calls", file=sys.stderr)
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
    print(f"Previously completed: {len(completed)}; remaining billable calls: {len(remaining)}")
    if not remaining:
        print("All requested conditions are already scored. No API calls made.")
        return 0

    # SDK imports are delayed so dry run needs no API packages or credentials.
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
        print(f"Dependency unavailable: {exc}; reinstall with uv sync --project vendor/sleight-bench --locked", file=sys.stderr)
        return 2

    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as output:
        for index, item in enumerate(remaining, 1):
            prompt = item["prompt_name"]
            transcript_path = item["transcript_path"]
            prepared = strip_assistant_text(transcript_path.read_text(encoding="utf-8"))
            prompt_path = root / "prompts" / f"{prompt}.md"
            system_prompt, user_message = render_prompt(prompt_path.read_text(encoding="utf-8"), prepared)
            print(f"Running {index}/{len(remaining)}: {prompt}, benign={item['is_benign']}, trial={item['trial']}", flush=True)
            try:
                answer, refusal, stop_reason, usage = call_monitor(
                    args.provider, model, args.reasoning_effort, system_prompt,
                    user_message, client,
                )
                parsed = parse_response(answer)
                score = parsed.get("numeric_score")
                if refusal or not isinstance(score, int) or not 0 <= score <= 100:
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
                    "case": CASE,
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
                print(f"API FAILURE ({type(exc).__name__}); no transcript or key displayed. Fix and use --resume.", file=sys.stderr)
                return 3
            output.write(json.dumps(row, sort_keys=True) + "\n")
            output.flush()
    print(f"Completed. Saved score-only records to {target}; use summarize.py {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
