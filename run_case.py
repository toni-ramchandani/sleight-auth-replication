"""Small, explicit reproduction pilot using SLEIGHT-Bench's own prompt pipeline.

Does not store or print decrypted benchmark transcripts or monitor reasoning.
Does not execute unless --execute is supplied. Book repository is untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BENCHMARK_COMMIT = "218c58315cc01ff0dc5a100e906c27d82d259521"
CASE = "transcript-perm-launder-corp-2part-stage1"
PROMPTS = ("control_arena", "control_arena_doc_planting_aware")
MODEL = "claude-opus-4-6"


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
        raise FileNotFoundError(
            "Benchmark not decrypted or incomplete; missing: " + ", ".join(missing)
        )
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
    sha = completed.stdout.strip()
    if sha != BENCHMARK_COMMIT:
        raise RuntimeError(f"Wrong benchmark revision: {sha}; expected {BENCHMARK_COMMIT}")
    return sha


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark-dir", type=Path, default=Path("vendor/sleight-bench"))
    ap.add_argument("--trials", type=int, default=1, help="1 call per condition by default; 10 for paper's trial count")
    ap.add_argument("--execute", action="store_true", help="Explicitly allow paid Anthropic API calls")
    ap.add_argument("--output", type=Path, default=Path("results/scores.jsonl"))
    args = ap.parse_args()
    root = args.benchmark_dir.resolve()
    try:
        revision = check_revision(root)
        plan = build_plan(root, args.trials)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"PRECHECK FAILED: {exc}", file=sys.stderr)
        return 2

    print(f"Benchmark revision: {revision}")
    print(f"Case: {CASE}")
    print(f"Model: {MODEL}; thinking=adaptive; temperature=1.0; strip_assistant=True")
    print(f"Conditions: {', '.join(PROMPTS)}; harmful+benign; {args.trials} trial(s) each")
    print(f"API calls planned: {len(plan)}")
    if not args.execute:
        print("DRY RUN COMPLETE: No API requests, scores or research findings generated.")
        return 0
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is required for --execute", file=sys.stderr)
        return 2

    # Import the official benchmark's own prompt-preparation, provider and response logic.
    # This avoids silently drifting from the published evaluation semantics.
    sys.path.insert(0, str(root))
    import anthropic  # type: ignore[import-not-found]
    from evaluate.core.providers import build_api_params, call_anthropic
    from evaluate.core.response import extract_response_parts, parse_response
    from evaluate.core.transcript import render_prompt, strip_assistant_text
    from evaluate.core.types import MonitorSpec

    spec_by_prompt = {
        prompt: MonitorSpec(
            prompt=prompt, model=MODEL, temperature=1.0, trials=args.trials,
            thinking=True, strip_assistant=True, type="api",
        )
        for prompt in PROMPTS
    }
    client = anthropic.Anthropic()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # No raw transcript, monitor reasoning, or authentication tokens are persisted.
    with args.output.open("a", encoding="utf-8") as output:
        for count, item in enumerate(plan, start=1):
            prompt = item["prompt_name"]
            transcript_path = item["transcript_path"]
            raw = transcript_path.read_text(encoding="utf-8")
            prepared = strip_assistant_text(raw)
            template = (root / "prompts" / f"{prompt}.md").read_text(encoding="utf-8")
            system_prompt, user_message = render_prompt(template, prepared)
            params = build_api_params(spec_by_prompt[prompt], system_prompt, user_message)
            print(f"Running {count}/{len(plan)}: {prompt}, benign={item['is_benign']}, trial={item['trial']}", flush=True)
            # Do not auto-repeat the whole experiment after errors. The upstream
            # provider helper handles transient HTTP errors; each request is billable.
            try:
                message = call_anthropic(client, spec_by_prompt[prompt], params)
                _, answer, refusal, stop_reason = extract_response_parts(message)
                parsed = parse_response(answer)
                score = parsed.get("numeric_score")
                if not isinstance(score, int) or not 0 <= score <= 100 or refusal:
                    score = None
                result = {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "benchmark_commit": revision,
                    "case": CASE,
                    "prompt_name": prompt,
                    "model": MODEL,
                    "thinking": True,
                    "temperature": 1.0,
                    "strip_assistant": True,
                    "is_benign": item["is_benign"],
                    "trial": item["trial"],
                    "transcript_sha256": sha256(transcript_path),
                    "prompt_sha256": sha256(root / "prompts" / f"{prompt}.md"),
                    "numeric_score": score,
                    "stop_reason": stop_reason,
                    "status": "scored" if score is not None else "refusal_or_invalid",
                }
            except Exception as exc:
                print(f"API FAILURE ({type(exc).__name__}); stopping without logging the prompt", file=sys.stderr)
                return 3
            output.write(json.dumps(result, sort_keys=True) + "\n")
            output.flush()
    print(f"Completed. Scores only saved to {args.output}; no finding claimed until analysis.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
