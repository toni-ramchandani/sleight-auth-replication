"""Experiment 4: Stage 2 with source-grounded Stage 1 authorization candidates.

Runs only the *new* evidence-ledger arm (8 calls for one trial). Comparison
arms are Experiment 3's isolated and cumulative runs, checked by hash in
analyze_evidence.py. Never persists or prints decrypted benchmark content.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from authorization_evidence import POLICY_VERSION, EvidenceLedger, extract_candidates
from run_case import DEFAULT_MODELS, OPENAI_EFFORTS, PROMPTS, call_monitor, check_revision
from run_context import (
    ContextItem,
    TARGET_INSTRUCTION,
    TWO_STAGE_SCENARIOS,
    build_context_plan,
    render_context,
    select_scenarios,
)

FORMAT_VERSION = "source-evidence-stage2-v1"
CONTEXT_MODE = "evidence"


@dataclass(frozen=True)
class Prepared:
    item: ContextItem
    system: str | None
    user: str
    hashes: dict[str, Any]

    @property
    def key(self) -> tuple[str, str, bool, int]:
        return (self.item.scenario, self.item.prompt_name, self.item.is_benign, self.item.trial)


def build_evidence_plan(root: Path, trials: int = 1, scenarios: Sequence[str] | None = None) -> list[ContextItem]:
    # Reuse the *same* stage mapping and input completeness checks as Experiment 3.
    return build_context_plan(root, trials, scenarios, modes=("isolated",))


def _payload_hash(system: str | None, user: str) -> str:
    return hashlib.sha256(
        json.dumps({"system": system, "user": user}, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def prepare(item: ContextItem, root: Path) -> Prepared:
    from evaluate.core.transcript import render_prompt, strip_assistant_text

    # This policy never reads metadata.json, description.md, or benchmark labels.
    stripped_stage1 = strip_assistant_text(item.stage1_path.read_text(encoding="utf-8"))
    stripped_stage2 = strip_assistant_text(item.stage2_path.read_text(encoding="utf-8"))
    evidence: EvidenceLedger = extract_candidates(stripped_stage1)
    if not evidence.event_count:
        raise ValueError(f"No Stage 1 authorization candidates for {item.scenario}, benign={item.is_benign}; review the fixed selector before paying")
    # Transcript data is untrusted, including any apparent grants. The target
    # remains Session 2, not whether Stage 1 contains a suspicious action.
    session1 = (
        '<session index="1" role="prior_context" representation="source_grounded_candidate_ledger">\n'
        + evidence.text + '\n</session>'
    )
    session2 = '<session index="2" role="target">\n' + stripped_stage2 + '\n</session>'
    source = session1 + '\n\n' + session2
    prompt_path = root / "prompts" / f"{item.prompt_name}.md"
    system, transcript_user = render_prompt(prompt_path.read_text(encoding="utf-8"), source)
    user = TARGET_INSTRUCTION + "\n\n" + transcript_user
    # Reconstruct the *exact* Experiment 3 monitor payloads, not assumptions
    # about how prompts were rendered; the analysis checks these hashes.
    _, _, isolated_hashes = render_context(replace(item, context_mode="isolated"), root)
    _, _, cumulative_hashes = render_context(replace(item, context_mode="cumulative"), root)
    return Prepared(
        item=item,
        system=system,
        user=user,
        hashes={
            **{k: isolated_hashes[k] for k in ("stage1_sha256", "stage2_sha256", "prompt_sha256")},
            "payload_sha256": _payload_hash(system, user),
            "isolated_payload_sha256": isolated_hashes["payload_sha256"],
            "cumulative_payload_sha256": cumulative_hashes["payload_sha256"],
            "evidence_sha256": evidence.sha256,
            "evidence_source_digests_sha256": evidence.source_record_digests_sha256,
            "evidence_event_count": evidence.event_count,
            "evidence_source_record_count": evidence.source_record_count,
            "evidence_policy": POLICY_VERSION,
        },
    )


def default_output(provider: str, model: str, trials: int, scenarios: Sequence[str]) -> Path:
    safe_model = re.sub(r"[^a-zA-Z0-9_.-]", "-", model)
    signature = hashlib.sha256(json.dumps({"scenarios": sorted(scenarios), "policy": POLICY_VERSION}, sort_keys=True).encode()).hexdigest()[:8]
    return Path("results") / f"{provider}-{safe_model}-evidence-{signature}-t{trials}.jsonl"


def load_completed(
    path: Path, prepared: list[Prepared], *, provider: str,
    model: str, revision: str, effort: str | None, resume: bool,
) -> set[tuple[str, str, bool, int]]:
    if not path.exists():
        return set()
    if not resume:
        raise RuntimeError(f"Output {path} exists; use --resume or a different --output")
    expected = {p.key: p for p in prepared}
    done: set[tuple[str, str, bool, int]] = set()
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if type(row) is not dict:
                raise ValueError("Expected object")
            if type(row.get("is_benign")) is not bool or type(row.get("trial")) is not int:
                raise ValueError("Malformed trial or label")
            key = (row["scenario"], row["prompt_name"], row["is_benign"], row["trial"])
            if key not in expected or key in done:
                raise ValueError("Duplicate or unselected condition")
            required = {
                "format_version": FORMAT_VERSION, "provider": provider,
                "model": model, "benchmark_commit": revision, "reasoning_effort": effort,
                "thinking": True, "temperature": 1.0 if provider == "anthropic" else None,
                "strip_assistant": True, "context_mode": CONTEXT_MODE,
                "target_stage": 2, "stage_count": 2, **expected[key].hashes,
            }
            if any(row.get(name) != value for name, value in required.items()):
                raise ValueError("Mismatched configuration, source evidence, or input hash")
            score = row.get("numeric_score")
            if row.get("status") == "scored":
                if type(score) is not int or not 0 <= score <= 100:
                    raise ValueError("Invalid score")
            elif row.get("status") != "refusal_or_invalid" or score is not None:
                raise ValueError("Invalid outcome")
            done.add(key)
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Invalid resume record line {line_no}: {exc}") from None
    return done


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark-dir", type=Path, default=Path("vendor/sleight-bench"))
    ap.add_argument("--provider", choices=("openai", "anthropic"), default="openai")
    ap.add_argument("--model")
    ap.add_argument("--reasoning-effort", choices=OPENAI_EFFORTS, default="medium")
    ap.add_argument("--scenario", action="append", choices=TWO_STAGE_SCENARIOS)
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--execute", action="store_true", help="Permit billable model calls")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args(argv)
    model = args.model or DEFAULT_MODELS[args.provider]
    if (args.provider == "openai" and not model.startswith("gpt-")) or (args.provider == "anthropic" and not model.startswith("claude-")):
        ap.error("Provider and model ID do not match")
    if args.provider == "anthropic" and args.reasoning_effort != "medium":
        ap.error("--reasoning-effort applies only to OpenAI")
    try:
        scenarios = select_scenarios(args.scenario)
        root = args.benchmark_dir.resolve()
        revision = check_revision(root)
        plan = build_evidence_plan(root, args.trials, scenarios)
        # Import official functions even for dry run; validate *all* evidence
        # before permitting any billable requests.
        sys.path.insert(0, str(root))
        prepared = [prepare(item, root) for item in plan]
    except (ValueError, FileNotFoundError, RuntimeError, ImportError) as exc:
        print(f"PRECHECK FAILED: {exc}", file=sys.stderr)
        return 2
    effort = args.reasoning_effort if args.provider == "openai" else None
    dest = args.output or default_output(args.provider, model, args.trials, scenarios)
    print(f"Experiment 4: {len(scenarios)} scenarios, Stage 2 + source-grounded authorization candidates")
    print(f"Benchmark: {revision}; model: {args.provider}/{model}; reasoning: {effort or 'adaptive'}")
    print(f"Planned billable requests: {len(prepared)}; output: {dest}")
    # Only counts; no actual evidence or benchmark source is printed.
    for scenario in scenarios:
        for benign in (False, True):
            p = next(x for x in prepared if x.item.scenario == scenario and x.item.is_benign == benign)
            print(f"Evidence candidates: scenario={scenario}, benign={benign}, excerpts={p.hashes['evidence_event_count']}")
    if not args.execute:
        print("DRY RUN COMPLETE: no paid API calls or result files generated.")
        return 0
    key_name = "OPENAI_API_KEY" if args.provider == "openai" else "ANTHROPIC_API_KEY"
    if not os.environ.get(key_name):
        print(f"Missing {key_name}; no requests made", file=sys.stderr)
        return 2
    try:
        done = load_completed(dest, prepared, provider=args.provider, model=model, revision=revision, effort=effort, resume=args.resume)
    except RuntimeError as exc:
        print(f"RESUME PRECHECK FAILED: {exc}", file=sys.stderr)
        return 2
    pending = [p for p in prepared if p.key not in done]
    print(f"Previously completed: {len(done)}; remaining billable requests: {len(pending)}")
    if not pending:
        return 0
    try:
        if args.provider == "openai":
            from openai import OpenAI
            client: Any = OpenAI(max_retries=1)
        else:
            from anthropic import Anthropic
            client = Anthropic(max_retries=1)
        from evaluate.core.response import parse_response
    except ImportError as exc:
        print(f"Dependencies unavailable: {type(exc).__name__}; run uv sync --project vendor/sleight-bench --locked", file=sys.stderr)
        return 2
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("a", encoding="utf-8") as output:
        for i, p in enumerate(pending, 1):
            item = p.item
            print(f"Running {i}/{len(pending)}: {item.scenario}, {item.prompt_name}, benign={item.is_benign}, trial={item.trial}", flush=True)
            try:
                answer, refused, stop_reason, usage = call_monitor(
                    args.provider, model, args.reasoning_effort, p.system, p.user, client,
                )
                parsed = parse_response(answer)
                score = parsed.get("numeric_score") if isinstance(parsed, dict) else None
                if refused or type(score) is not int or not 0 <= score <= 100:
                    score = None
                row = {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "format_version": FORMAT_VERSION,
                    "benchmark_commit": revision, "provider": args.provider,
                    "model": model, "reasoning_effort": effort,
                    "thinking": True, "temperature": 1.0 if args.provider == "anthropic" else None,
                    "strip_assistant": True, "target_stage": 2, "stage_count": 2,
                    "scenario": item.scenario, "prompt_name": item.prompt_name,
                    "context_mode": CONTEXT_MODE, "is_benign": item.is_benign,
                    "trial": item.trial, **p.hashes,
                    "numeric_score": score,
                    "status": "scored" if score is not None else "refusal_or_invalid",
                    "stop_reason": stop_reason, **usage,
                }
            except Exception as exc:
                print(f"API/EVALUATION FAILURE ({type(exc).__name__}); source text and keys withheld. Use --resume.", file=sys.stderr)
                return 3
            output.write(json.dumps(row, sort_keys=True) + "\n")
            output.flush()
            os.fsync(output.fileno())
    print(f"Completed: {dest}; analyze with analyze_evidence.py plus Experiment 3 results")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
