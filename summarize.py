"""Descriptive one-case summary only. Never estimates population FPR."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


def summarize(rows: list[dict]) -> dict:
    grouped: dict[tuple[str, bool], list[float]] = defaultdict(list)
    errors: dict[tuple[str, bool], int] = defaultdict(int)
    for row in rows:
        key = (str(row["prompt_name"]), bool(row["is_benign"]))
        score = row.get("numeric_score")
        if isinstance(score, (int, float)) and not isinstance(score, bool) and 0 <= score <= 100:
            grouped[key].append(float(score))
        else:
            errors[key] += 1
    prompts = sorted({name for name, _ in grouped.keys() | errors.keys()})
    summary = {}
    for prompt in prompts:
        attack = grouped[(prompt, False)]
        benign = grouped[(prompt, True)]
        summary[prompt] = {
            "attack_n": len(attack),
            "benign_n": len(benign),
            "attack_scores": attack,
            "benign_scores": benign,
            "attack_mean": mean(attack) if attack else None,
            "benign_mean": mean(benign) if benign else None,
            "mean_gap": mean(attack) - mean(benign) if attack and benign else None,
            "invalid_or_refusal": errors[(prompt, False)] + errors[(prompt, True)],
        }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", type=Path, default=Path("results/scores.jsonl"))
    args = ap.parse_args()
    rows = [json.loads(line) for line in args.path.read_text().splitlines() if line.strip()]
    result = summarize(rows)
    print(json.dumps(result, indent=2))
    print("\nCAUTION: One case; these are descriptive scores, NOT a catch rate at 1% FPR.")


if __name__ == "__main__":
    main()
