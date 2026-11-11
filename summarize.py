"""Describe one-pair monitor scores. Never estimates population-level FPR."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


def summarize(rows: list[dict]) -> dict:
    grouped: dict[tuple[str, str, str, bool], list[float]] = defaultdict(list)
    errors: dict[tuple[str, str, str, bool], int] = defaultdict(int)
    for row in rows:
        key = (
            str(row.get("provider", "unknown")), str(row.get("model", "unknown")),
            str(row["prompt_name"]), bool(row["is_benign"]),
        )
        score = row.get("numeric_score")
        if isinstance(score, (int, float)) and not isinstance(score, bool) and 0 <= score <= 100:
            grouped[key].append(float(score))
        else:
            errors[key] += 1
    configs = sorted({key[:3] for key in grouped.keys() | errors.keys()})
    result = {}
    for provider, model, prompt in configs:
        attack = grouped[(provider, model, prompt, False)]
        benign = grouped[(provider, model, prompt, True)]
        name = f"{provider}/{model}/{prompt}"
        result[name] = {
            "attack_n": len(attack),
            "benign_n": len(benign),
            "attack_scores": attack,
            "benign_scores": benign,
            "attack_mean": mean(attack) if attack else None,
            "benign_mean": mean(benign) if benign else None,
            "mean_gap": mean(attack) - mean(benign) if attack and benign else None,
            "invalid_or_refusal": errors[(provider, model, prompt, False)] + errors[(provider, model, prompt, True)],
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Results JSONL path printed by run_case.py")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.path.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(json.dumps(summarize(rows), indent=2))
    print("\nCAUTION: one attack/benign pair; these descriptive differences are NOT catch rates or FPR estimates.")


if __name__ == "__main__":
    main()
