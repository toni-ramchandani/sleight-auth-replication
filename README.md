# Authorization-confusion replication pilot

Independent empirical replication project. **This repository does not alter or redistribute the Manning MCP book implementation or SLEIGHT-Bench datasets.**

## Hypothesis and scope

For the SLEIGHT-Bench case `transcript-perm-launder-corp-2part-stage1`, compare the original `control_arena` and published `control_arena_doc_planting_aware` monitor prompts, holding model settings and the harmful/benign transcript pair fixed.

This is a **one-case replication pilot**, not a novelty claim or reproduction of the paper's aggregate 1%-FPR detection rate. Neither experiment nor MCP extension has produced a model result yet.

## Primary sources

- [SLEIGHT-Bench paper](https://alignment.anthropic.com/2026/sleight-bench/)
- [Official benchmark](https://github.com/safety-research/sleight-bench), pinned at `218c58315cc01ff0dc5a100e906c27d82d259521`

The benchmark authors encrypt transcripts to avoid accidental ingestion into training corpora. Keep decrypted materials on your workstation and out of this Git repository, chat, and public reporting.

## Prepare on Windows PowerShell

Requires Git, Python 3.10+, uv, and (for execution) access to the Anthropic API. The pilot imports prompt preparation, stripping, provider configuration, and response parsing from the pinned upstream code. It saves only scores/metadata to local JSONL; the official benchmark uses BigQuery.

```powershell
git clone https://github.com/safety-research/sleight-bench.git vendor/sleight-bench
git -C vendor/sleight-bench checkout 218c58315cc01ff0dc5a100e906c27d82d259521
uv sync --project vendor/sleight-bench --locked

# Insert the published decryption key from the upstream README:
uv run --project vendor/sleight-bench python vendor/sleight-bench/decrypt.py --key '<KEY FROM UPSTREAM README>'

# Check the experiment plan without making model calls:
uv run --project vendor/sleight-bench python run_case.py --trials 1

# Set ANTHROPIC_API_KEY privately in your environment, then run 4 API calls:
uv run --project vendor/sleight-bench python run_case.py --trials 1 --execute
uv run --project vendor/sleight-bench python summarize.py
```

After the smoke test, `--trials 10 --execute` requests **40** evaluations (2 prompts × 2 labels × 10 trials). These calls can be costly. Results are append-only; change `--output` or archive previous scores before rerunning.

## Fixed variables

- Model: `claude-opus-4-6`
- Temperature 1.0; adaptive thinking enabled; assistant text stripped
- Same attack and benign transcripts across prompt conditions
- Published monitor prompts unchanged
- Original upstream prompt-preparation and score-parsing functions

## Limitations

One pair cannot estimate a population false-positive rate or reproduce the authors' aggregate result. Small numbers of repeated calls on one pair measure stochastic variation, not independent attacks. The simplified runner lacks some upstream result-storage and retry behavior. A secure MCP server refusing unauthorized calls and a monitor identifying unsafe agent intent are distinct controls; this pilot cannot show a real MCP policy bypass.

The Manning source repository remains unchanged. A separate, executable MCP transfer test should be designed only after assessing replication results.

## Offline tests

```powershell
python -m unittest discover -s tests -v
```
