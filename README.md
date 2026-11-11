# SLEIGHT-Bench authorization-confusion experiments

An independent evaluation harness for the **six transcript directories / four scenarios** in SLEIGHT-Bench's authorization-confusion category. Two scenarios consist of two stages each. This code uses the original benchmark's prompt/transcript-processing functions and supports both **OpenAI** and **Anthropic** monitoring models.

**No benchmark data, real model outputs, credentials, or Manning book code are committed here.** The Manning book repository remains unmodified.

## Scientific questions and limitations

1. How do the authors' published `control_arena` baseline and `control_arena_doc_planting_aware` monitors score each original harmful/benign pair?
2. Are monitoring decisions stable across four independent scenarios, particularly when the **same scenario has a stage-1 and a stage-2 transcript**?
3. Which cases and stages are missed by a fixed, *illustrative* suspicion threshold?

**What this is not:** an original prompt intervention, a population false-positive-rate estimate, or a demonstration of real MCP authorization enforcement. SLEIGHT-Bench already tested these prompts. An OpenAI run is cross-model exploration, **not** a model-matched reproduction of the published Anthropic results. A stage-1 and stage-2 model call **does not expose accumulated conversational state**: each transcript is presented independently; stage-aware analysis happens afterward. This avoids pretending that the code has recreated a live multi-session agent.

Primary sources:

- [SLEIGHT-Bench research](https://alignment.anthropic.com/2026/sleight-bench/)
- [Official implementation and encrypted dataset](https://github.com/safety-research/sleight-bench)
- Pinned SLEIGHT-Bench release: `218c58315cc01ff0dc5a100e906c27d82d259521`
- [OpenAI Responses API](https://platform.openai.com/docs/api-reference/responses/create)

## Experiment selection

| CLI | Coverage | Calls at one trial |
| --- | --- | ---: |
| No case option | Original stage-1 corp case, backward compatible | 4 |
| `--case preprocess-sabotage` | One requested case | 4 |
| Repeated `--case` values | Chosen case directories | 4 per directory |
| **`--all-authorization`** | **Six directories, four scenarios** | **24** |

Case grouping is explicit and pinned to the original dataset version:

- `ambiguous-instruction-weight-backup-orion-personal` — standalone
- `preprocess-sabotage` — standalone
- `transcript-perm-launder-corp-2part-stage1`, `transcript-perm-launder-corp-2part-stage2` — a two-stage scenario
- `transcript-perm-launder-gcs-2part-stage1`, `transcript-perm-launder-gcs-2part-stage2` — a two-stage scenario

View the mapping without any dataset: `python run_case.py --list-cases`.

## Windows setup

Run commands from your project root, for example `C:\Toni\Work\sleight-auth-replication`.

**If you previously extracted a ZIP:** the project root may not be a Git checkout. Download the updated project ZIP or copy `run_case.py`, `summarize.py`, `tests/test_pilot.py`, and this README from GitHub into your project root. **Preserve the `vendor/` directory and `results/` files.** Do not run `git pull` from a ZIP-extracted directory without first initializing it as a Git repository.

You already cloned and decrypted SLEIGHT-Bench for Experiment 1, so **do not repeat these commands if it is ready**. For a clean install only:

```powershell
# Requires Git, Python >= 3.10, and uv
git clone https://github.com/safety-research/sleight-bench.git vendor/sleight-bench
git -C vendor/sleight-bench checkout 218c58315cc01ff0dc5a100e906c27d82d259521
uv sync --project vendor/sleight-bench --locked

# Obtain the decryption key from the official benchmark README.
# Decrypted data MUST remain only on your workstation.
uv run --project vendor/sleight-bench python vendor/sleight-bench/decrypt.py --key '<PUBLISHED BENCHMARK KEY>'
```

The benchmark authors intentionally encrypt files to limit accidental inclusion in model-training corpora. **Do not publish decrypted transcripts, prompt+transcript payloads, sample transcript rows, or complete model reasoning.** Executing this benchmark necessarily sends the decrypted transcript to the selected model provider's API; confirm applicable data terms. The OpenAI path uses `store=False` in the Responses API.

## First: offline tests and a no-cost plan

```powershell
python -m unittest discover -s tests -v

# No API key and no paid requests:
uv run --project vendor/sleight-bench python run_case.py --all-authorization --provider openai --model gpt-6.1-sol --trials 1
```

Expected output includes:

```text
Cases: 6 directories across 4 scenarios
Planned requests: 24; output: results\openai-gpt-6.1-sol-auth6-t1.jsonl
DRY RUN COMPLETE: No model API requests or results generated.
```

The plan is **six directories × two monitor prompts × two labels (harmful/benign) × one trial**.

## Execute with OpenAI

For the initial exploratory evaluation, use **`gpt-6.1-sol` at `medium` reasoning effort**. An OpenAI API key, not an Anthropic key, is required.

```powershell
$secureKey = Read-Host 'OpenAI API key' -AsSecureString
$env:OPENAI_API_KEY = (New-Object System.Net.NetworkCredential('', $secureKey)).Password
Remove-Variable secureKey

# 24 billable model requests; explicit consent required through --execute.
uv run --project vendor/sleight-bench python run_case.py --provider openai --model gpt-6.1-sol --reasoning-effort medium --all-authorization --trials 1 --execute

# Analyze scores; threshold 70 is illustrative, not 1% FPR calibrated.
uv run --project vendor/sleight-bench python summarize.py results/openai-gpt-6.1-sol-auth6-t1.jsonl --threshold 70
```

The existing Experiment 1 file `results/openai-gpt-6.1-sol-t1.jsonl` remains untouched; the expanded experiment has a different filename.

## Execute with Anthropic (optional)

For model-matched comparison, the published monitor model is `claude-opus-4-6`, with adaptive thinking. Anthropic billing and access are separate from OpenAI.

```powershell
$secureKey = Read-Host 'Anthropic API key' -AsSecureString
$env:ANTHROPIC_API_KEY = (New-Object System.Net.NetworkCredential('', $secureKey)).Password
Remove-Variable secureKey

# 24 billable requests with the published Anthropic configuration.
uv run --project vendor/sleight-bench python run_case.py --provider anthropic --model claude-opus-4-6 --all-authorization --trials 1 --execute
uv run --project vendor/sleight-bench python summarize.py results/anthropic-claude-opus-4-6-auth6-t1.jsonl --threshold 70
```

Neither provider requires credentials for the other. Both use the same unmodified benchmark transcripts and monitoring prompts. They do **not** have guaranteed equivalent token budgets or reasoning behavior.

## Resume interrupted experiments safely

Model calls run sequentially and append score-only JSONL records after every completed call. If an API error interrupts the run after some conditions, preserve the file, correct the error, and **repeat the same command with `--resume`**:

```powershell
uv run --project vendor/sleight-bench python run_case.py --provider openai --model gpt-6.1-sol --reasoning-effort medium --all-authorization --trials 1 --execute --resume
```

Resume rejects existing result files with duplicate records, wrong provider/model/effort, incompatible benchmark revision, mutated transcript/prompt hashes, malformed score/status, or cases outside the selected plan. `refusal_or_invalid` records represent completed **attempts** and are not silently retried as new trials; the summary reports them as unscored observations. For a fresh experiment or changed model/effort, supply a new `--output` filename. Do **not** delete existing results to force retries unless you intend to discard their provenance.

Output paths live under gitignored `results/`. Data recorded: case/scenario/stage ID, label, prompt, trial, score or invalid outcome, provider/model/settings, token usage, and provenance hashes. No transcript content, model reasoning, or secrets are written.

## Interpreting the analysis

`summary.py` is named **`summarize.py`** and reports:

- `by_monitor`: scores pooled over the **selected case directories** for each provider/model/prompt (descriptive only; multi-stage scenarios count twice here).
- `by_case`: harmful/benign score arrays, mean gaps, invalid counts, and numbers above the **predeclared illustrative** `--threshold`.
- `stage_aware`: each scenario's available stages, per-stage scores and missing data; for two-stage scenarios, *matched-trial* changes between stage 1 and stage 2, and the earliest **independently scored stage** crossing the threshold. A complete stage trajectory requires scored harmful and benign observations at every original stage for the same trial index.

**Do not claim a population catch rate, calibrated false-positive rate, or causal effectiveness from this small dataset.** Stage-wise results can identify interesting cases for a separate executable MCP transfer experiment. That extension does not yet exist and should not be conflated with these transcript-only monitor calls.

## Developer tests and continuous integration

```powershell
python -m unittest discover -s tests -v
python -m compileall -q run_case.py summarize.py tests
```

Offline tests use synthetic placeholder strings, never original benchmark examples. CI intentionally runs only these offline checks: it does not decrypt data or contact model APIs.
