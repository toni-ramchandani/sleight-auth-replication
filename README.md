# SLEIGHT-Bench authorization-confusion monitor pilot

Independent evaluation harness for **one** SLEIGHT-Bench authorization-confusion case, with interchangeable OpenAI and Anthropic API monitors. This repository **does not modify or redistribute** the Manning book server, or the encrypted/decrypted benchmark dataset.

## Research question and scientific scope

Compare two **already-published** monitor prompts on the attack and paired benign transcript for `transcript-perm-launder-corp-2part-stage1`:

1. `control_arena` (original baseline)
2. `control_arena_doc_planting_aware` (authors' documentation-aware ablation)

The OpenAI path is an exploratory **cross-model evaluation**, not a model-matched reproduction of published Anthropic results. The Anthropic path uses the paper's `claude-opus-4-6` configuration for a narrow replication pilot. Four calls on one attack/benign pair do **not** estimate an aggregate catch rate or FPR. No results are supplied or claimed in this repository.

### Primary sources

- [SLEIGHT-Bench publication](https://alignment.anthropic.com/2026/sleight-bench/)
- [Official SLEIGHT-Bench dataset and implementation](https://github.com/safety-research/sleight-bench), pinned to `218c58315cc01ff0dc5a100e906c27d82d259521`
- [OpenAI models](https://developers.openai.com/api/docs/models) and [Responses API](https://developers.openai.com/api/reference/resources/responses/methods/create)

## Models

| Provider | Default model | Configuration | Purpose |
|---|---|---|---|
| OpenAI | `gpt-6.1-sol` | Responses API, `reasoning.effort=medium`, no temperature | Recommended first experiment with existing OpenAI credits |
| Anthropic | `claude-opus-4-6` | Original benchmark's adaptive thinking and `temperature=1.0` | Model-matched pilot |

Optional OpenAI models: `gpt-6-luna` for an inexpensive pipeline check, `gpt-6-astra` for higher-capability comparison. Available models and API rates depend on your account. Do not infer equal inference effort between providers. Keep the provider/model/effort fixed when comparing the two prompts.

## Windows PowerShell setup

Run from the project root (`C:\Toni\Work\sleight-auth-replication`). Requires Git, `uv`, Python 3.10+ and the API key for whichever provider you execute.

```powershell
# Skip these cloning/setup commands if you've already done them successfully.
git clone https://github.com/safety-research/sleight-bench.git vendor/sleight-bench
git -C vendor/sleight-bench checkout 218c58315cc01ff0dc5a100e906c27d82d259521
uv sync --project vendor/sleight-bench --locked

# Download the benchmark's published decryption key from the upstream README.
uv run --project vendor/sleight-bench python vendor/sleight-bench/decrypt.py --key '<KEY FROM OFFICIAL README>'

# Free dry run (no API requests, no credentials necessary)
uv run --project vendor/sleight-bench python run_case.py --provider openai --trials 1
```

The authors deliberately encrypt data to avoid accidental inclusion in training corpora. Decrypt only on your machine. Do not publish, paste, or upload the benchmark transcript, even if someone asks for debug logs. This experiment necessarily sends the transcript to the selected provider's API when executed; verify your applicable terms and data controls before running. The OpenAI API requests set `store=False`.

### Run with OpenAI (recommended initial experiment)

Obtain a project key from the [OpenAI Platform](https://platform.openai.com/api-keys). In the current PowerShell session:

```powershell
$secureKey = Read-Host 'OpenAI API key' -AsSecureString
$env:OPENAI_API_KEY = (New-Object System.Net.NetworkCredential('', $secureKey)).Password
Remove-Variable secureKey

# Four billable API requests: 2 prompts x attack+benign, 1 trial each.
uv run --project vendor/sleight-bench python run_case.py --provider openai --model gpt-6.1-sol --reasoning-effort medium --trials 1 --execute

# Descriptive summary: no FPR estimate.
uv run --project vendor/sleight-bench python summarize.py results/openai-gpt-6.1-sol-t1.jsonl
```

To try a cheaper model, set `--model gpt-6-luna` and use the corresponding output file name. For a more capable (and more expensive) monitor, use `--model gpt-6-astra`.

### Run with Anthropic (optional, later)

```powershell
$secureKey = Read-Host 'Anthropic API key' -AsSecureString
$env:ANTHROPIC_API_KEY = (New-Object System.Net.NetworkCredential('', $secureKey)).Password
Remove-Variable secureKey

uv run --project vendor/sleight-bench python run_case.py --provider anthropic --trials 1 --execute
uv run --project vendor/sleight-bench python summarize.py results/anthropic-claude-opus-4-6-t1.jsonl
```

**No Anthropic key is required to run OpenAI, and no OpenAI key is required to run Anthropic.** API billing is separate from ChatGPT/Claude subscriptions.

## Results, resuming and safeguards

- Output is stored in **provider/model/trial-specific** JSONL files under `results/`, which is gitignored. Only numerical scores, metadata hashes, request status and token usage are written: **no raw prompts, transcript text, model reasoning, or API secrets**.
- A second execution with the same output file is blocked. If a request fails after earlier successful calls, fix the issue and re-run the same command adding `--resume`. This checks existing rows' provider/model/effort, hashes, and trial IDs and submits **only missing conditions**. Do not delete files that include valid scores.
- Increase trials to 10 only after interpreting the first pilot and considering cost. This produces **40 planned requests**; it still covers one underlying attack/benign pair.
- `--execute` is required to send any billable request. A dry run never requires a key.
- For an explicit new experiment, use `--output results/my-new-run.jsonl` rather than overwriting an existing result file.

## Tests (no credentials required)

```powershell
python -m unittest discover -s tests -v
```

## Limitations and next research stage

The upstream runner uses BigQuery and more elaborate retries; this small harness reproduces the authors' transcript stripping, prompt rendering, and numeric score parsing but is **not identical to their end-to-end infrastructure**. Different provider inference configurations are not equivalent, and one paired example cannot establish a population-level improvement or novel result. Any executable MCP authorization-transfer experiment is a separate future phase; the Manning server remains unchanged.
