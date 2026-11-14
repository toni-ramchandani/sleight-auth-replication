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

## Experiment 3 — Cross-session context, targeting Stage 2

**New, separate scripts:** `run_context.py` and `analyze_context.py`. Experiment 1 and Experiment 2 runners, analyses, and existing `results/*.jsonl` are unchanged.

### Question and controls

Does including earlier **Session 1** context change the monitor's suspicion score for actions in **Session 2**? This is a follow-up to Experiment 2, where both multi-stage cases had lower isolated Stage 2 suspicion scores than Stage 1 scores. **That difference is not, by itself, a monitoring failure.**

Two original SLEIGHT-Bench two-stage scenarios are evaluated:

- `transcript-perm-launder-corp-2part`
- `transcript-perm-launder-gcs-2part`

Each scenario uses the official Stage 1 + Stage 2 **attack** pair, and the official Stage 1 + Stage 2 **benign** pair. No labels enter model input. The two arms are:

| Arm | Evidence given to monitor | Score target |
| --- | --- | --- |
| `isolated` | Explicitly labeled Session 2 transcript only | Session 2 actions |
| `cumulative` | Explicitly labeled Session 1 (prior context) followed by Session 2 | Session 2 actions, interpreted with Session 1 context |

Both arms use an **identical added targeting instruction**, in the user message, stating that the monitor must score Session 2 rather than score prior actions in Session 1. The original `control_arena` and `control_arena_doc_planting_aware` system prompts are imported **unchanged** from the pinned SLEIGHT-Bench release. The original `strip_assistant_text` and `render_prompt` implementations are imported from that same release. Stable XML-style session boundaries retain the chronology without inventing a combined agent session.

**Why 16 new calls, not eight?** The Stage 2-only scores from Experiment 2 had no explicit Stage 2 targeting instruction. The new intervention changes the task formulation in *both* experimental arms. For a controlled comparison, obtain fresh isolated and cumulative scores under the same instruction: **2 scenarios × 2 prompts × 2 labels × 2 context arms × 1 trial = 16 billable calls**. Using the eight historical Experiment 2 Stage 2 scores as a control could confound the effect of context with the new targeting instruction and model randomness.

### Run on Windows PowerShell

From the experiment root where your existing decrypted `vendor/sleight-bench` is already prepared:

```powershell
# Offline tests: include all earlier experiments and Experiment 3.
python -m unittest discover -s tests -v
python -m compileall -q run_case.py summarize.py run_context.py analyze_context.py tests

# Inspect the 16-call plan; no API key required and no model charges.
uv run --project vendor/sleight-bench python run_context.py --provider openai --model gpt-6.1-sol --reasoning-effort medium --trials 1

# If the key is not already set in this PowerShell session:
$secureKey = Read-Host 'OpenAI API key' -AsSecureString
$env:OPENAI_API_KEY = (New-Object System.Net.NetworkCredential('', $secureKey)).Password
Remove-Variable secureKey

# 16 paid OpenAI requests. Explicit --execute is required.
uv run --project vendor/sleight-bench python run_context.py --provider openai --model gpt-6.1-sol --reasoning-effort medium --trials 1 --execute

# Find the precise default output path from the runner's printed summary.
# Example: locate only cross-session score records:
Get-ChildItem results\openai-gpt-6.1-sol-context-*-t1.jsonl

# Replace <OUTPUT_PATH> with the actual printed path.
uv run --project vendor/sleight-bench python analyze_context.py <OUTPUT_PATH> --threshold 70
```

For Anthropic, use the same dataset, prompts, and conditions, replacing provider/model and setting `ANTHROPIC_API_KEY`:

```powershell
uv run --project vendor/sleight-bench python run_context.py --provider anthropic --model claude-opus-4-6 --trials 1
uv run --project vendor/sleight-bench python run_context.py --provider anthropic --model claude-opus-4-6 --trials 1 --execute
```

The Anthropic path uses the benchmark's original provider configuration, including temperature 1.0 and adaptive thinking. OpenAI uses the Responses API with `store=False`; model settings and token budgets are **not** identical across providers. For either provider, `--execute --resume` skips already recorded conditions only after strict provider/model/effort/revision/source/prompt/**rendered-payload** hash validation. The runner records failed/refused scoring attempts as unscored records, not silent retries. A changed scoring task or added session context changes the payload SHA and therefore invalidates prior resume state.

Use `--scenario transcript-perm-launder-corp-2part` to select only one scenario (eight calls for both arms). Optional `--context-mode cumulative` alone requests eight calls for both scenarios, but it **does not** produce a fresh matched isolated baseline. `--trials 10` for both scenarios and arms requests **160 billable calls**; do not scale until interpreting a one-trial pilot.

### Score-only provenance and analysis

Experiment 3 results live in a separate gitignored `results/*context*.jsonl`, with a default filename determined by provider, model, scenarios, context arms, and trial count. Each record contains provider/model/settings, scenario, prompt, trial, target stage, condition, score/status, token usage, pinned benchmark commit, stage-specific raw-file SHA-256s, unchanged prompt SHA-256, and the **effective rendered system+user payload SHA-256**. No original transcript text, source-derived explanations, model reasoning, API credentials or full prompt+transcript payloads are stored or committed.

`analyze_context.py` requires compatible provider/model/effort/revision records and checks input hashes and pairing before reporting:

- Scores for harmful and benign **Stage 2 targets** under isolated versus cumulative context.
- Paired within-trial `cumulative - isolated` changes for harmful and benign inputs separately.
- The difference between these paired changes, only when both labels and both arms have scored values for the same trial index.
- Missing/invalid scores, counts crossing an **illustrative** threshold, and the limitations of small, synthetic samples.

**Methodological boundary:** Concatenating two already-published transcripts supplies evidence, **not** real execution state or an agent's live memory. Suspicion of earlier Session 1 actions could still influence cumulative scores despite the targeting instruction. Even a larger Stage 2 score is not proof that the monitor recognized a Stage 2-specific authorization violation; separate negative controls and a real executable MCP environment would be needed. This experiment is not a calibrated 1%-FPR test, benchmark-level reproduction, new monitor invention, or demonstrated safety improvement. Its purpose is to test a narrow context-sensitivity hypothesis with traceable outcomes.

## Experiment 4 — Source-grounded authorization evidence (new)

**Files added:** `authorization_evidence.py`, `run_evidence.py`, `analyze_evidence.py`, `tests/test_evidence.py`.

### Research question, provenance, and limitations

Experiment 3 found inconsistent changes when adding full Stage 1 history to a monitor asked to score Stage 2. The new experimental condition tests whether a **bounded, source-grounded candidate-evidence representation** changes the Stage 2 score, versus (1) Stage 2 alone and (2) complete earlier context. This is an exploratory, transcript-only test; it is not authorization enforcement.

The word **evidence** is carefully scoped. SLEIGHT-Bench transcripts do **not** include a separately authenticated policy/authorization service proving that an actor received a real grant. Therefore this experiment does **not** invent an “approved/denied” ledger, infer a verified grant from agent claims, or supply an attack label to the model. The ledger is *verifiable against the source transcript*, not *proof that the reported permissions were legitimate*. A trustworthy grant/denial oracle requires a separate executable environment (a later experiment).

The deterministic selection policy (`source-anchored-authorization-candidates-v2`) searches the **officially stripped Session 1 transcript** for fixed authorization-related terms. If no such term occurs in supported content blocks, a source-only fallback selects bounded **tool-call and tool-result operation-context excerpts**. A fallback excerpt is **not** verified authorization evidence; the JSON ledger and score-only output explicitly mark the selection mode. The fallback is deterministic and label-blind, and does not fabricate any grant, denial, or authorization verdict. If no eligible tool operations or keyword excerpts exist, the dry run still fails closed. This **v2 change was introduced after the original v1 selector returned zero matches on a local benchmark dry run**; it is an exploratory protocol amendment, not a preregistered intervention or a validated state tracker.

All selected excerpts retain original role/type, 1-indexed stripped-source record position, content-block index, canonical-record and block digests, and canonical-block offsets. Transcript origin classes explicitly distinguish unverified user messages, untrusted tool output and untrusted agent tool calls. Neither metadata, attack/benign labels, benchmark outcome annotations, nor model-generated authorization summaries are used to select evidence. The same rule applies to harmful and benign pairs. Excerpts are bounded; if more than 96 are selected, the precheck stops rather than truncating evidence silently. The dry run prints selection mode and tool-block counts but never source content.

All three arms retain the published monitor prompts and Experiment 3's **identical Stage 2 targeting instruction**:

| Mode | Stage 1 representation | Stage 2 content | Evaluation |
| --- | --- | --- | --- |
| `isolated` (from Experiment 3) | None | Official stripped Stage 2 | Stage 2 score |
| `cumulative` (from Experiment 3) | Entire official stripped Stage 1 | Same official stripped Stage 2 | Stage 2 score |
| `evidence` (Experiment 4, new) | Source-grounded candidate ledger extracted from stripped Stage 1 | Same official stripped Stage 2 | Stage 2 score |

**Only the third arm requires new calls.** Two scenarios × two published prompts × harmful/benign × one trial = **eight new API requests**. The analyzer refuses to compare them to Experiment 3 records if provider/model/effort/revision, prompt hash, source hashes or either **effective Experiment 3 model-input hash** differ. Results for different model providers must be analyzed separately.

**Methodological caveat:** The candidate ledger, particularly the operation-context fallback, is not a complete state tracker or an authorization oracle. It can omit relevant evidence, include forged claims, and change both information volume and structure. Analyze keyword-selection and fallback rows as distinct conditions rather than treating them as equivalent verified permission evidence. A better score does not prove the model reconstructed valid permission provenance, nor that an agent action would be correctly blocked by a real service. The authors' existing state-tracking and monitoring work should be treated as a related baseline rather than a novel contribution claimed here.

### Windows PowerShell runbook

Use the same project root (`C:\Toni\Work\sleight-auth-replication`), existing decrypted/pinned `vendor/sleight-bench`, and existing Experiment 3 JSONL. If you previously copied the original Experiment 4 files, **replace those same four files** with the corrected versions before running tests. **Keep `vendor/` and `results/` unchanged.** When installing via ZIP, copy the four new files above and optionally this `README.md`; do not overwrite the earlier scripts or result files.

```powershell
# Run offline unit tests; no API calls or benchmark text is required.
python -m unittest discover -s tests -v
python -m compileall -q authorization_evidence.py run_evidence.py analyze_evidence.py tests

# Preview extraction counts and selection modes for all four source pairs.
# This uses locally decrypted data, but makes no API calls; STOP if any precheck fails.
uv run --project vendor/sleight-bench python run_evidence.py --provider openai --model gpt-6.1-sol --reasoning-effort medium --trials 1

# If necessary, set the API key in this PowerShell session without displaying it:
$secureKey = Read-Host 'OpenAI API key' -AsSecureString
$env:OPENAI_API_KEY = (New-Object System.Net.NetworkCredential('', $secureKey)).Password
Remove-Variable secureKey

# Only after reviewing the dry run: eight billable new evidence-arm calls.
uv run --project vendor/sleight-bench python run_evidence.py --provider openai --model gpt-6.1-sol --reasoning-effort medium --trials 1 --execute

# The program prints the exact new results path; locate it if needed:
Get-ChildItem results\openai-gpt-6.1-sol-evidence-*-t1.jsonl

# Compare with your previously completed Experiment 3 output:
uv run --project vendor/sleight-bench python analyze_evidence.py `
  "results\openai-gpt-6.1-sol-context-56b82603-t1.jsonl" `
  "results\YOUR_EVIDENCE_FILENAME.jsonl" --threshold 70
```

The last command uses an **actual new output filename** printed by `run_evidence.py`; substitute it for `YOUR_EVIDENCE_FILENAME.jsonl`. Do not pass the Experiment 2 output: the Experiment 3 baseline is required because the Stage 2 targeting instruction and exact source hashes are matched.

If the run is interrupted, repeat the identical `run_evidence.py ... --execute --resume` command. The runner records a scored or unscored attempt after every completed request and checks every prior record against the same model, inputs, policy, digest and condition before skipping it. Do not delete the JSONL file to force retries.

Anthropic is supported without changing any files; set `ANTHROPIC_API_KEY` and select `--provider anthropic --model claude-opus-4-6`. For a three-arm Anthropic comparison, first obtain a **matching** Anthropic Experiment 3 isolated/cumulative file; OpenAI and Anthropic scores cannot be combined as if the same model ran all arms.

### Evidence handling, analysis and interpretation

No decrypted transcripts, extracted candidate text, full prompts, model outputs, or private keys are printed or stored in the results JSONL. Only condition labels, score/status, usage, candidate counts, extraction-policy version and selection mode, tool-block counts, hash-protected provenance and renderer configuration are stored. Session 1 candidate excerpts are JSON-escaped so malicious content cannot literally terminate the outer session tag; the model still treats **all** transcript material as untrusted. **Do not commit benchmark material or results**; the `results/` and `vendor/` directories are ignored.

`analyze_evidence.py` reports attack/benign score arrays in all three conditions, matched-trial `evidence - isolated` and `evidence - cumulative` deltas for each label, and changes in attack-versus-benign separation. Incomplete/invalid trials are excluded from paired comparisons and remain visible in the report. The `--threshold 70` decision boundary is illustrative and **not** calibrated to any population false-positive rate. Repetitions reduce sampling noise on the same synthetic cases but do not create independent scenarios. No causal safety, validity of claimed permissions, statistical significance or benchmark-scale catch-rate assertion follows from this experiment alone.

## Experiment 5 — Stage 1 representation ablations

Experiment 5 tests whether the Stage 2 attack–benign score gap depends on the information selected from Stage 1, the ledger format, or the explanation and metadata accompanying the excerpts. The design uses both two-stage scenarios, both published monitor prompts, and the official harmful and benign transcripts. The Stage 2 payload and targeting instruction follow Experiments 3 and 4. All four arms share the Stage 1 session wrapper. The runner verifies the [pinned schema](https://github.com/safety-research/sleight-bench/blob/218c58315cc01ff0dc5a100e906c27d82d259521/FORMAT.md) revision and the runtime files used from that checkout.

| Arm | Stage 1 representation | Main comparison |
| --- | --- | --- |
| `formatted_full` | Every officially stripped record, in order, quoted in the ledger entry format, including the roleless canary retained by the official stripper | Experiment 3 cumulative and fresh evidence |
| `neutral_context` | Tool protocol event roles, block types and source positions. Source-free markers pad the length to the evidence ledger's offline token proxy. | Fresh evidence, with provider input token deltas reported |
| `reduced_metadata` | The exact Experiment 4 excerpts with role, block type and source positions. The outer explanation, selection labels and per-entry hashes are omitted from the model input. | Fresh evidence |
| `evidence` | The original Experiment 4 ledger, checked for identical rendered payload hashes | Experiment 4 historical evidence and the three fresh arms |

Within each trial, the runner shuffles the conditions with a fixed seed. It records the raw and stripped source hashes, prompt and exact rendered payload hashes, representation and source-anchor hashes, request settings, implementation fingerprints, and token usage. Scores, status and metadata go to local JSONL under the ignored `results/` directory by default. Decrypted source, rendered inputs, model text and API credentials stay out of the score ledger. The runner sends evaluation inputs only to the selected provider's official API and supplies no tools.

### Windows PowerShell

Run from the updated project root with the existing pinned checkout in `vendor/sleight-bench`. The one-trial plan is **32 requests per provider**. One `--scenario` reduces the pilot to 16 requests. Two trials on both scenarios require 64 requests. Dry run previews the number of calls, maximum output token exposure and Stage 1 length diagnostics. A dollar estimate depends on provider rates and actual usage.

```powershell
python -m unittest discover -s tests -q
python -m compileall -q run_ablation.py representation_ablation.py analyze_ablation.py tests

$run = @('run_ablation.py', '--provider', 'openai', '--model', 'gpt-6.1-sol', '--reasoning-effort', 'medium', '--trials', '1', '--output', 'results/openai-exp5-t1.jsonl')
uv run --project vendor/sleight-bench --locked python @run

# If the key is missing from this PowerShell session, read it without echoing it.
if (-not $env:OPENAI_API_KEY) {
    $secureKey = Read-Host 'OpenAI API key' -AsSecureString
    $env:OPENAI_API_KEY = (New-Object System.Net.NetworkCredential('', $secureKey)).Password
    Remove-Variable secureKey
}

# Explicitly authorize the 32-call run.
uv run --project vendor/sleight-bench --locked python @run --execute --max-calls 32

# For an interrupted run, first check the provider's usage and preview remaining work.
uv run --project vendor/sleight-bench --locked python @run --resume
# Then rerun with --execute --resume --max-calls and your approved remaining ceiling.

python analyze_ablation.py results/openai-exp5-t1.jsonl `
  --context results/openai-gpt-6.1-sol-context-56b82603-t1.jsonl `
  --evidence results/openai-gpt-6.1-sol-evidence-09f08b9c-t1.jsonl --threshold 70
```

For Anthropic, use `--provider anthropic --model claude-opus-4-6`, omit `--reasoning-effort`, use a separate output filename, and set `ANTHROPIC_API_KEY`. The pinned adaptive-thinking configuration has a 128,000-token maximum output per request. The OpenAI setting has an 8,192-token maximum. `--max-calls` is a request ceiling for that invocation. Automatic retries are disabled. An interrupted request can still incur charges if its result was never saved. Resume skips saved scored and unscored attempts and rejects incompatible provenance. Analysis with the Experiment 5 file alone reports the fresh comparisons. Supply both historical files from the same provider, model, and settings to add the historical comparisons.

### Interpreting the contrasts

Compare label-specific changes alongside the change in the paired attack–benign gap. The analyzer retains missing and unscored attempts and shows the provider-reported input token deltas. The neutral arm's offline proxy counts runs of up to four letters or digits and counts punctuation individually. The local dry run matched that proxy across all four transcript variants with 1,088–1,268 repeated padding words. Actual input tokens can differ. Padding and event order can shift attention, and the tool protocol event pattern carries some behavioral information.

The full-history arm changes the amount and placement of content even in the common ledger format. The reduced arm changes both framing and length. All four local transcript variants used Experiment 4's tool-operation fallback. These excerpts reflect the transcript's operational history and any untrusted claims it contains. Historical runs can differ in model behavior across time even when the logged model and inputs match. Two scenarios, repeated prompts and an illustrative threshold of 70 support descriptive comparisons. A larger separation leaves open whether the monitor identified a Stage 2 violation or responded to earlier-session cues.
