# cain-complient-lie

Code and evaluation artifacts for *Compliant Lie* detection with Air-Gapped Cascade and Premise Divergence.

## Structure

- `src/` — Core library modules (verifiers, premise contracts, dataset generator, metrics, models, LLM cache).
- `scripts/` — Runnable experiment and analysis scripts (`run_experiments.py`, `cascade_and_slider.py`, `make_figures.py`, `probe_extractor.py`, `scan_nvidia_models.py`).
- `tests/` — Test suite for dataset, tiers, comparators, verifiers, and cache.
- `results/` — Cached experiment results, logs, and SQLite LLM cache.
- `figures/` — Generated publication figures (PDF and PNG).
- `paper/` — LaTeX source and compiled short paper.

## Requirements & Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

Core deterministic runtime is provided by [`dir_core==0.4.0`](https://pypi.org/project/dir-core/).

### Configuration (.env)

Copy `.env.sample` to `.env` and adjust settings:
```bash
cp .env.sample .env
```

Parameters in `.env`:
- `LLM_PROVIDER`: `nvidia` (default) or `ollama`.
- `LLM_MODEL`: Active model name (e.g. `nvidia/nemotron-3.5-lightning-30b-a3b`).
- `NVIDIA_API_KEY`: API key for NVIDIA NIM integrate API.
- `NVIDIA_INVOKE_URL`: Endpoint URL (default: `https://integrate.api.nvidia.com/v1/chat/completions`).
- `NVIDIA_TEMPERATURE`: Sampling temperature (default: `0.5`).
- `NVIDIA_MAX_TOKENS`: Max tokens to generate (default: `1024`).
- `NVIDIA_TOP_P`: Nucleus sampling top-p (default: `1.0`).
- `NVIDIA_SEED`: Random seed (default: `0`).
- `NVIDIA_ENABLE_THINKING`: Enable reasoning mode for thinking models (default: `false`).
- `NVIDIA_MAX_RETRIES`: Retries for transient errors (429/5xx/timeout); default `6`.
- `NVIDIA_BACKOFF_BASE_S` / `NVIDIA_BACKOFF_CAP_S`: Exponential backoff base and cap (default `1.0` / `30.0`).
- `NVIDIA_RESPECT_RETRY_AFTER`: Honor the `Retry-After` header on 429/503 (default `true`).
- `NVIDIA_ADAPTIVE_RATE`: AIMD throttle that adapts the request interval to 429s (default `true`).
- `NVIDIA_DISCOVERY_TIMEOUT_S` / `NVIDIA_DISCOVERY_SKIP_SPECIALIZED`: Model-scan probe timeout and whether to skip specialized models.
- `LLM_MAX_WORKERS` / `LLM_MIN_INTERVAL_S`: Concurrency and the minimum spacing between request starts.
- `OLLAMA_URL`: Local Ollama URL (default: `http://127.0.0.1:11434`).

### NVIDIA NIM usage limits

The hosted NVIDIA NIM API (`integrate.api.nvidia.com`) runs on a shared **free credit pool**
with a per-minute request rate limit. The exact numbers change over time and depend on the
account tier, so the client does **not** hard-code them — it reacts to the server's signals:

- **`429` (rate limited / quota)** — transient. The client retries, honoring the `Retry-After`
  header when present, and adaptively widens its request interval (AIMD) to back off.
- **`403` (forbidden)** — the account is not entitled to that model. Permanent; fails fast.
- **`410` (Gone / EOL)** — the model has been retired. Permanent; fails fast.
- **`404` (not found)** — unknown model id. Permanent; fails fast.
- **`5xx` / timeouts** — transient; retried with exponential backoff and jitter.

### Discovering active models

List the catalog and probe which chat models actually respond right now:

```bash
python scripts/scan_nvidia_models.py                 # skips embedding/audio/guard models
python scripts/scan_nvidia_models.py --no-skip-specialized --timeout 8
```

The report classifies every model as available / rate-limited (429) / EOL (410) /
not-found (404) / forbidden (403). Active probing consumes a tiny amount of credits
(one `max_tokens=1` request per model). The footer also reports how many probe
requests were sent and the effective requests-per-minute throughput of the scan.

### Measuring token & request usage

The NVIDIA API does not expose account quotas, so the client measures real usage locally:

- Per call: `CallStats` carries `prompt_tokens` / `completion_tokens` (actual usage reported by the API).
- Per client: `NvidiaClient.usage_stats()` returns a `UsageStats` snapshot with cumulative
  `prompt_tokens`, `completion_tokens`, `total_tokens`, `live_calls`, `cached_calls`,
  `http_requests` (all POSTs incl. retries — what the rate limit sees), `requests_last_minute`
  (trailing-60 s window) and `peak_requests_per_minute`.

`scripts/run_experiments.py` and `scripts/cascade_and_slider.py` print this summary at the end
of a run (tokens, HTTP requests, peak RPM), and `run_experiments.py` stores it in the run's JSON.

### Optimizing the availability pool

- **Adaptive rate limiting** (`NVIDIA_ADAPTIVE_RATE`) widens the request interval after a 429
  and relaxes it again after sustained successes, keeping throughput just under the limit.
- **`Retry-After` is honored** so retries resume exactly when the server allows.
- **SQLite response cache** means identical prompts never spend a second credit — reruns are free.
- Tune **`LLM_MAX_WORKERS`** and **`LLM_MIN_INTERVAL_S`** together to stay under the per-minute limit;
  use the measured `peak_requests_per_minute` to calibrate them.
- The model scan uses **minimal probes** and **skips specialized models** to avoid wasting credits.

## Running Tests

```bash
pytest
```

## Running Experiments & Generating Figures

```bash
# Generate figures from cached results
python scripts/make_figures.py

# Run benchmark evaluation
python scripts/run_experiments.py --fast

# Run cascade economics and financial slider
python scripts/cascade_and_slider.py --exp all
```