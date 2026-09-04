# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

All commands are run via `./scripts/actions.sh` using `uv` as the package manager.

**Setup:**
```bash
./scripts/actions.sh sync          # Install dependencies (auto-detects GPU/CPU)
```

**Linting:**
```bash
./scripts/actions.sh lint          # Format and auto-fix with ruff
./scripts/actions.sh lint-check    # Check-only (ruff format, ruff check, pylint) — used in CI
./scripts/actions.sh toml-check    # Validate TOML files
```

**Type checking:**
```bash
./scripts/actions.sh type-check    # Run pyrefly on packages/
```

**Testing:**
```bash
./scripts/actions.sh unit-test                     # pytest on src/ (CPU, no GPU required)
./scripts/actions.sh integration-test-single       # integration_tests/small1_test.py (GPU)
./scripts/actions.sh integration-test-jepa         # integration_tests/jepa1_test.py (GPU)
./scripts/actions.sh integration-test              # integration_tests/small_multi_stream_test.py (GPU)
./scripts/actions.sh integration-test-all          # All three integration tests
```

To run a single unit test file directly:
```bash
uv run pytest src/weathergen/datasets/utils_test.py
```

**Training:**
```bash
uv run train --config config/config_mae.yml       # Train with a specific config
```

## Architecture

WeatherGenerator is a distributed deep learning model for Earth system simulation, trained on atmospheric reanalyses, forecasts, and observations.

### Key concepts

- **Learning paradigms**: The model supports MAE (Masked AutoEncoder), JEPA (Joint-Embedding Predictive Architecture), and autoregressive forecasting. These are selected via config files in `config/`.
- **Data streams**: Multiple heterogeneous data sources (ERA5 reanalysis, FESOM ocean model, observational data) are combined via a multi-stream sampler. Each stream is configured in `config/streams/`.
- **Distributed training**: Uses PyTorch FSDP (Fully Sharded Data Parallel) for GPU cluster training. DDP initialization and sharding are handled in `src/weathergen/run_train.py` and `src/weathergen/model/model_interface.py`.

### Package structure

- **`src/weathergen/`** — Core training package:
  - `run_train.py` — CLI entry point; supports `train`, `train_continue`, `inference` stages
  - `model/` — Transformer architecture (attention, encoder, embeddings, RoPE-2D positional encoding, EMA teacher)
  - `datasets/` — Data loading, tokenization, and masking strategies
  - `train/` — Training loop (`trainer.py`), loss calculation, LR scheduling, collapse monitoring
  - `utils/` — Distributed utilities, MLflow logging, metrics, CLI parsing

- **`packages/`** — Reusable workspace packages:
  - `common/` — Configuration management (OmegaConf-based), logging, I/O abstractions, HPC environment detection
  - `evaluate/` — Post-inference diagnostics; reads zarr/GRIB/netCDF without format conversion
  - `metrics/` — MLflow experiment tracking utilities
  - `readers_extra/` — Additional data source readers

### Configuration system

Configs are layered and merged with OmegaConf: private/HPC config → `config/default_config.yml` → user config file(s) → CLI overrides. All training hyperparameters (model size, attention type, loss weights, LR schedule, data streams, hardware settings) are in `config/default_config.yml`, with named presets in `config/config_*.yml`.

### Code style

- Line length: 100 characters
- Formatter: ruff (black-compatible)
- Linter: ruff + pylint
- Python 3.12 only (enforced via `.python-version`)

### Branching

- `main` — stable, used for running experiments
- `develop` — latest features, fast-evolving; PRs target this branch

Every PR must be linked to a GitHub issue (enforced in CI via `scripts/check_gh_issue.py`).

## Branch memory

Each branch has a `.claude/branch_memory.md` file that is tracked by git. Because it lives in the repo, checking out a branch gives you that branch's memory automatically.

**Rules for maintaining branch memory:**
- Read `.claude/branch_memory.md` at the start of every conversation.
- Update it whenever you learn something non-obvious about this branch: config choices and why they were made, key code decisions, dataset-specific quirks, known issues, or experiments in progress.
- Keep entries short — one or two sentences each. The file should stay under ~60 lines.
- Do not duplicate things already in CLAUDE.md or obviously derivable from the code.
- If the file doesn't exist yet on the current branch, create it.

**Sections to use (add only what is relevant):**
```markdown
## Goal
One-line description of what this branch is trying to achieve.

## Config
Key config choices and the reason behind them.

## Code changes
Non-obvious modifications and why they were made.

## Data / datasets
Dataset-specific knowledge relevant to this branch.

## Notes
Anything else worth remembering.
```
