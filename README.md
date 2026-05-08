# INVITA-WheatFieldState

Anonymous reviewer code package for the paper **INVITA-WheatFieldState: A Real-World Benchmark for Crop-State Estimation in Wheat Field Trials**.

This repository contains code, lightweight configuration, reviewer documentation, and a static project page source. It intentionally does **not** contain raw data, derived benchmark tables, prediction files, metrics files, run logs, or trained artifacts.

## What Is Included

- Baseline and neural benchmark code under `src/benchmarks/`.
- Experiment runners under `experiments/runs/`.
- Data-loading and QA-generation utilities under `src/data_processing/`.
- Reviewer package checks under `scripts/`.
- A static project page in `index.html` and `assets/`.

## What Is Not Included

- Raw field-trial records, imagery, sensor payloads, private source ledgers, or exact private locations.
- Derived release tables such as target tables, available-observation indexes, split CSVs, feature stores, and QA JSONL files.
- Machine-readable result artifacts such as predictions, metrics, coverage tables, run configs, summaries, or logs.

The governed derived release should be placed outside this repository and supplied through environment variables when reproducing experiments.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Set the release paths:

```bash
export INVITA_DATA_ROOT=<INVITA_DATA_ROOT>
export INVITA_SPLIT_ROOT=<INVITA_DATA_ROOT>/splits/<CANONICAL_SPLIT>
export INVITA_OUTPUT_ROOT=outputs
```

## Canonical Full-Task Runs

The governed release supplies the canonical plot-disjoint split for the four crop-state target families: NDVI, LAI, FCover, and growth stage. The experiment runners default to the full task set when `INVITA_DATA_ROOT` and `INVITA_SPLIT_ROOT` are configured.

Use `docs/reviewer_guide.md` for the full-dataset reproduction commands and interpretation notes. Any row-limit flags in neural runners are smoke-test controls only and are not part of the canonical full-dataset setting.

## Validation

```bash
PYTHONPYCACHEPREFIX="$(mktemp -d)" python3 -m compileall src experiments/runs scripts
python3 scripts/audit_anonymity.py .
python3 scripts/validate_reviewer_package.py .
python3 -m pytest src/benchmarks/baselines/
```

## Reviewer Notes

See `docs/reviewer_guide.md` for the release boundary, task definitions, experiment surfaces, and interpretation rules.
