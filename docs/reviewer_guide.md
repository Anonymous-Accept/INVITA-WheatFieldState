# Reviewer Guide

This package is a code-only review artifact for **INVITA-WheatFieldState: A Real-World Benchmark for Crop-State Estimation in Wheat Field Trials**. It is designed to let reviewers inspect the benchmark implementation, leakage controls, representation boundaries, and reproducibility commands without bundling governed data or generated results.

## Release Boundary

Included:

- Benchmark dataloaders, split validation logic, baseline predictors, neural representation implementations, fusion diagnostics, and QA-generation utilities.
- Experiment runners for all-example evaluation, required-modality subset diagnostics, same-row fusion, neural diagnostics, and VLM QA generation. One raster-geometry route is not included because validated plot-geometry raster extraction is outside the code-only release boundary.
- Package-level checks for anonymity and repository hygiene.
- Static project page source.

Excluded:

- Raw field-trial assets and source ledgers.
- Derived target tables, available-observation indexes, official split CSVs, feature stores, payload databases, QA JSONL files, predictions, metrics, coverage summaries, run configs, training summaries, model checkpoints, and logs.

Reproduction requires the governed derived release outside the repository.
The project page source in this repository points reviewers to the code and
data-release instructions; the derived benchmark artifacts themselves are not
stored in Git. Point the code to that release with `INVITA_DATA_ROOT` and
`INVITA_SPLIT_ROOT`.

## Tasks

The benchmark uses four plot-date crop-state target families.

| Target family | Crop-state target | Plot-date examples | Evaluation status |
| --- | --- | ---: | --- |
| NDVI | Canopy greenness | 181,555 | all-example and required-modality subset diagnostics |
| LAI | Canopy amount product/proxy family | 17,816 | all-example and selected diagnostics |
| FCover | Canopy closure product/proxy family | 24,612 | all-example and selected diagnostics |
| Growth stage | Zadoks growth stage / phenological timing | 3,420 | all-example and selected diagnostics |

LAI and FCover are provenance-labeled product/proxy target families. Aggregate scores are useful benchmark summaries, but scientific interpretation should use target-name and provenance slices.

## Canonical Split

The governed release supplies the canonical plot-disjoint split. No plot appears in more than one split, and no plot-date target appears in more than one split. Trials remain shared context, so this evaluates held-out plots within the archive rather than transfer to entirely unseen trials, sites, or years.

## Evaluation Surfaces

All-example regression covers Source-date prior, Crop-date prior, Tabular metadata model, Observation-availability model, and neural representation diagnostics.

Required-modality subset diagnostics cover sensor-history, field-camera feature, neural sensor-sequence, and field-camera image-set routes on examples where the required pre-target evidence is available.

Same-row fusion evaluates linear and gated stackers only on shared examples where component predictions exist.

Fusion is evaluated only on shared examples where component predictions exist. It must be compared with the strongest component on the same examples, not with full-test metrics.

VLM QA diagnostics:

- QA generation code is included, but generated QA JSONL files are not included in this code-only package.
- These probes are interaction diagnostics, not crop-state regression routes.


## Runner Map

| Paper term | Code entry point |
| --- | --- |
| Source-date prior | `experiments/runs/run_source_date_prior.py` |
| Crop-date prior | `experiments/runs/run_crop_date_prior.py` |
| Tabular metadata model | `experiments/runs/run_tabular_metadata_model.py` |
| Observation-availability model | `experiments/runs/run_observation_availability_model.py` |
| Sensor-summary model | `experiments/runs/run_sensor_summary_model.py` |
| Frozen image-feature model | `experiments/runs/run_frozen_image_feature_model.py` |
| Sensor-sequence Transformer | `experiments/runs/run_sensor_sequence_transformer.py` |
| Linear stacker | `experiments/runs/run_linear_stacker.py` |
| Tabular Transformer | `experiments/runs/run_tabular_transformer.py` |
| Observation-set Transformer | `experiments/runs/run_observation_set_transformer.py` |
| Sensor-sequence TCN | `experiments/runs/run_sensor_sequence_tcn.py` |
| Field-camera image-set model | `experiments/runs/run_field_camera_image_set_model.py` |
| Gated stacker | `experiments/runs/run_gated_stacker.py` |

## Full Reproduction Commands

Use the governed derived release:

```bash
export INVITA_DATA_ROOT=<INVITA_DATA_ROOT>
export INVITA_SPLIT_ROOT=<INVITA_DATA_ROOT>/splits/<CANONICAL_SPLIT>
export INVITA_OUTPUT_ROOT=outputs
export INVITA_RUN_ID=plot_disjoint
```

Run all-example baselines:

```bash
python3 experiments/runs/<all-example-runner>.py --run-id "$INVITA_RUN_ID"
```

Run required-modality subset diagnostics. The Sensor-sequence Transformer uses a fixed canonical run id because the experiment matrix reads that route explicitly.

```bash
python3 experiments/runs/<required-modality-runner>.py --run-id "$INVITA_RUN_ID"
python3 experiments/runs/run_sensor_sequence_transformer.py \
  --run-id plot_disjoint_sensor_sequence_transformer
```

Run Linear stacker same-row fusion with the canonical summary run id:

```bash
python3 experiments/runs/run_linear_stacker.py \
  --run-id plot_disjoint_linear_stacker \
  --route-root "<route-name>=outputs/<route-output-dir>/${INVITA_RUN_ID}"
```

Run neural representations and summaries:

```bash
python3 experiments/runs/<neural-runner>.py --run-id "$INVITA_RUN_ID"
python3 experiments/runs/summarize_neural_representations.py
```

Summarize all-example and Linear stacker outputs:

```bash
python3 experiments/runs/summarize_experiment_matrix.py
```

Concrete runner filenames are intentionally kept in `experiments/runs/` rather than repeated throughout the public-facing documentation, so reviewers can inspect the implementation without mixing code identifiers into the paper-level terminology.

Smoke-test limits may be used for local debugging, but any run using row limits is not a canonical benchmark run.

## Package Checks

Before review upload or commit:

```bash
PYTHONPYCACHEPREFIX="$(mktemp -d)" python3 -m compileall src experiments/runs scripts
python3 scripts/audit_anonymity.py .
python3 scripts/validate_reviewer_package.py .
```

The validation script fails if data directories, result artifacts, logs, hidden scratch files, or generated table files are present in the repository.
