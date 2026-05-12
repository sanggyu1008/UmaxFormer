# Experiment Overview

Active experiment configs live under `configs/experiments/` and are run from the project root.

| Experiment | Run | Model | Validation | Cache | Output | Config |
| --- | --- | --- | --- | --- | --- | --- |
| umaxformer_v1 | umaxformer_v1 | umaxformer_v1 | validation_soda_20crv2_1871_1978, validation_oras5_era5_1958_1978 | 10vars | `outputs/experiments/umaxformer_v1/umaxformer_v1` | `configs/experiments/umaxformer_v1/umaxformer_v1.yaml` |
| umaxformer_v2 | umaxformer_v2 | umaxformer_v2 | validation_oras5_era5_1958_1978 | 10vars_oras5 | `outputs/experiments/umaxformer_v2/umaxformer_v2` | `configs/experiments/umaxformer_v2/umaxformer_v2.yaml` |

Archived configs under `configs/archive/` are intentionally excluded from this table.
