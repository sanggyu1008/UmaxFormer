# Config Layout

모든 실행 config는 프로젝트 루트 기준 상대경로를 사용합니다.

| Path | Role |
| --- | --- |
| `experiments/umaxformer_v1/` | SODA+ORAS5 validation pair를 사용하는 v1 baseline config. |
| `experiments/umaxformer_v2/` | ORAS5-only validation을 사용하는 v2 active config와 후속 실험 config. |
| `ensembles/` | Ensemble figure/evaluation helper config. |
| `archive/umaxformer_v1_legacy_oras5/` | v2로 승격하기 전의 seed-42 v1 ORAS5-only 실험 기록. Active discovery 대상은 아님. |

새 실험을 추가할 때는 가장 가까운 기존 config를 복사하고, `experiment_name`, `output_dir`, data cache namespace, model/training 설정을 함께 갱신합니다.

경로 필드는 `data/input/...`, `data/target/...`, `data/cache/...`, `outputs/...` 형식을 권장합니다. 런타임은 상대경로를 프로젝트 루트 기준으로 해석하고, 절대경로가 들어오면 그대로 사용합니다.

Cache namespace는 validation/data line을 드러내도록 관리합니다. `10vars`는 SODA+ORAS5 baseline cache, `10vars_oras5`는 ORAS5-only 10-variable cache, `9vars_oras5_*`는 ORAS5-only variable ablation cache에 사용합니다.

현재 active training order는 `umaxformer_v1`을 먼저 재훈련한 뒤 `umaxformer_v2`를 재훈련하는 순서입니다.
