# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

ENSO 연구 프로젝트로, Niño 3.4 지수의 lead-time 1–24 개월 예측 모델(주로 CNN-Transformer hybrid)을 학습/평가합니다. NetCDF 기후 자료(CMIP6, SODA/20CRv2, ORAS5/ERA5, GODAS, ERSSTv5)를 PyTorch로 처리합니다.

## Working directory

모든 명령은 프로젝트 루트에서 실행합니다. 런타임은 config의 상대경로를 프로젝트 루트 기준으로 해석합니다 ([src/utils/paths.py](src/utils/paths.py)).

```bash
cd /home/sanggyu1008/project/UmaxFormer
```

## Core commands

전체 파이프라인 명령 모음은 [RUN_COMMANDS.md](RUN_COMMANDS.md)에 있습니다. 활성 순서는 v1 → v2 입니다.

```bash
# Cache 생성만 (학습 전 사전 빌드)
python train_model.py --config <cfg> --build-cache-only

# 학습 (resume 가능: --resume latest 또는 체크포인트 경로)
python train_model.py --config <cfg>

# 평가 (split: validation | val | test)
python evaluate_model.py --config <cfg> --checkpoint <ckpt> --split <split>

# Figures
python make_figures.py --config <cfg> --split <split>

# Experiment overview 재생성 (config/output 추가·변경 후 반드시 실행)
python3 tools/docs/generate_experiment_overview.py
```

## Tests

`unittest` 기반. 단일 테스트 실행:

```bash
python -m unittest tests.test_hamcnn_project
python -m unittest tests.test_training_diagnostics.TestClassName.test_method
```

학습 진단 케이스가 필요할 때는 학습 스크립트에 `--diagnostics --diagnostic-epoch N --diagnostic-batch-index B --diagnostic-sample-index S` 플래그를 사용합니다.

## Architecture

### Config-driven pipeline

모든 실행은 YAML config 한 파일이 진입점입니다. 학습/평가/figure 스크립트는 동일 config를 공유합니다 — config의 `data`, `model`, `training` 섹션이 dataset, 모델 팩토리, 학습 루프를 모두 구동합니다. 경로 필드는 `resolve_config_paths` ([src/utils/paths.py](src/utils/paths.py))가 일괄적으로 프로젝트 루트 기준 절대경로로 변환합니다.

### Module layout

- [src/data/dataset.py](src/data/dataset.py): NetCDF input/target 로딩, validation pair, cache 빌드/로드, 정규화 통계. `build_train_val_test_datasets`가 전체 dataset 구성 진입점.
- [src/models/factory.py](src/models/factory.py): `MODEL_REGISTRY`와 `build_model(config)`. 현재 `umaxformer_v1`, `umaxformer_v2`(동일 클래스 alias), `hamcnn_project` 지원. config의 `model.name`이 dispatch 키.
- [src/utils/metrics.py](src/utils/metrics.py): lead-wise / equal-group / 요약 지표.
- [src/utils/paths.py](src/utils/paths.py): 경로 해석. 새 config 키에 경로가 들어가면 `resolve_config_paths`에 추가해야 합니다.
- [src/figures/figure_style.py](src/figures/figure_style.py): 공유 matplotlib 스타일.

루트의 `train_model.py`, `evaluate_model.py`, `make_figures.py`는 thin driver입니다. `tools/`의 스크립트는 분석·문서·ensemble·figure helper·maintenance 유틸이며 import 경로 확보를 위해 [tools/_bootstrap.py](tools/_bootstrap.py)를 통해 프로젝트 루트를 `sys.path`에 추가합니다.

### Data cache & namespaces

`data.cache_root`(기본 `data/cache/shared_dataset`)에 sample tensor cache가 저장되며, **`cache_namespace`로 데이터 라인을 분리**합니다:

- `10vars` — SODA+ORAS5 baseline cache (v1).
- `10vars_oras5` — ORAS5-only 10-variable cache (v2).
- `9vars_oras5_*` — ORAS5-only variable ablation.

Namespace를 잘못 매칭하면 서로 다른 validation 라인의 cache를 공유하게 되므로, 새 실험은 가장 가까운 기존 config 복사 + `experiment_name` / `output_dir` / `cache_namespace` 동시 갱신이 원칙입니다.

### Experiment lines

활성 라인은 두 개입니다 ([docs/experiments/](docs/experiments/)):

- **v1** (`umaxformer_v1`): SODA+ORAS5 dual validation pair, canonical restart baseline.
- **v2** (`umaxformer_v2`): ORAS5-only validation, seed 42 hybrid a2/b0.5 setting의 후계. **새 후속 실험은 v2를 베이스로** 생성하고 SODA를 validation에 넣지 않는 것이 기본 규칙입니다 (별도 baseline 연구를 새로 열 때만 예외).

`configs/archive/`와 `outputs/legacy_flat/`는 참조용 보관소로, active discovery 대상이 아닙니다. `2026-05-11` 기준 이전 산출물이 전면 재훈련을 위해 정리되어 있어, `RUN_COMMANDS.md` 순서를 그대로 따라 v1부터 재생성합니다.

### Output layout convention

`outputs/experiments/<line>/<experiment_name>/`에 `checkpoints/`, `predictions/`, `metrics/`, `logs/`, `figures/`가 생성됩니다. Config의 `output_dir`이 이 경로와 일치해야 overview 생성기가 인식합니다.

## Working rules

- Config / output 추가·삭제·재훈련·재평가 후에는 반드시 `python3 tools/docs/generate_experiment_overview.py`로 [docs/experiments/experiment_overview.md](docs/experiments/experiment_overview.md)를 재생성합니다.
- 새 모델을 추가할 때는 클래스 + [src/models/factory.py](src/models/factory.py)의 `MODEL_REGISTRY` 및 `build_model` dispatch 둘 다 갱신해야 합니다.
- 경로는 config에 항상 `data/...`, `outputs/...` 같은 프로젝트 루트 기준 상대경로로 작성합니다. 절대경로는 그대로 통과하지만 portability를 잃습니다.
- 폐기된 staged branch 결과를 active overview 문서에 다시 끌어오지 않습니다 ([docs/experiments/model_design_order_sheet.md](docs/experiments/model_design_order_sheet.md)).
