# UmaxFormer Experiment Docs

이 디렉토리는 실험 설명과 실험군별 해석 기준을 관리합니다. 실행 명령은 [RUN_COMMANDS.md](../../RUN_COMMANDS.md)에서 관리합니다.

## Documents

| Document | Purpose |
| --- | --- |
| [experiment_overview.md](experiment_overview.md) | Config와 `outputs/` 상태에서 자동 생성되는 전체 실험 요약, status, ranking, genealogy. |
| [umaxformer_v1_family.md](umaxformer_v1_family.md) | SODA+ORAS5 validation pair를 사용하는 UmaxFormer V1 baseline 기준. |
| [umaxformer_v2_family.md](umaxformer_v2_family.md) | ORAS5-only validation을 사용하는 UmaxFormer V2 active line. |
| [model_design_order_sheet.md](model_design_order_sheet.md) | V1 baseline 재훈련과 V2 이후 실험 설계 순서. |

## Regenerate Overview

`experiment_overview.md`는 아래 명령으로 갱신합니다.

```bash
cd /home/sanggyu1008/project/UmaxFormer
python tools/docs/generate_experiment_overview.py
```
