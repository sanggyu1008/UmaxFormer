# ENSO Data Layout

이 디렉토리는 학습과 평가에 사용하는 데이터와 cache를 관리합니다.

| Path | Role |
| --- | --- |
| `raw/` | CMIP6, ERA5, GODAS, ORAS5 등 원천 기후 자료. |
| `interim/` | 전처리 중간 산출물. |
| `input/` | 학습/검증/테스트 입력 NetCDF. |
| `target/` | Niño 3.4 target NetCDF. |
| `cache/` | 학습용 tensor cache와 공유 dataset cache. |

Config는 프로젝트 루트 기준 상대경로(`data/input/...`, `data/target/...`, `data/cache/...`)를 사용합니다.
