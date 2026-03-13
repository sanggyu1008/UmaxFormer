#!/usr/bin/env bash
set -euo pipefail

command -v cdo >/dev/null 2>&1 || { echo "[error] cdo not found"; exit 127; }
command -v ncrcat >/dev/null 2>&1 || { echo "[error] ncrcat not found"; exit 127; }
command -v ncks >/dev/null 2>&1 || { echo "[error] ncks not found"; exit 127; }

ROOT="/mnt/d/project/01_ENSO/01_data/01_raw/oras5"
TMP="${ROOT}/_tmp_surface_oras5"

mkdir -p "$TMP"

build_one() {
    local INDIR="$1"
    local STEM="$2"      # vozocrtx or vomecrty
    local OUTVAR="$3"    # uos or vos
    local WORK="${TMP}/${OUTVAR}"
    local SELDIR="${WORK}/selz"

    mkdir -p "$SELDIR"

    echo "=== building ${OUTVAR} from ${STEM} ==="

    shopt -s nullglob
    local files=( "${INDIR}/${STEM}"_control_monthly_highres_3D_*.nc )
    shopt -u nullglob

    if [ ${#files[@]} -eq 0 ]; then
        echo "[error] no input files found for ${STEM} in ${INDIR}"
        exit 1
    fi

    # 1) 표층 level만 선택하되, 아직 reduce_dim 하지 않음
    for f in "${files[@]}"; do
        b=$(basename "$f")
        out="${SELDIR}/${b}"

        cdo -L -O sellevidx,1 "$f" "$out"

        # 2) ncrcat이 확실히 시간축으로 이어붙일 수 있게 record dimension 보정
        ncks -O --mk_rec_dmn time_counter "$out" "$out"
    done

    # 3) 시간 병합
    ncrcat -O "${SELDIR}"/*.nc "${WORK}/${STEM}_surface_3d_1958-1978.nc"

    # 4) 병합 후 singleton depth 차원 제거
    cdo -L -O --reduce_dim copy \
        "${WORK}/${STEM}_surface_3d_1958-1978.nc" \
        "${WORK}/${STEM}_surface_2d_1958-1978.nc"

    # 5) 변수명 변경
    cdo -L -O chname,${STEM},${OUTVAR} \
        "${WORK}/${STEM}_surface_2d_1958-1978.nc" \
        "${ROOT}/${OUTVAR}.1958-1978.nc"

    echo "=== check: ${OUTVAR} ==="
    cdo showname "${ROOT}/${OUTVAR}.1958-1978.nc"
    cdo ntime   "${ROOT}/${OUTVAR}.1958-1978.nc"
    ncdump -h   "${ROOT}/${OUTVAR}.1958-1978.nc" | sed -n '1,40p'
    echo
}

build_one "${ROOT}/zonal_velocity"      "vozocrtx" "uos"
build_one "${ROOT}/meridional_velocity" "vomecrty" "vos"

echo "[done]"