#!/usr/bin/env bash
set -euo pipefail

command -v cdo >/dev/null 2>&1 || {
  echo "[error] cdo not found in PATH" >&2
  exit 127
}

# ============================================================
# 사용자 설정
# ============================================================
INROOT="/mnt/d/project/01_ENSO/01_data/01_raw/cmip6/ssp370"
OUTROOT="/mnt/d/project/01_ENSO/01_data/02_interim/cmip6/ssp370"
GRIDFILE="${INROOT}/grid_1x2_60S60N_120x180.txt"

ZIPLVL="${ZIPLVL:-4}"
KEEP_INTERMEDIATE="${KEEP_INTERMEDIATE:-0}"   # 1이면 중간 산출물 보관
OVERWRITE="${OVERWRITE:-0}"                   # 1이면 기존 결과 덮어씀

mkdir -p "$OUTROOT"

# thetao 제외
is_target_var() {
  local v="$1"
  case "$v" in
    mlotst|ohc300|psl|sos|tos|uas|uos|vas|vos) return 0 ;;
    *) return 1 ;;
  esac
}

process_one() {
  local in="$1"

  local rel outdir base var
  rel="${in#$INROOT/}"
  outdir="$OUTROOT/$(dirname "$rel")"
  base="$(basename "$in" .nc)"
  var="${base%%_*}"

  if ! is_target_var "$var"; then
    echo "[skip-var] $in"
    return 0
  fi

  mkdir -p "$outdir"

  local sel detr clim anom out tmp
  sel="${outdir}/${base}.sel.nc"
  detr="${outdir}/${base}.detr.nc"
  clim="${outdir}/${base}.clim.nc"
  anom="${outdir}/${base}.anom.nc"
  out="${outdir}/${base}.anom_1x2.nc"
  tmp="${out}.tmp.$$"

  if [[ -s "$out" && "$OVERWRITE" != "1" ]]; then
    echo "[skip-exists] $out"
    return 0
  fi

  echo "[proc] $in"

  # 0) 주 변수만 선택
  cdo -L -O -f nc4c -z "zip_${ZIPLVL}" \
    selname,"${var}" "$in" "$sel"

  # 1) 장기 선형 추세 제거
  cdo -L -O -f nc4c -z "zip_${ZIPLVL}" \
    detrend "$sel" "$detr"

  # 2) detrended 자료의 월기후값
  cdo -L -O -f nc4c -z "zip_${ZIPLVL}" \
    ymonmean "$detr" "$clim"

  # 3) 월 anomaly = detrended - monthly climatology
  cdo -L -O -f nc4c -z "zip_${ZIPLVL}" \
    ymonsub "$detr" "$clim" "$anom"

  # 4) 1°x2° (lat 1°, lon 2°), 60S-60N, 0-360E 선형보간
  cdo -L -O -f nc4c -z "zip_${ZIPLVL}" \
    remapbil,"$GRIDFILE" "$anom" "$tmp"

  mv -f "$tmp" "$out"

  if [[ "$KEEP_INTERMEDIATE" != "1" ]]; then
    rm -f "$sel" "$detr" "$clim" "$anom"
  fi

  echo "[done] $out"
}

export INROOT OUTROOT GRIDFILE ZIPLVL KEEP_INTERMEDIATE OVERWRITE
export -f is_target_var
export -f process_one

find "$INROOT" -type f -path "*/monthly/*.nc" -print0 | sort -z | \
while IFS= read -r -d '' f; do
  process_one "$f"
done

echo "[all done]"
