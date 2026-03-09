#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# Usage:
#   bash cmip6_monthly_anom_1x2.sh [BASE_DIR] [OUT_DIR]
# Example:
#   bash cmip6_monthly_anom_1x2.sh \
#     /mnt/d/project/01_ENSO/01_data/01_raw/cmip6/ssp370 \
#     /mnt/d/project/01_ENSO/01_data/02_processed/cmip6/ssp370
#
# What it does for every */*/monthly/*.nc file:
#   1) remove linear long-term trend
#   2) remove monthly climatology  -> monthly anomaly
#   3) bilinearly remap to 1° x 2° over ~60S-60N, 0-360E
#
# Notes:
# - The target grid below is 120 x 180 (lat x lon), matching the 120x180
#   tensor shape described in Chen et al. (2025).
# - Latitudes are centered at -59.5, -58.5, ..., 59.5.
# - If you want integer latitude endpoints (-60..60), change ysize/yfirst.

BASE_DIR="${1:-/mnt/d/project/01_ENSO/01_data/01_raw/cmip6/ssp370}"
OUT_DIR="${2:-/mnt/d/project/01_ENSO/01_data/02_interim/cmip6/ssp370_anom_1x2}"
TMP_ROOT="${TMPDIR:-/tmp}/cmip6_preproc_${USER:-user}_$$"

mkdir -p "$OUT_DIR" "$TMP_ROOT"
trap 'rm -rf "$TMP_ROOT"' EXIT

GRID_FILE="$OUT_DIR/grid_1x2_60S60N_120x180.txt"
cat > "$GRID_FILE" <<'GRID'
gridtype = lonlat
xsize    = 180
ysize    = 120
xfirst   = 0
yfirst   = -59.5
xinc     = 2
yinc     = 1
GRID

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "ERROR: required command not found: $1" >&2
    exit 1
  }
}

need_cmd cdo

preprocess_one() {
  local infile="$1"
  local rel outfile outsubdir base workdir detr anom afile bfile

  rel="${infile#${BASE_DIR}/}"
  outfile="$OUT_DIR/${rel%.nc}.anom1x2.nc"
  outsubdir="$(dirname "$outfile")"
  base="$(basename "${infile%.nc}")"
  workdir="$TMP_ROOT/$(dirname "$rel")"

  mkdir -p "$outsubdir" "$workdir"

  if [[ -s "$outfile" ]]; then
    log "SKIP  $rel"
    return 0
  fi

  detr="$workdir/${base}.detr.nc"
  anom="$workdir/${base}.anom.nc"
  afile="$workdir/${base}.trend_a.nc"
  bfile="$workdir/${base}.trend_b.nc"

  log "START $rel"

  # 1) linear detrend
  if ! cdo -L -s detrend "$infile" "$detr"; then
    log "WARN  detrend failed, fallback to trend/subtrend: $rel"
    rm -f "$detr"
    cdo -L -s trend "$infile" "$afile" "$bfile"
    cdo -L -s subtrend "$infile" "$afile" "$bfile" "$detr"
    rm -f "$afile" "$bfile"
  fi

  # 2) remove monthly climatology (computed from detrended series)
  cdo -L -s ymonsub "$detr" -ymonmean "$detr" "$anom"

  # 3) bilinear remap to 1° x 2° target grid
  cdo -L -s -f nc4c -z zip_4 remapbil,"$GRID_FILE" "$anom" "$outfile"

  rm -f "$detr" "$anom"
  log "DONE  $rel"
}

export BASE_DIR OUT_DIR TMP_ROOT GRID_FILE
export -f preprocess_one log

mapfile -t FILES < <(find "$BASE_DIR" -type f -path '*/monthly/*.nc' | sort)

if [[ ${#FILES[@]} -eq 0 ]]; then
  echo "No monthly NetCDF files found under: $BASE_DIR" >&2
  exit 1
fi

for f in "${FILES[@]}"; do
  preprocess_one "$f"
done

log "All done. Output root: $OUT_DIR"
