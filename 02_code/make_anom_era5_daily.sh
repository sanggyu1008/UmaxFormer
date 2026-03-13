#!/usr/bin/env bash
set -euo pipefail

command -v cdo >/dev/null 2>&1 || {
  echo "[error] cdo not found in PATH" >&2
  exit 127
}

# ============================================================
# ERA5 daily uas anomaly (always remap to 1x2 by default)
# - input directory : /mnt/d/project/01_ENSO/01_data/01_raw/era5
# - output directory: /mnt/d/project/01_ENSO/01_data/02_interim/era5
# - target file     : uas.day.195801-202512.nc
# - process         : select main var -> rename if needed -> detrend
#                     -> ydaymean -> ydaysub -> remapbil -> setmissval
# ============================================================

IN_DIR="/mnt/d/project/01_ENSO/01_data/01_raw/era5"
OUT_DIR="/mnt/d/project/01_ENSO/01_data/02_interim/era5"
GRIDFILE="/mnt/d/project/01_ENSO/01_data/01_raw/cmip6/ssp370/grid_1x2_60S60N_120x180.txt"
INFILE="$IN_DIR/uas.day.195801-202512.nc"
OUTVAR="uas"
MISSVAL="1e20"
ZIPLVL="zip_4"
OVERWRITE="${OVERWRITE:-0}"
KEEP_INTERMEDIATE="${KEEP_INTERMEDIATE:-0}"

mkdir -p "$OUT_DIR"

[[ -d "$IN_DIR" ]] || { echo "[error] input dir not found: $IN_DIR" >&2; exit 1; }
[[ -f "$INFILE" ]] || { echo "[error] input file not found: $INFILE" >&2; exit 1; }
[[ -f "$GRIDFILE" ]] || { echo "[error] grid file not found: $GRIDFILE" >&2; exit 1; }

resolve_var() {
  local infile="$1"; shift
  local names
  names="$(cdo -s showname "$infile" | tr ' ' '\n' | sed '/^$/d')"
  local cand
  for cand in "$@"; do
    if grep -Fxq "$cand" <<< "$names"; then
      printf '%s\n' "$cand"
      return 0
    fi
  done
  printf '%s\n' "$(head -n 1 <<< "$names")"
}

process_one() {
  local infile="$1"
  local outvar="$2"
  shift 2
  local invar
  invar="$(resolve_var "$infile" "$@")"

  local bname stem tmpdir sel ren detr clim anom remap out
  bname="$(basename "$infile")"
  stem="${bname%.nc}"
  out="$OUT_DIR/${stem}.anom_1x2.nc"

  if [[ -s "$out" && "$OVERWRITE" != "1" ]]; then
    echo "[skip] $out"
    return 0
  fi

  tmpdir="$(mktemp -d "$OUT_DIR/.tmp_${outvar}_XXXXXX")"
  sel="$tmpdir/${stem}.sel.nc"
  ren="$tmpdir/${stem}.ren.nc"
  detr="$tmpdir/${stem}.detr.nc"
  clim="$tmpdir/${stem}.ydayclim.nc"
  anom="$tmpdir/${stem}.anom.nc"
  remap="$tmpdir/${stem}.anom_1x2.tmp.nc"

  echo "[proc] $infile (input var: $invar -> output var: $outvar)"

  # 0) 주 변수만 선택
  cdo -L -O -f nc4c -z "$ZIPLVL" selname,"$invar" "$infile" "$sel"

  # 1) 변수명 통일(u10 -> uas)
  if [[ "$invar" != "$outvar" ]]; then
    cdo -L -O -f nc4c -z "$ZIPLVL" chname,"$invar","$outvar" "$sel" "$ren"
  else
    cp -f "$sel" "$ren"
  fi

  # 2) 장기 선형 추세 제거
  cdo -L -O -f nc4c -z "$ZIPLVL" detrend "$ren" "$detr"

  # 3) detrended 자료의 일기후값(day-of-year climatology)
  cdo -L -O -f nc4c -z "$ZIPLVL" ydaymean "$detr" "$clim"

  # 4) 일 anomaly = detrended - day-of-year climatology
  cdo -L -O -f nc4c -z "$ZIPLVL" ydaysub "$detr" "$clim" "$anom"

  # 5) 1°x2° (lat 1°, lon 2°), 60S-60N, 0-360E 선형보간
  cdo -L -O -f nc4c -z "$ZIPLVL" remapbil,"$GRIDFILE" "$anom" "$remap"

  # 6) finite missing value 명시
  cdo -L -O -f nc4c -z "$ZIPLVL" setmissval,"$MISSVAL" "$remap" "$out"

  if [[ "$KEEP_INTERMEDIATE" == "1" ]]; then
    local keepdir="$OUT_DIR/_intermediate/${outvar}_day"
    mkdir -p "$keepdir"
    cp -f "$sel"   "$keepdir/"
    cp -f "$ren"   "$keepdir/"
    cp -f "$detr"  "$keepdir/"
    cp -f "$clim"  "$keepdir/"
    cp -f "$anom"  "$keepdir/"
    cp -f "$remap" "$keepdir/"
  fi

  rm -rf "$tmpdir"
  echo "[done] $out"
}

process_one "$INFILE" "$OUTVAR" uas u10

echo "[all done] ERA5 daily uas anomaly + 1x2 remap finished."