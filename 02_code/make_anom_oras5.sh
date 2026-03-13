#!/usr/bin/env bash
set -euo pipefail

command -v cdo >/dev/null 2>&1 || {
  echo "[error] cdo not found in PATH" >&2
  exit 127
}

# ============================================================
# ORAS5 monthly anomaly (always remap to 1x2 by default)
# - input directory : /mnt/d/project/01_ENSO/01_data/01_raw/oras5
# - output directory: /mnt/d/project/01_ENSO/01_data/02_interim/oras5
# - target files    : mlotst.195801-197812.nc
#                     ohc300.195801-197812.nc
#                     sos.195801-197812.nc
#                     tos.195801-197812.nc
#                     uos.195801-197812.nc
#                     vos.195801-197812.nc
# - process         : select main var -> rename if needed -> detrend
#                     -> ymonmean -> ymonsub -> remapbil -> setmissval
# ============================================================

IN_DIR="/mnt/d/project/01_ENSO/01_data/01_raw/oras5"
OUT_DIR="/mnt/d/project/01_ENSO/01_data/02_interim/oras5"
GRIDFILE="/mnt/d/project/01_ENSO/01_data/01_raw/grid_1x2_60S60N_120x180.txt"
MISSVAL="1e20"
ZIPLVL="zip_4"
OVERWRITE="${OVERWRITE:-0}"
KEEP_INTERMEDIATE="${KEEP_INTERMEDIATE:-0}"

mkdir -p "$OUT_DIR"

[[ -d "$IN_DIR" ]] || { echo "[error] input dir not found: $IN_DIR" >&2; exit 1; }
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
  clim="$tmpdir/${stem}.clim.nc"
  anom="$tmpdir/${stem}.anom.nc"
  remap="$tmpdir/${stem}.anom_1x2.tmp.nc"

  echo "[proc] $infile (input var: $invar -> output var: $outvar)"

  cdo -L -O -f nc4c -z "$ZIPLVL" selname,"$invar" "$infile" "$sel"

  if [[ "$invar" != "$outvar" ]]; then
    cdo -L -O -f nc4c -z "$ZIPLVL" chname,"$invar","$outvar" "$sel" "$ren"
  else
    cp -f "$sel" "$ren"
  fi

  cdo -L -O -f nc4c -z "$ZIPLVL" detrend "$ren" "$detr"
  cdo -L -O -f nc4c -z "$ZIPLVL" ymonmean "$detr" "$clim"
  cdo -L -O -f nc4c -z "$ZIPLVL" ymonsub  "$detr" "$clim" "$anom"
  cdo -L -O -f nc4c -z "$ZIPLVL" remapbil,"$GRIDFILE" "$anom" "$remap"
  cdo -L -O -f nc4c -z "$ZIPLVL" setmissval,"$MISSVAL" "$remap" "$out"

  if [[ "$KEEP_INTERMEDIATE" == "1" ]]; then
    local keepdir="$OUT_DIR/_intermediate/$outvar"
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

process_one "$IN_DIR/mlotst.195801-197812.nc" mlotst mlotst somxl010
process_one "$IN_DIR/ohc300.195801-197812.nc" ohc300 ohc300 sohtc300
process_one "$IN_DIR/sos.195801-197812.nc"    sos    sos
process_one "$IN_DIR/tos.195801-197812.nc"    tos    tos
process_one "$IN_DIR/uos.195801-197812.nc"    uos    uos vozocrtx
process_one "$IN_DIR/vos.195801-197812.nc"    vos    vos vomecrty

echo "[all done] ORAS5 monthly anomaly + 1x2 remap finished."