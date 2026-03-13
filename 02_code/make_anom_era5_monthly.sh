#!/usr/bin/env bash
set -euo pipefail

command -v cdo >/dev/null 2>&1 || {
  echo "[error] cdo not found in PATH" >&2
  exit 127
}
command -v python3 >/dev/null 2>&1 || {
  echo "[error] python3 not found in PATH" >&2
  exit 127
}

# ============================================================
# ERA5 monthly anomaly (fixed version)
# - input directory : /mnt/d/project/01_ENSO/01_data/01_raw/era5
# - output directory: /mnt/d/project/01_ENSO/01_data/02_interim/era5
# - target files    : psl.mon.195801-202512.nc
#                     uas.mon.195801-202512.nc
#                     vas.mon.195801-202512.nc
# - process         : select main var -> rename if needed -> rewrite with
#                     explicit finite _FillValue -> detrend -> ymonmean
#                     -> ymonsub -> remapbil -> setmissval
# ============================================================

IN_DIR="/mnt/d/project/01_ENSO/01_data/01_raw/era5"
OUT_DIR="/mnt/d/project/01_ENSO/01_data/02_interim/era5"
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

rewrite_with_finite_fill() {
  local src="$1"
  local dst="$2"
  local var="$3"
  local missval="$4"

  python3 - "$src" "$dst" "$var" "$missval" <<'PY'
import sys
import numpy as np
import xarray as xr

src = sys.argv[1]
dst = sys.argv[2]
var = sys.argv[3]
fill = np.float32(sys.argv[4])

with xr.open_dataset(src, decode_times=False) as ds:
    if var not in ds.variables:
        raise ValueError(f"Variable '{var}' not found in {src}")

    out = ds.copy()
    for key in ('_FillValue', 'missing_value', 'valid_min', 'valid_max', 'valid_range'):
        out[var].attrs.pop(key, None)

    encoding = {
        var: {
            'zlib': True,
            'complevel': 4,
            'shuffle': True,
            'dtype': 'float32',
            '_FillValue': fill,
        }
    }
    out.to_netcdf(dst, encoding=encoding)
PY
}

process_one() {
  local infile="$1"
  local outvar="$2"
  shift 2
  local invar
  invar="$(resolve_var "$infile" "$@")"

  local bname stem tmpdir sel ren clean detr clim anom remap out
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
  clean="$tmpdir/${stem}.clean.nc"
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

  # 중요: detrend 전에 explicit finite _FillValue 로 다시 저장
  rewrite_with_finite_fill "$ren" "$clean" "$outvar" "$MISSVAL"

  cdo -L -O -f nc4c -z "$ZIPLVL" detrend "$clean" "$detr"
  cdo -L -O -f nc4c -z "$ZIPLVL" ymonmean "$detr" "$clim"
  cdo -L -O -f nc4c -z "$ZIPLVL" ymonsub  "$detr" "$clim" "$anom"
  cdo -L -O -f nc4c -z "$ZIPLVL" remapbil,"$GRIDFILE" "$anom" "$remap"
  cdo -L -O -f nc4c -z "$ZIPLVL" setmissval,"$MISSVAL" "$remap" "$out"

  if [[ "$KEEP_INTERMEDIATE" == "1" ]]; then
    local keepdir="$OUT_DIR/_intermediate/$outvar"
    mkdir -p "$keepdir"
    cp -f "$sel"   "$keepdir/"
    cp -f "$ren"   "$keepdir/"
    cp -f "$clean" "$keepdir/"
    cp -f "$detr"  "$keepdir/"
    cp -f "$clim"  "$keepdir/"
    cp -f "$anom"  "$keepdir/"
    cp -f "$remap" "$keepdir/"
  fi

  rm -rf "$tmpdir"
  echo "[done] $out"
}

process_one "$IN_DIR/psl.mon.195801-202512.nc" psl psl msl
process_one "$IN_DIR/uas.mon.195801-202512.nc" uas uas u10
process_one "$IN_DIR/vas.mon.195801-202512.nc" vas vas v10

echo "[all done] ERA5 monthly anomaly + 1x2 remap finished."