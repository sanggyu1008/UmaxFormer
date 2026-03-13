#!/usr/bin/env bash
set -euo pipefail

# --- basic checks ---
command -v cdo >/dev/null 2>&1 || {
  echo "[error] cdo not found in PATH" >&2
  exit 127
}

# script location (so you can run this from anywhere)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# default ROOT relative to /02_code
DEFAULT_ROOT="${SCRIPT_DIR}/../01_data/01_raw/cmip6/ssp370"

# ROOT can be provided as 1st arg; otherwise default is used
ROOT="${1:-$DEFAULT_ROOT}"

# canonicalize ROOT (fail fast if not accessible)
ROOT="$(cd "$ROOT" && pwd)"

# compression level
ZIPLVL="${ZIPLVL:-4}"

# remove singleton dimensions after selecting first level
# 1 = use --reduce_dim, 0 = keep singleton vertical dimension
REDUCE_DIM="${REDUCE_DIM:-1}"

echo "[info] ROOT       = $ROOT"
echo "[info] ZIPLVL     = $ZIPLVL"
echo "[info] REDUCE_DIM = $REDUCE_DIM"
echo

make_one() {
  local in="$1"
  local out="$2"
  shift 2
  local tmp="${out}.tmp.$$"

  if [[ -s "$out" ]]; then
    echo "[skip] $out"
    return 0
  fi

  local -a cdo_opts=(-O -L -f nc4c -z "zip_${ZIPLVL}")
  if [[ "$REDUCE_DIM" == "1" ]]; then
    cdo_opts+=(--reduce_dim)
  fi

  echo "[make] $out"
  if cdo "${cdo_opts[@]}" "$@" "$in" "$tmp"; then
    mv -f "$tmp" "$out"
  else
    rm -f "$tmp"
    echo "[error] failed: $in -> $out" >&2
    exit 1
  fi
}

# --- thetao -> tos (surface/first level) ---
find "$ROOT" -type f -path "*/monthly/thetao_Omon_*_time.nc" -print0 | \
while IFS= read -r -d '' f; do
  d="$(dirname "$f")"
  b="$(basename "$f")"
  out="$d/${b/thetao_/tos_}"

  make_one "$f" "$out" \
    -sellevidx,1 \
    -chname,thetao,tos
done

# --- uo -> uos (surface/first level) ---
find "$ROOT" -type f -path "*/monthly/uo_Omon_*_time.nc" -print0 | \
while IFS= read -r -d '' f; do
  d="$(dirname "$f")"
  b="$(basename "$f")"
  out="$d/${b/uo_/uos_}"

  make_one "$f" "$out" \
    -sellevidx,1 \
    -chname,uo,uos
done

# --- vo -> vos (surface/first level) ---
find "$ROOT" -type f -path "*/monthly/vo_Omon_*_time.nc" -print0 | \
while IFS= read -r -d '' f; do
  d="$(dirname "$f")"
  b="$(basename "$f")"
  out="$d/${b/vo_/vos_}"

  make_one "$f" "$out" \
    -sellevidx,1 \
    -chname,vo,vos
done

echo
echo "[done] surface fields generated into each monthly/ directory."