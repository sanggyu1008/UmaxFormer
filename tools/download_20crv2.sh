#!/usr/bin/env bash
set -euo pipefail

OUTDIR="../data/raw/20crv2/uas_daily_10m"
mkdir -p "${OUTDIR}"

BASE="https://psl.noaa.gov/thredds/fileServer/Datasets/20thC_ReanV2/Dailies/gaussian/monolevel"

seq 1871 2012 | xargs -n1 -P4 -I{} bash -c '
  y="$1"
  f="uwnd.10m.${y}.nc"
  echo "[DOWN] ${f}"
  wget -c -nv \
    --retry-connrefused --waitretry=5 --tries=20 --timeout=60 \
    -O "'"${OUTDIR}"'/${f}" \
    "'"${BASE}"'/${f}"
' _ {}
