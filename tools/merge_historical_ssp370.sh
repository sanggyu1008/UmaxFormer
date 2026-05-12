#!/usr/bin/env bash
set -euo pipefail

# Merge CMIP6 historical + ssp370 files along time axis.
# Output layout:
#   <OUT_ROOT>/<MODEL>/<MEMBER>/monthly/*.nc
#   <OUT_ROOT>/<MODEL>/<MEMBER>/daily/*.nc
# Default OUT_ROOT is:
#   /mnt/d/project/UmaxFormer/data/raw/cmip6/historical_ssp370
#
# Notes:
# - MRI-ESM2-0 is excluded by default.
# - Monthly vars: mlotst ohc300 psl sos tos uas uos vas vos
# - Daily var for uasmax source: uas_day
# - Assumes historical and ssp370 share the same member name.
# - Monthly: cdo mergetime + sorttimestamp
# - Daily uas_day: cdo mergetime only (skip sorttimestamp to avoid HDF read errors on large files)

RAW_ROOT_DEFAULT="/mnt/d/project/UmaxFormer/data/raw/cmip6"
OUT_ROOT_DEFAULT="${RAW_ROOT_DEFAULT}/historical_ssp370"
MONTHLY_VARS_DEFAULT="mlotst ohc300 psl sos tos uas uos vas vos"
EXCLUDE_MODELS_DEFAULT="MRI-ESM2-0"
OVERWRITE=0
DRY_RUN=0

usage() {
  cat <<USAGE
Usage: $(basename "$0") [options]

Options:
  --raw-root PATH          Base CMIP6 raw root containing historical/ and ssp370/
                           default: ${RAW_ROOT_DEFAULT}
  --out-root PATH          Output root for merged historical_ssp370 files
                           default: ${OUT_ROOT_DEFAULT}
  --monthly-vars "..."     Space-separated monthly variable list
                           default: "${MONTHLY_VARS_DEFAULT}"
  --exclude-models "..."   Space-separated model names to skip
                           default: "${EXCLUDE_MODELS_DEFAULT}"
  --overwrite              Overwrite existing merged outputs
  --dry-run                Print actions without executing merges
  -h, --help               Show this help
USAGE
}

RAW_ROOT="$RAW_ROOT_DEFAULT"
OUT_ROOT="$OUT_ROOT_DEFAULT"
MONTHLY_VARS="$MONTHLY_VARS_DEFAULT"
EXCLUDE_MODELS="$EXCLUDE_MODELS_DEFAULT"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --raw-root)
      RAW_ROOT="$2"; shift 2 ;;
    --out-root)
      OUT_ROOT="$2"; shift 2 ;;
    --monthly-vars)
      MONTHLY_VARS="$2"; shift 2 ;;
    --exclude-models)
      EXCLUDE_MODELS="$2"; shift 2 ;;
    --overwrite)
      OVERWRITE=1; shift ;;
    --dry-run)
      DRY_RUN=1; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1 ;;
  esac
done

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Required command not found: $1" >&2
    exit 1
  }
}

need_cmd cdo
need_cmd find
need_cmd sed
need_cmd awk

HIST_ROOT="${RAW_ROOT}/historical"
SSP_ROOT="${RAW_ROOT}/ssp370"

[[ -d "$HIST_ROOT" ]] || { echo "Missing directory: $HIST_ROOT" >&2; exit 1; }
[[ -d "$SSP_ROOT"  ]] || { echo "Missing directory: $SSP_ROOT" >&2; exit 1; }
mkdir -p "$OUT_ROOT"

contains_word() {
  local needle="$1"; shift
  for item in "$@"; do
    [[ "$item" == "$needle" ]] && return 0
  done
  return 1
}

read -r -a EXCLUDE_ARR <<< "$EXCLUDE_MODELS"
read -r -a MONTHLY_ARR <<< "$MONTHLY_VARS"

pick_one_file() {
  local dir="$1"
  local pattern="$2"
  find "$dir" -maxdepth 1 -type f -name "$pattern" | sort | head -n 1
}

merge_monthly_pair() {
  local hist="$1"
  local ssp="$2"
  local out="$3"

  if [[ -f "$out" && "$OVERWRITE" -ne 1 ]]; then
    echo "[skip] exists: $out"
    return 0
  fi

  mkdir -p "$(dirname "$out")"

  local tmp="${out%.nc}.tmp.nc"
  local sorted="${out%.nc}.sorted.nc"

  echo "[merge-monthly]"
  echo "  hist : $hist"
  echo "  ssp  : $ssp"
  echo "  out  : $out"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    return 0
  fi

  rm -f "$tmp" "$sorted"
  cdo -O mergetime "$hist" "$ssp" "$tmp"
  cdo -O sorttimestamp "$tmp" "$sorted"
  mv -f "$sorted" "$out"
  rm -f "$tmp"
}

merge_daily_pair() {
  local hist="$1"
  local ssp="$2"
  local out="$3"

  if [[ -f "$out" && "$OVERWRITE" -ne 1 ]]; then
    echo "[skip] exists: $out"
    return 0
  fi

  mkdir -p "$(dirname "$out")"

  local tmp="${out%.nc}.tmp.nc"

  echo "[merge-daily]"
  echo "  hist : $hist"
  echo "  ssp  : $ssp"
  echo "  out  : $out"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    return 0
  fi

  rm -f "$tmp"
  cdo -O mergetime "$hist" "$ssp" "$tmp"
  mv -f "$tmp" "$out"
}

monthly_out_name() {
  local hist_base="$1"
  echo "$hist_base" | sed 's/_historical_/_historical_ssp370_/'
}

daily_out_name() {
  local hist_base="$1"
  echo "$hist_base" | sed 's/_historical_/_historical_ssp370_/'
}

count_merged=0
count_missing=0
count_skipped_model=0
count_skipped_member=0

shopt -s nullglob
for model_dir in "$HIST_ROOT"/*; do
  [[ -d "$model_dir" ]] || continue
  model="$(basename "$model_dir")"

  if contains_word "$model" "${EXCLUDE_ARR[@]}"; then
    echo "[skip-model] $model"
    ((count_skipped_model+=1))
    continue
  fi

  if [[ ! -d "$SSP_ROOT/$model" ]]; then
    echo "[warn] ssp370 model directory missing: $SSP_ROOT/$model"
    ((count_missing+=1))
    continue
  fi

  for hist_member_dir in "$model_dir"/r*; do
    [[ -d "$hist_member_dir" ]] || continue
    member="$(basename "$hist_member_dir")"
    ssp_member_dir="$SSP_ROOT/$model/$member"

    if [[ ! -d "$ssp_member_dir" ]]; then
      echo "[warn] ssp370 member missing: $model / $member"
      ((count_skipped_member+=1))
      continue
    fi

    out_monthly_dir="$OUT_ROOT/$model/$member/monthly"
    out_daily_dir="$OUT_ROOT/$model/$member/daily"
    mkdir -p "$out_monthly_dir" "$out_daily_dir"

    # Monthly variables: mergetime + sorttimestamp
    for v in "${MONTHLY_ARR[@]}"; do
      hist_file="$(pick_one_file "$hist_member_dir/monthly" "${v}_*_historical_${member}_*_time.nc")"
      ssp_file="$(pick_one_file "$ssp_member_dir/monthly"  "${v}_*_ssp370_${member}_*_time.nc")"

      if [[ -z "$hist_file" || -z "$ssp_file" ]]; then
        echo "[warn] monthly missing: model=$model member=$member var=$v"
        echo "       hist=${hist_file:-NONE}"
        echo "       ssp =${ssp_file:-NONE}"
        ((count_missing+=1))
        continue
      fi

      out_name="$(monthly_out_name "$(basename "$hist_file")")"
      out_file="$out_monthly_dir/$out_name"
      merge_monthly_pair "$hist_file" "$ssp_file" "$out_file"
      ((count_merged+=1))
    done

    # Daily uas for later uasmax construction: mergetime only
    hist_daily="$(pick_one_file "$hist_member_dir/daily" "uas_day_*_historical_${member}_*_time.nc")"
    ssp_daily="$(pick_one_file "$ssp_member_dir/daily"  "uas_day_*_ssp370_${member}_*_time.nc")"

    if [[ -n "$hist_daily" && -n "$ssp_daily" ]]; then
      out_name="$(daily_out_name "$(basename "$hist_daily")")"
      out_file="$out_daily_dir/$out_name"
      merge_daily_pair "$hist_daily" "$ssp_daily" "$out_file"
      ((count_merged+=1))
    else
      echo "[warn] daily uas missing: model=$model member=$member"
      echo "       hist=${hist_daily:-NONE}"
      echo "       ssp =${ssp_daily:-NONE}"
      ((count_missing+=1))
    fi
  done
done

cat <<SUMMARY

=== merge summary ===
raw_root          : $RAW_ROOT
out_root          : $OUT_ROOT
monthly_vars      : $MONTHLY_VARS
exclude_models    : $EXCLUDE_MODELS
merged_outputs    : $count_merged
missing_warnings  : $count_missing
skipped_models    : $count_skipped_model
skipped_members   : $count_skipped_member
SUMMARY
