set -euo pipefail

CODE_ROOT="/mnt/d/project/01_ENSO/02_code"
ROOT="$PWD"

python "${CODE_ROOT}/make_surface_fields.py" \
  --source cmip6 \
  --cmip6-root "${ROOT}" \
  --cmip6-vars thetao uo vo

python "${CODE_ROOT}/calculate_ohc300.py" cmip6 \
  --inroot "${ROOT}" \
  --outroot "${ROOT}" \
  --pattern "*/*/monthly/thetao_*.nc"

python - <<'PY'
import os
from pathlib import Path
import xarray as xr

root = Path(".").resolve()
dry_run = os.environ.get("DRY_RUN", "1") != "0"

def valid_var(path: Path, var: str) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with xr.open_dataset(path, decode_times=False) as ds:
            return var in ds.data_vars and ds[var].size > 0
    except Exception:
        return False

deleted = {"thetao": 0, "uo": 0, "vo": 0}
kept = []

for src in sorted(root.glob("*/*/monthly/thetao_Omon_*_time.nc")):
    tos = src.with_name(src.name.replace("thetao_", "tos_", 1))
    ohc = src.with_name(src.name.replace("thetao_", "ohc300_", 1))
    ok = valid_var(tos, "tos") and valid_var(ohc, "ohc300")
    if ok:
        print(f"[OK] thetao delete candidate: {src}")
        print(f"     verified -> {tos.name}, {ohc.name}")
        if not dry_run:
            src.unlink()
        deleted["thetao"] += 1
    else:
        kept.append((src, "need valid tos and ohc300"))

for src in sorted(root.glob("*/*/monthly/uo_Omon_*_time.nc")):
    out = src.with_name(src.name.replace("uo_", "uos_", 1))
    ok = valid_var(out, "uos")
    if ok:
        print(f"[OK] uo delete candidate: {src}")
        print(f"     verified -> {out.name}")
        if not dry_run:
            src.unlink()
        deleted["uo"] += 1
    else:
        kept.append((src, "need valid uos"))

for src in sorted(root.glob("*/*/monthly/vo_Omon_*_time.nc")):
    out = src.with_name(src.name.replace("vo_", "vos_", 1))
    ok = valid_var(out, "vos")
    if ok:
        print(f"[OK] vo delete candidate: {src}")
        print(f"     verified -> {out.name}")
        if not dry_run:
            src.unlink()
        deleted["vo"] += 1
    else:
        kept.append((src, "need valid vos"))

print()
print("[SUMMARY]")
print(f"  thetao removable : {deleted['thetao']}")
print(f"  uo removable     : {deleted['uo']}")
print(f"  vo removable     : {deleted['vo']}")
print(f"  kept             : {len(kept)}")
print(f"  mode             : {'DRY_RUN' if dry_run else 'DELETE'}")

if kept:
    print("\n[KEPT LIST]")
    for p, reason in kept:
        print(f"  {p}  -- {reason}")
PY
