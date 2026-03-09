#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/d/project/01_ENSO/01_data/01_raw/oras5

# =========================
# uos
# =========================
mkdir -p "$ROOT/zonal_velocity/_surf2d"

for f in "$ROOT"/zonal_velocity/vozocrtx_control_monthly_highres_3D_*.nc; do
    b=$(basename "$f")
    cdo -L -O --reduce_dim sellevidx,1 "$f" "$ROOT/zonal_velocity/_surf2d/$b"
done

cdo -L -O mergetime \
    "$ROOT"/zonal_velocity/_surf2d/vozocrtx_control_monthly_highres_3D_*.nc \
    "$ROOT"/zonal_velocity/_surf2d/vozocrtx_surface_2d_1958-1978.nc

cdo -L -O chname,vozocrtx,uos \
    "$ROOT"/zonal_velocity/_surf2d/vozocrtx_surface_2d_1958-1978.nc \
    "$ROOT"/uos.1958-1978.nc

# =========================
# vos
# =========================
mkdir -p "$ROOT/meridional_velocity/_surf2d"

for f in "$ROOT"/meridional_velocity/vomecrty_control_monthly_highres_3D_*.nc; do
    b=$(basename "$f")
    cdo -L -O --reduce_dim sellevidx,1 "$f" "$ROOT/meridional_velocity/_surf2d/$b"
done

cdo -L -O mergetime \
    "$ROOT"/meridional_velocity/_surf2d/vomecrty_control_monthly_highres_3D_*.nc \
    "$ROOT"/meridional_velocity/_surf2d/vomecrty_surface_2d_1958-1978.nc

cdo -L -O chname,vomecrty,vos \
    "$ROOT"/meridional_velocity/_surf2d/vomecrty_surface_2d_1958-1978.nc \
    "$ROOT"/vos.1958-1978.nc

echo "=== check ==="
cdo showname "$ROOT/uos.1958-1978.nc"
cdo showname "$ROOT/vos.1958-1978.nc"
cdo ntime "$ROOT/uos.1958-1978.nc"
cdo ntime "$ROOT/vos.1958-1978.nc"

