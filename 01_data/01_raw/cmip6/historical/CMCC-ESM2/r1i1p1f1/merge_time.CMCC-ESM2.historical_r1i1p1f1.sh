#!/usr/bin/env bash
set -euo pipefail

echo '[START] vo_Omon_CMCC-ESM2_historical_r1i1p1f1_gn_time.nc'
rm -f "vo_Omon_CMCC-ESM2_historical_r1i1p1f1_gn_time.nc.tmp"
cdo -O mergetime "vo_Omon_CMCC-ESM2_historical_r1i1p1f1_gn_185001-186912.nc" "vo_Omon_CMCC-ESM2_historical_r1i1p1f1_gn_187001-188912.nc" "vo_Omon_CMCC-ESM2_historical_r1i1p1f1_gn_189001-190912.nc" "vo_Omon_CMCC-ESM2_historical_r1i1p1f1_gn_191001-192912.nc" "vo_Omon_CMCC-ESM2_historical_r1i1p1f1_gn_193001-194912.nc" "vo_Omon_CMCC-ESM2_historical_r1i1p1f1_gn_195001-196912.nc" "vo_Omon_CMCC-ESM2_historical_r1i1p1f1_gn_197001-198912.nc" "vo_Omon_CMCC-ESM2_historical_r1i1p1f1_gn_199001-200912.nc" "vo_Omon_CMCC-ESM2_historical_r1i1p1f1_gn_201001-201412.nc" "vo_Omon_CMCC-ESM2_historical_r1i1p1f1_gn_time.nc.tmp"
mv "vo_Omon_CMCC-ESM2_historical_r1i1p1f1_gn_time.nc.tmp" "vo_Omon_CMCC-ESM2_historical_r1i1p1f1_gn_time.nc"
[[ -s "vo_Omon_CMCC-ESM2_historical_r1i1p1f1_gn_time.nc" ]] || { echo "ERROR: output missing: vo_Omon_CMCC-ESM2_historical_r1i1p1f1_gn_time.nc" >&2; exit 1; }
rm -f "vo_Omon_CMCC-ESM2_historical_r1i1p1f1_gn_185001-186912.nc" "vo_Omon_CMCC-ESM2_historical_r1i1p1f1_gn_187001-188912.nc" "vo_Omon_CMCC-ESM2_historical_r1i1p1f1_gn_189001-190912.nc" "vo_Omon_CMCC-ESM2_historical_r1i1p1f1_gn_191001-192912.nc" "vo_Omon_CMCC-ESM2_historical_r1i1p1f1_gn_193001-194912.nc" "vo_Omon_CMCC-ESM2_historical_r1i1p1f1_gn_195001-196912.nc" "vo_Omon_CMCC-ESM2_historical_r1i1p1f1_gn_197001-198912.nc" "vo_Omon_CMCC-ESM2_historical_r1i1p1f1_gn_199001-200912.nc" "vo_Omon_CMCC-ESM2_historical_r1i1p1f1_gn_201001-201412.nc"
echo '[DONE ] vo_Omon_CMCC-ESM2_historical_r1i1p1f1_gn_time.nc  (source files removed)'

echo '[START] uo_Omon_CMCC-ESM2_historical_r1i1p1f1_gn_time.nc'
rm -f "uo_Omon_CMCC-ESM2_historical_r1i1p1f1_gn_time.nc.tmp"
cdo -O mergetime "uo_Omon_CMCC-ESM2_historical_r1i1p1f1_gn_185001-186912.nc" "uo_Omon_CMCC-ESM2_historical_r1i1p1f1_gn_187001-188912.nc" "uo_Omon_CMCC-ESM2_historical_r1i1p1f1_gn_189001-190912.nc" "uo_Omon_CMCC-ESM2_historical_r1i1p1f1_gn_191001-192912.nc" "uo_Omon_CMCC-ESM2_historical_r1i1p1f1_gn_193001-194912.nc" "uo_Omon_CMCC-ESM2_historical_r1i1p1f1_gn_195001-196912.nc" "uo_Omon_CMCC-ESM2_historical_r1i1p1f1_gn_197001-198912.nc" "uo_Omon_CMCC-ESM2_historical_r1i1p1f1_gn_199001-200912.nc" "uo_Omon_CMCC-ESM2_historical_r1i1p1f1_gn_201001-201412.nc" "uo_Omon_CMCC-ESM2_historical_r1i1p1f1_gn_time.nc.tmp"
mv "uo_Omon_CMCC-ESM2_historical_r1i1p1f1_gn_time.nc.tmp" "uo_Omon_CMCC-ESM2_historical_r1i1p1f1_gn_time.nc"
[[ -s "uo_Omon_CMCC-ESM2_historical_r1i1p1f1_gn_time.nc" ]] || { echo "ERROR: output missing: uo_Omon_CMCC-ESM2_historical_r1i1p1f1_gn_time.nc" >&2; exit 1; }
rm -f "uo_Omon_CMCC-ESM2_historical_r1i1p1f1_gn_185001-186912.nc" "uo_Omon_CMCC-ESM2_historical_r1i1p1f1_gn_187001-188912.nc" "uo_Omon_CMCC-ESM2_historical_r1i1p1f1_gn_189001-190912.nc" "uo_Omon_CMCC-ESM2_historical_r1i1p1f1_gn_191001-192912.nc" "uo_Omon_CMCC-ESM2_historical_r1i1p1f1_gn_193001-194912.nc" "uo_Omon_CMCC-ESM2_historical_r1i1p1f1_gn_195001-196912.nc" "uo_Omon_CMCC-ESM2_historical_r1i1p1f1_gn_197001-198912.nc" "uo_Omon_CMCC-ESM2_historical_r1i1p1f1_gn_199001-200912.nc" "uo_Omon_CMCC-ESM2_historical_r1i1p1f1_gn_201001-201412.nc"
echo '[DONE ] uo_Omon_CMCC-ESM2_historical_r1i1p1f1_gn_time.nc  (source files removed)'

echo '[START] thetao_Omon_CMCC-ESM2_historical_r1i1p1f1_gn_time.nc'
rm -f "thetao_Omon_CMCC-ESM2_historical_r1i1p1f1_gn_time.nc.tmp"
cdo -O mergetime "thetao_Omon_CMCC-ESM2_historical_r1i1p1f1_gn_185001-186912.nc" "thetao_Omon_CMCC-ESM2_historical_r1i1p1f1_gn_187001-188912.nc" "thetao_Omon_CMCC-ESM2_historical_r1i1p1f1_gn_189001-190912.nc" "thetao_Omon_CMCC-ESM2_historical_r1i1p1f1_gn_191001-192912.nc" "thetao_Omon_CMCC-ESM2_historical_r1i1p1f1_gn_193001-194912.nc" "thetao_Omon_CMCC-ESM2_historical_r1i1p1f1_gn_195001-196912.nc" "thetao_Omon_CMCC-ESM2_historical_r1i1p1f1_gn_197001-198912.nc" "thetao_Omon_CMCC-ESM2_historical_r1i1p1f1_gn_199001-200912.nc" "thetao_Omon_CMCC-ESM2_historical_r1i1p1f1_gn_201001-201412.nc" "thetao_Omon_CMCC-ESM2_historical_r1i1p1f1_gn_time.nc.tmp"
mv "thetao_Omon_CMCC-ESM2_historical_r1i1p1f1_gn_time.nc.tmp" "thetao_Omon_CMCC-ESM2_historical_r1i1p1f1_gn_time.nc"
[[ -s "thetao_Omon_CMCC-ESM2_historical_r1i1p1f1_gn_time.nc" ]] || { echo "ERROR: output missing: thetao_Omon_CMCC-ESM2_historical_r1i1p1f1_gn_time.nc" >&2; exit 1; }
rm -f "thetao_Omon_CMCC-ESM2_historical_r1i1p1f1_gn_185001-186912.nc" "thetao_Omon_CMCC-ESM2_historical_r1i1p1f1_gn_187001-188912.nc" "thetao_Omon_CMCC-ESM2_historical_r1i1p1f1_gn_189001-190912.nc" "thetao_Omon_CMCC-ESM2_historical_r1i1p1f1_gn_191001-192912.nc" "thetao_Omon_CMCC-ESM2_historical_r1i1p1f1_gn_193001-194912.nc" "thetao_Omon_CMCC-ESM2_historical_r1i1p1f1_gn_195001-196912.nc" "thetao_Omon_CMCC-ESM2_historical_r1i1p1f1_gn_197001-198912.nc" "thetao_Omon_CMCC-ESM2_historical_r1i1p1f1_gn_199001-200912.nc" "thetao_Omon_CMCC-ESM2_historical_r1i1p1f1_gn_201001-201412.nc"
echo '[DONE ] thetao_Omon_CMCC-ESM2_historical_r1i1p1f1_gn_time.nc  (source files removed)'

