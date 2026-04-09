#!/usr/bin/env bash
set -euo pipefail

echo '[START] thetao_Omon_MIROC6_historical_r1i1p1f1_gn_time.nc'
rm -f "thetao_Omon_MIROC6_historical_r1i1p1f1_gn_time.nc.tmp"
cdo -O mergetime "thetao_Omon_MIROC6_historical_r1i1p1f1_gn_185001-185912.nc" "thetao_Omon_MIROC6_historical_r1i1p1f1_gn_186001-186912.nc" "thetao_Omon_MIROC6_historical_r1i1p1f1_gn_187001-187912.nc" "thetao_Omon_MIROC6_historical_r1i1p1f1_gn_188001-188912.nc" "thetao_Omon_MIROC6_historical_r1i1p1f1_gn_189001-189912.nc" "thetao_Omon_MIROC6_historical_r1i1p1f1_gn_190001-190912.nc" "thetao_Omon_MIROC6_historical_r1i1p1f1_gn_191001-191912.nc" "thetao_Omon_MIROC6_historical_r1i1p1f1_gn_192001-192912.nc" "thetao_Omon_MIROC6_historical_r1i1p1f1_gn_193001-193912.nc" "thetao_Omon_MIROC6_historical_r1i1p1f1_gn_194001-194912.nc" "thetao_Omon_MIROC6_historical_r1i1p1f1_gn_195001-195912.nc" "thetao_Omon_MIROC6_historical_r1i1p1f1_gn_196001-196912.nc" "thetao_Omon_MIROC6_historical_r1i1p1f1_gn_197001-197912.nc" "thetao_Omon_MIROC6_historical_r1i1p1f1_gn_198001-198912.nc" "thetao_Omon_MIROC6_historical_r1i1p1f1_gn_199001-199912.nc" "thetao_Omon_MIROC6_historical_r1i1p1f1_gn_200001-200912.nc" "thetao_Omon_MIROC6_historical_r1i1p1f1_gn_201001-201412.nc" "thetao_Omon_MIROC6_historical_r1i1p1f1_gn_time.nc.tmp"
mv "thetao_Omon_MIROC6_historical_r1i1p1f1_gn_time.nc.tmp" "thetao_Omon_MIROC6_historical_r1i1p1f1_gn_time.nc"
[[ -s "thetao_Omon_MIROC6_historical_r1i1p1f1_gn_time.nc" ]] || { echo "ERROR: output missing: thetao_Omon_MIROC6_historical_r1i1p1f1_gn_time.nc" >&2; exit 1; }
rm -f "thetao_Omon_MIROC6_historical_r1i1p1f1_gn_185001-185912.nc" "thetao_Omon_MIROC6_historical_r1i1p1f1_gn_186001-186912.nc" "thetao_Omon_MIROC6_historical_r1i1p1f1_gn_187001-187912.nc" "thetao_Omon_MIROC6_historical_r1i1p1f1_gn_188001-188912.nc" "thetao_Omon_MIROC6_historical_r1i1p1f1_gn_189001-189912.nc" "thetao_Omon_MIROC6_historical_r1i1p1f1_gn_190001-190912.nc" "thetao_Omon_MIROC6_historical_r1i1p1f1_gn_191001-191912.nc" "thetao_Omon_MIROC6_historical_r1i1p1f1_gn_192001-192912.nc" "thetao_Omon_MIROC6_historical_r1i1p1f1_gn_193001-193912.nc" "thetao_Omon_MIROC6_historical_r1i1p1f1_gn_194001-194912.nc" "thetao_Omon_MIROC6_historical_r1i1p1f1_gn_195001-195912.nc" "thetao_Omon_MIROC6_historical_r1i1p1f1_gn_196001-196912.nc" "thetao_Omon_MIROC6_historical_r1i1p1f1_gn_197001-197912.nc" "thetao_Omon_MIROC6_historical_r1i1p1f1_gn_198001-198912.nc" "thetao_Omon_MIROC6_historical_r1i1p1f1_gn_199001-199912.nc" "thetao_Omon_MIROC6_historical_r1i1p1f1_gn_200001-200912.nc" "thetao_Omon_MIROC6_historical_r1i1p1f1_gn_201001-201412.nc"
echo '[DONE ] thetao_Omon_MIROC6_historical_r1i1p1f1_gn_time.nc  (source files removed)'

echo '[START] sos_Omon_MIROC6_historical_r1i1p1f1_gn_time.nc'
rm -f "sos_Omon_MIROC6_historical_r1i1p1f1_gn_time.nc.tmp"
cdo -O mergetime "sos_Omon_MIROC6_historical_r1i1p1f1_gn_185001-194912.nc" "sos_Omon_MIROC6_historical_r1i1p1f1_gn_195001-201412.nc" "sos_Omon_MIROC6_historical_r1i1p1f1_gn_time.nc.tmp"
mv "sos_Omon_MIROC6_historical_r1i1p1f1_gn_time.nc.tmp" "sos_Omon_MIROC6_historical_r1i1p1f1_gn_time.nc"
[[ -s "sos_Omon_MIROC6_historical_r1i1p1f1_gn_time.nc" ]] || { echo "ERROR: output missing: sos_Omon_MIROC6_historical_r1i1p1f1_gn_time.nc" >&2; exit 1; }
rm -f "sos_Omon_MIROC6_historical_r1i1p1f1_gn_185001-194912.nc" "sos_Omon_MIROC6_historical_r1i1p1f1_gn_195001-201412.nc"
echo '[DONE ] sos_Omon_MIROC6_historical_r1i1p1f1_gn_time.nc  (source files removed)'

echo '[START] vo_Omon_MIROC6_historical_r1i1p1f1_gn_time.nc'
rm -f "vo_Omon_MIROC6_historical_r1i1p1f1_gn_time.nc.tmp"
cdo -O mergetime "vo_Omon_MIROC6_historical_r1i1p1f1_gn_185001-185912.nc" "vo_Omon_MIROC6_historical_r1i1p1f1_gn_186001-186912.nc" "vo_Omon_MIROC6_historical_r1i1p1f1_gn_187001-187912.nc" "vo_Omon_MIROC6_historical_r1i1p1f1_gn_188001-188912.nc" "vo_Omon_MIROC6_historical_r1i1p1f1_gn_189001-189912.nc" "vo_Omon_MIROC6_historical_r1i1p1f1_gn_190001-190912.nc" "vo_Omon_MIROC6_historical_r1i1p1f1_gn_191001-191912.nc" "vo_Omon_MIROC6_historical_r1i1p1f1_gn_192001-192912.nc" "vo_Omon_MIROC6_historical_r1i1p1f1_gn_193001-193912.nc" "vo_Omon_MIROC6_historical_r1i1p1f1_gn_194001-194912.nc" "vo_Omon_MIROC6_historical_r1i1p1f1_gn_195001-195912.nc" "vo_Omon_MIROC6_historical_r1i1p1f1_gn_196001-196912.nc" "vo_Omon_MIROC6_historical_r1i1p1f1_gn_197001-197912.nc" "vo_Omon_MIROC6_historical_r1i1p1f1_gn_198001-198912.nc" "vo_Omon_MIROC6_historical_r1i1p1f1_gn_199001-199912.nc" "vo_Omon_MIROC6_historical_r1i1p1f1_gn_200001-200912.nc" "vo_Omon_MIROC6_historical_r1i1p1f1_gn_201001-201412.nc" "vo_Omon_MIROC6_historical_r1i1p1f1_gn_time.nc.tmp"
mv "vo_Omon_MIROC6_historical_r1i1p1f1_gn_time.nc.tmp" "vo_Omon_MIROC6_historical_r1i1p1f1_gn_time.nc"
[[ -s "vo_Omon_MIROC6_historical_r1i1p1f1_gn_time.nc" ]] || { echo "ERROR: output missing: vo_Omon_MIROC6_historical_r1i1p1f1_gn_time.nc" >&2; exit 1; }
rm -f "vo_Omon_MIROC6_historical_r1i1p1f1_gn_185001-185912.nc" "vo_Omon_MIROC6_historical_r1i1p1f1_gn_186001-186912.nc" "vo_Omon_MIROC6_historical_r1i1p1f1_gn_187001-187912.nc" "vo_Omon_MIROC6_historical_r1i1p1f1_gn_188001-188912.nc" "vo_Omon_MIROC6_historical_r1i1p1f1_gn_189001-189912.nc" "vo_Omon_MIROC6_historical_r1i1p1f1_gn_190001-190912.nc" "vo_Omon_MIROC6_historical_r1i1p1f1_gn_191001-191912.nc" "vo_Omon_MIROC6_historical_r1i1p1f1_gn_192001-192912.nc" "vo_Omon_MIROC6_historical_r1i1p1f1_gn_193001-193912.nc" "vo_Omon_MIROC6_historical_r1i1p1f1_gn_194001-194912.nc" "vo_Omon_MIROC6_historical_r1i1p1f1_gn_195001-195912.nc" "vo_Omon_MIROC6_historical_r1i1p1f1_gn_196001-196912.nc" "vo_Omon_MIROC6_historical_r1i1p1f1_gn_197001-197912.nc" "vo_Omon_MIROC6_historical_r1i1p1f1_gn_198001-198912.nc" "vo_Omon_MIROC6_historical_r1i1p1f1_gn_199001-199912.nc" "vo_Omon_MIROC6_historical_r1i1p1f1_gn_200001-200912.nc" "vo_Omon_MIROC6_historical_r1i1p1f1_gn_201001-201412.nc"
echo '[DONE ] vo_Omon_MIROC6_historical_r1i1p1f1_gn_time.nc  (source files removed)'

echo '[START] uo_Omon_MIROC6_historical_r1i1p1f1_gn_time.nc'
rm -f "uo_Omon_MIROC6_historical_r1i1p1f1_gn_time.nc.tmp"
cdo -O mergetime "uo_Omon_MIROC6_historical_r1i1p1f1_gn_185001-185912.nc" "uo_Omon_MIROC6_historical_r1i1p1f1_gn_186001-186912.nc" "uo_Omon_MIROC6_historical_r1i1p1f1_gn_187001-187912.nc" "uo_Omon_MIROC6_historical_r1i1p1f1_gn_188001-188912.nc" "uo_Omon_MIROC6_historical_r1i1p1f1_gn_189001-189912.nc" "uo_Omon_MIROC6_historical_r1i1p1f1_gn_190001-190912.nc" "uo_Omon_MIROC6_historical_r1i1p1f1_gn_191001-191912.nc" "uo_Omon_MIROC6_historical_r1i1p1f1_gn_192001-192912.nc" "uo_Omon_MIROC6_historical_r1i1p1f1_gn_193001-193912.nc" "uo_Omon_MIROC6_historical_r1i1p1f1_gn_194001-194912.nc" "uo_Omon_MIROC6_historical_r1i1p1f1_gn_195001-195912.nc" "uo_Omon_MIROC6_historical_r1i1p1f1_gn_196001-196912.nc" "uo_Omon_MIROC6_historical_r1i1p1f1_gn_197001-197912.nc" "uo_Omon_MIROC6_historical_r1i1p1f1_gn_198001-198912.nc" "uo_Omon_MIROC6_historical_r1i1p1f1_gn_199001-199912.nc" "uo_Omon_MIROC6_historical_r1i1p1f1_gn_200001-200912.nc" "uo_Omon_MIROC6_historical_r1i1p1f1_gn_201001-201412.nc" "uo_Omon_MIROC6_historical_r1i1p1f1_gn_time.nc.tmp"
mv "uo_Omon_MIROC6_historical_r1i1p1f1_gn_time.nc.tmp" "uo_Omon_MIROC6_historical_r1i1p1f1_gn_time.nc"
[[ -s "uo_Omon_MIROC6_historical_r1i1p1f1_gn_time.nc" ]] || { echo "ERROR: output missing: uo_Omon_MIROC6_historical_r1i1p1f1_gn_time.nc" >&2; exit 1; }
rm -f "uo_Omon_MIROC6_historical_r1i1p1f1_gn_185001-185912.nc" "uo_Omon_MIROC6_historical_r1i1p1f1_gn_186001-186912.nc" "uo_Omon_MIROC6_historical_r1i1p1f1_gn_187001-187912.nc" "uo_Omon_MIROC6_historical_r1i1p1f1_gn_188001-188912.nc" "uo_Omon_MIROC6_historical_r1i1p1f1_gn_189001-189912.nc" "uo_Omon_MIROC6_historical_r1i1p1f1_gn_190001-190912.nc" "uo_Omon_MIROC6_historical_r1i1p1f1_gn_191001-191912.nc" "uo_Omon_MIROC6_historical_r1i1p1f1_gn_192001-192912.nc" "uo_Omon_MIROC6_historical_r1i1p1f1_gn_193001-193912.nc" "uo_Omon_MIROC6_historical_r1i1p1f1_gn_194001-194912.nc" "uo_Omon_MIROC6_historical_r1i1p1f1_gn_195001-195912.nc" "uo_Omon_MIROC6_historical_r1i1p1f1_gn_196001-196912.nc" "uo_Omon_MIROC6_historical_r1i1p1f1_gn_197001-197912.nc" "uo_Omon_MIROC6_historical_r1i1p1f1_gn_198001-198912.nc" "uo_Omon_MIROC6_historical_r1i1p1f1_gn_199001-199912.nc" "uo_Omon_MIROC6_historical_r1i1p1f1_gn_200001-200912.nc" "uo_Omon_MIROC6_historical_r1i1p1f1_gn_201001-201412.nc"
echo '[DONE ] uo_Omon_MIROC6_historical_r1i1p1f1_gn_time.nc  (source files removed)'

echo '[START] vas_Amon_MIROC6_historical_r1i1p1f1_gn_time.nc'
rm -f "vas_Amon_MIROC6_historical_r1i1p1f1_gn_time.nc.tmp"
cdo -O mergetime "vas_Amon_MIROC6_historical_r1i1p1f1_gn_185001-194912.nc" "vas_Amon_MIROC6_historical_r1i1p1f1_gn_195001-201412.nc" "vas_Amon_MIROC6_historical_r1i1p1f1_gn_time.nc.tmp"
mv "vas_Amon_MIROC6_historical_r1i1p1f1_gn_time.nc.tmp" "vas_Amon_MIROC6_historical_r1i1p1f1_gn_time.nc"
[[ -s "vas_Amon_MIROC6_historical_r1i1p1f1_gn_time.nc" ]] || { echo "ERROR: output missing: vas_Amon_MIROC6_historical_r1i1p1f1_gn_time.nc" >&2; exit 1; }
rm -f "vas_Amon_MIROC6_historical_r1i1p1f1_gn_185001-194912.nc" "vas_Amon_MIROC6_historical_r1i1p1f1_gn_195001-201412.nc"
echo '[DONE ] vas_Amon_MIROC6_historical_r1i1p1f1_gn_time.nc  (source files removed)'

echo '[START] psl_Amon_MIROC6_historical_r1i1p1f1_gn_time.nc'
rm -f "psl_Amon_MIROC6_historical_r1i1p1f1_gn_time.nc.tmp"
cdo -O mergetime "psl_Amon_MIROC6_historical_r1i1p1f1_gn_185001-194912.nc" "psl_Amon_MIROC6_historical_r1i1p1f1_gn_195001-201412.nc" "psl_Amon_MIROC6_historical_r1i1p1f1_gn_time.nc.tmp"
mv "psl_Amon_MIROC6_historical_r1i1p1f1_gn_time.nc.tmp" "psl_Amon_MIROC6_historical_r1i1p1f1_gn_time.nc"
[[ -s "psl_Amon_MIROC6_historical_r1i1p1f1_gn_time.nc" ]] || { echo "ERROR: output missing: psl_Amon_MIROC6_historical_r1i1p1f1_gn_time.nc" >&2; exit 1; }
rm -f "psl_Amon_MIROC6_historical_r1i1p1f1_gn_185001-194912.nc" "psl_Amon_MIROC6_historical_r1i1p1f1_gn_195001-201412.nc"
echo '[DONE ] psl_Amon_MIROC6_historical_r1i1p1f1_gn_time.nc  (source files removed)'

echo '[START] uas_Amon_MIROC6_historical_r1i1p1f1_gn_time.nc'
rm -f "uas_Amon_MIROC6_historical_r1i1p1f1_gn_time.nc.tmp"
cdo -O mergetime "uas_Amon_MIROC6_historical_r1i1p1f1_gn_185001-194912.nc" "uas_Amon_MIROC6_historical_r1i1p1f1_gn_195001-201412.nc" "uas_Amon_MIROC6_historical_r1i1p1f1_gn_time.nc.tmp"
mv "uas_Amon_MIROC6_historical_r1i1p1f1_gn_time.nc.tmp" "uas_Amon_MIROC6_historical_r1i1p1f1_gn_time.nc"
[[ -s "uas_Amon_MIROC6_historical_r1i1p1f1_gn_time.nc" ]] || { echo "ERROR: output missing: uas_Amon_MIROC6_historical_r1i1p1f1_gn_time.nc" >&2; exit 1; }
rm -f "uas_Amon_MIROC6_historical_r1i1p1f1_gn_185001-194912.nc" "uas_Amon_MIROC6_historical_r1i1p1f1_gn_195001-201412.nc"
echo '[DONE ] uas_Amon_MIROC6_historical_r1i1p1f1_gn_time.nc  (source files removed)'

echo '[START] mlotst_Omon_MIROC6_historical_r1i1p1f1_gn_time.nc'
rm -f "mlotst_Omon_MIROC6_historical_r1i1p1f1_gn_time.nc.tmp"
cdo -O mergetime "mlotst_Omon_MIROC6_historical_r1i1p1f1_gn_185001-194912.nc" "mlotst_Omon_MIROC6_historical_r1i1p1f1_gn_195001-201412.nc" "mlotst_Omon_MIROC6_historical_r1i1p1f1_gn_time.nc.tmp"
mv "mlotst_Omon_MIROC6_historical_r1i1p1f1_gn_time.nc.tmp" "mlotst_Omon_MIROC6_historical_r1i1p1f1_gn_time.nc"
[[ -s "mlotst_Omon_MIROC6_historical_r1i1p1f1_gn_time.nc" ]] || { echo "ERROR: output missing: mlotst_Omon_MIROC6_historical_r1i1p1f1_gn_time.nc" >&2; exit 1; }
rm -f "mlotst_Omon_MIROC6_historical_r1i1p1f1_gn_185001-194912.nc" "mlotst_Omon_MIROC6_historical_r1i1p1f1_gn_195001-201412.nc"
echo '[DONE ] mlotst_Omon_MIROC6_historical_r1i1p1f1_gn_time.nc  (source files removed)'

