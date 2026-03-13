#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

# ============================================================
# 사용자 설정
# ============================================================
ROOT = Path("/mnt/d/project/01_ENSO/01_data/01_raw/godas")
OUTDIR = Path("/mnt/d/project/01_ENSO/03_output/qc_godas_mask")
OUTDIR.mkdir(parents=True, exist_ok=True)

VARS = ["mlotst", "ohc300", "sos", "tos", "uos", "vos"]

# ============================================================
# 유틸
# ============================================================
def find_time_name(ds: xr.Dataset) -> str:
    for cand in ["time", "TIME", "t"]:
        if cand in ds.coords or cand in ds.variables:
            return cand
    for name, var in ds.variables.items():
        axis = str(var.attrs.get("axis", "")).upper()
        stdn = str(var.attrs.get("standard_name", "")).lower()
        if axis == "T" or stdn == "time":
            return name
    raise ValueError("time coordinate not found")


def open_dataarray(ncfile: Path, varname: str) -> tuple[xr.Dataset, xr.DataArray, str]:
    ds = xr.open_dataset(ncfile, decode_times=True)

    time_name = find_time_name(ds)

    if varname in ds.data_vars:
        da = ds[varname]
    else:
        # fallback: time 차원을 가진 대표 변수
        cands = []
        for name, x in ds.data_vars.items():
            if time_name in x.dims:
                size = int(np.prod([x.sizes[d] for d in x.dims], dtype=np.int64))
                cands.append((name, x.ndim, size))
        if not cands:
            ds.close()
            raise ValueError(f"data variable not found in {ncfile.name}")
        cands.sort(key=lambda z: (z[1], z[2]), reverse=True)
        da = ds[cands[0][0]]

    return ds, da, time_name


def make_mask_change(da: xr.DataArray, time_name: str) -> xr.DataArray:
    # 시간에 따라 valid/missing이 한 번이라도 바뀐 격자
    return da.notnull().astype(np.int8).std(time_name) > 0


def make_flip_count(da: xr.DataArray, time_name: str) -> xr.DataArray:
    # valid/missing 상태 전환 횟수
    return np.abs(da.notnull().astype(np.int8).diff(time_name)).sum(time_name)


def make_nan_count(da: xr.DataArray, time_name: str) -> xr.DataArray:
    space_dims = [d for d in da.dims if d != time_name]
    return da.isnull().sum(dim=space_dims)


def summarize_one(var: str) -> dict:
    ncfile = ROOT / f"{var}.198001-202512.nc"
    if not ncfile.exists():
        return {
            "variable": var,
            "file": str(ncfile),
            "status": "ERROR",
            "n_time": np.nan,
            "n_grid": np.nan,
            "changed_cells": np.nan,
            "fraction_changed": np.nan,
            "max_flip_count": np.nan,
            "mean_flip_on_changed_cells": np.nan,
            "nan_count_min": np.nan,
            "nan_count_max": np.nan,
            "nan_count_range": np.nan,
            "note": "file not found",
        }

    ds = None
    try:
        ds, da, time_name = open_dataarray(ncfile, var)

        mask_change = make_mask_change(da, time_name)
        flip_count = make_flip_count(da, time_name)
        nan_count = make_nan_count(da, time_name)
        nan_count_delta = nan_count - nan_count.min()

        changed_cells = int(mask_change.sum().item())
        total_cells = int(mask_change.size)
        fraction_changed = changed_cells / total_cells if total_cells > 0 else np.nan

        max_flip = int(flip_count.max().item())
        changed_flip = flip_count.where(flip_count > 0)
        mean_flip_changed = float(changed_flip.mean().item()) if changed_cells > 0 else 0.0

        nan_min = int(nan_count.min().item())
        nan_max = int(nan_count.max().item())
        nan_rng = nan_max - nan_min

        # --------------------------------------------------------
        # Figure 1: changed/not changed
        # --------------------------------------------------------
        plt.figure(figsize=(10, 4))
        mask_change.plot()
        plt.title(f"Cells where valid/missing status changes over time: {var}")
        plt.tight_layout()
        plt.savefig(OUTDIR / f"{var}_mask_change.png", dpi=150, bbox_inches="tight")
        plt.close()

        # --------------------------------------------------------
        # Figure 2: flip count
        # --------------------------------------------------------
        plt.figure(figsize=(10, 4))
        flip_count.plot()
        plt.title(f"Number of valid/missing flips by grid cell: {var}")
        plt.tight_layout()
        plt.savefig(OUTDIR / f"{var}_flip_count.png", dpi=150, bbox_inches="tight")
        plt.close()

        # --------------------------------------------------------
        # Figure 3: NaN count delta from minimum
        # offset 문제 피하려고 min 대비 변화량으로 그림
        # --------------------------------------------------------
        plt.figure(figsize=(10, 4))
        nan_count_delta.plot()
        plt.ticklabel_format(axis="y", style="plain", useOffset=False)
        plt.title(f"NaN count by time (delta from minimum): {var}")
        plt.ylabel("NaN count - min(NaN count)")
        plt.tight_layout()
        plt.savefig(OUTDIR / f"{var}_nan_count_delta.png", dpi=150, bbox_inches="tight")
        plt.close()

        # 간단 판정
        # - 변경 격자가 아주 적고
        # - NaN 총개수 변화폭도 작으면 OK
        # - 아니면 REVIEW
        if fraction_changed < 0.01 and nan_rng <= 20:
            status = "OK"
            note = "small mask variation"
        else:
            status = "REVIEW"
            note = "inspect maps and time series"

        return {
            "variable": var,
            "file": str(ncfile),
            "status": status,
            "n_time": int(da.sizes[time_name]),
            "n_grid": total_cells,
            "changed_cells": changed_cells,
            "fraction_changed": fraction_changed,
            "max_flip_count": max_flip,
            "mean_flip_on_changed_cells": mean_flip_changed,
            "nan_count_min": nan_min,
            "nan_count_max": nan_max,
            "nan_count_range": nan_rng,
            "note": note,
        }

    except Exception as e:
        return {
            "variable": var,
            "file": str(ncfile),
            "status": "ERROR",
            "n_time": np.nan,
            "n_grid": np.nan,
            "changed_cells": np.nan,
            "fraction_changed": np.nan,
            "max_flip_count": np.nan,
            "mean_flip_on_changed_cells": np.nan,
            "nan_count_min": np.nan,
            "nan_count_max": np.nan,
            "nan_count_range": np.nan,
            "note": f"{type(e).__name__}: {e}",
        }

    finally:
        if ds is not None:
            ds.close()


def main():
    rows = []
    for var in VARS:
        print(f"[CHECK] {var}")
        rows.append(summarize_one(var))

    df = pd.DataFrame(rows)
    df.to_csv(OUTDIR / "godas_mask_change_summary.csv", index=False)

    print("\n============================================================")
    print(f"Input dir : {ROOT}")
    print(f"Output dir: {OUTDIR}")
    print("------------------------------------------------------------")
    print(df.to_string(index=False))
    print("============================================================\n")


if __name__ == "__main__":
    main()