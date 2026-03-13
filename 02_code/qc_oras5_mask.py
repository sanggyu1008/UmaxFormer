#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


# ============================================================
# 고정 설정
# ============================================================
DEFAULT_ROOT = Path("/mnt/d/project/01_ENSO/01_data/01_raw/oras5")
OUTPUT_DIR = Path("/mnt/d/project/01_ENSO/03_output/qc_oras5_mask")
OCEAN_VARS = ["mlotst", "ohc300", "sos", "tos", "uos", "vos"]


# ============================================================
# 유틸
# ============================================================
def find_time_name(ds: xr.Dataset) -> str:
    for cand in ["time", "time_counter", "TIME", "t"]:
        if cand in ds.coords or cand in ds.variables:
            return cand
    for name, var in ds.variables.items():
        axis = str(var.attrs.get("axis", "")).upper()
        stdn = str(var.attrs.get("standard_name", "")).lower()
        if axis == "T" or stdn == "time":
            return name
    raise ValueError("time coordinate not found")


def find_target_file(root: Path, var: str) -> Path | None:
    matches = sorted([p for p in root.glob(f"{var}*.nc") if p.is_file()])
    if not matches:
        return None
    exact = [p for p in matches if p.name.startswith(f"{var}.")]
    return exact[0] if exact else matches[0]


def open_dataarray(ncfile: Path, varname: str) -> tuple[xr.Dataset, xr.DataArray, str]:
    try:
        ds = xr.open_dataset(ncfile, decode_times=True)
    except Exception:
        ds = xr.open_dataset(ncfile, decode_times=False)

    time_name = find_time_name(ds)

    if varname in ds.data_vars:
        da = ds[varname]
    else:
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
    return da.notnull().astype(np.int8).std(time_name) > 0


def make_flip_count(da: xr.DataArray, time_name: str) -> xr.DataArray:
    return np.abs(da.notnull().astype(np.int8).diff(time_name)).sum(time_name)


def make_nan_count(da: xr.DataArray, time_name: str) -> xr.DataArray:
    space_dims = [d for d in da.dims if d != time_name]
    return da.isnull().sum(dim=space_dims)


def summarize_one(root: Path, outdir: Path, var: str) -> dict:
    ncfile = find_target_file(root, var)
    if ncfile is None:
        return {
            "variable": var,
            "file": "",
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

        plt.figure(figsize=(10, 4))
        mask_change.plot()
        plt.title(f"Cells where valid/missing status changes over time: {var}")
        plt.tight_layout()
        plt.savefig(outdir / f"{var}_mask_change.png", dpi=150, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(10, 4))
        flip_count.plot()
        plt.title(f"Number of valid/missing flips by grid cell: {var}")
        plt.tight_layout()
        plt.savefig(outdir / f"{var}_flip_count.png", dpi=150, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(10, 4))
        nan_count_delta.plot()
        plt.ticklabel_format(axis="y", style="plain", useOffset=False)
        plt.title(f"NaN count by time (delta from minimum): {var}")
        plt.ylabel("NaN count - min(NaN count)")
        plt.tight_layout()
        plt.savefig(outdir / f"{var}_nan_count_delta.png", dpi=150, bbox_inches="tight")
        plt.close()

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
    parser = argparse.ArgumentParser(
        description="Mask-change QC plots for ORAS5 merged monthly nc files"
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=str(DEFAULT_ROOT),
        help=f"directory containing ORAS5 merged nc files (default: {DEFAULT_ROOT})",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    outdir = OUTPUT_DIR
    outdir.mkdir(parents=True, exist_ok=True)

    if not root.exists() or not root.is_dir():
        raise SystemExit(f"[ERROR] directory not found: {root}")

    rows = []
    for var in OCEAN_VARS:
        print(f"[CHECK] {var}")
        rows.append(summarize_one(root, outdir, var))

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "oras5_mask_change_summary.csv", index=False)

    print("\n============================================================")
    print(f"Input dir : {root}")
    print(f"Output dir: {outdir}")
    print("------------------------------------------------------------")
    print(df.to_string(index=False))
    print("============================================================\n")


if __name__ == "__main__":
    main()