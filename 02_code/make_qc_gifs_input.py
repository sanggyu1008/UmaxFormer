#!/usr/bin/env python3
from __future__ import annotations
import argparse
import io
import math
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

LAT_CANDIDATES = ["lat", "latitude", "y"]
LON_CANDIDATES = ["lon", "longitude", "x"]
TIME_CANDIDATES = ["time", "valid_time"]

PREFERRED_VAR_ORDER = [
    "tos", "ohc300", "mlotst", "sos", "uos",
    "vos", "uas", "uasmax", "vas", "psl",
]


def find_coord_name(ds: xr.Dataset, candidates: list[str]) -> str | None:
    for name in candidates:
        if name in ds.coords or name in ds.dims:
            return name
    return None


def sort_vars(var_names: list[str]) -> list[str]:
    ordered = [v for v in PREFERRED_VAR_ORDER if v in var_names]
    rest = sorted(v for v in var_names if v not in ordered)
    return ordered + rest


def pick_plot_vars(
    ds: xr.Dataset,
    time_name: str,
    lat_name: str,
    lon_name: str,
    user_vars: list[str] | None,
) -> list[str]:
    if user_vars:
        missing = [v for v in user_vars if v not in ds.data_vars]
        if missing:
            raise ValueError(f"Requested vars not found: {missing}")
        return user_vars

    out = []
    for v in ds.data_vars:
        da = ds[v]
        dims = set(da.dims)
        if {time_name, lat_name, lon_name}.issubset(dims):
            out.append(v)

    return sort_vars(out)


def compute_symmetric_limits(
    da: xr.DataArray,
    time_name: str,
    sample_idx: np.ndarray,
) -> tuple[float, float]:
    arr = da.isel({time_name: sample_idx}).values
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return -1.0, 1.0

    q02, q98 = np.nanpercentile(finite, [2, 98])
    lim = float(max(abs(q02), abs(q98)))

    if (not np.isfinite(lim)) or lim == 0.0:
        lim = float(np.nanmax(np.abs(finite)))
    if (not np.isfinite(lim)) or lim == 0.0:
        lim = 1.0

    return -lim, lim


def build_time_label(ds: xr.Dataset, time_name: str, idx: int) -> str:
    t = ds[time_name].isel({time_name: idx}).values
    try:
        return np.datetime_as_string(t, unit="D")
    except Exception:
        return str(t)


def render_frame(
    ds: xr.Dataset,
    file_label: str,
    plot_vars: list[str],
    time_name: str,
    lat_name: str,
    lon_name: str,
    tidx: int,
    vlims: dict[str, tuple[float, float]],
    ncols: int,
    dpi: int,
) -> np.ndarray:
    nvars = len(plot_vars)
    nrows = math.ceil(nvars / ncols)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(3.0 * ncols, 2.4 * nrows),
        dpi=dpi
    )
    axes = np.array(axes).reshape(-1)

    time_label = build_time_label(ds, time_name, tidx)

    lat = ds[lat_name].values
    lon = ds[lon_name].values
    extent = [
        float(np.nanmin(lon)),
        float(np.nanmax(lon)),
        float(np.nanmin(lat)),
        float(np.nanmax(lat)),
    ]

    for ax, v in zip(axes, plot_vars):
        da = ds[v].isel({time_name: tidx})
        img = np.asarray(da.values)
        vmin, vmax = vlims[v]

        im = ax.imshow(
            img,
            origin="lower",
            aspect="auto",
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
            extent=extent,
            interpolation="nearest",
        )
        ax.set_title(v, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)

    for ax in axes[nvars:]:
        ax.axis("off")

    fig.suptitle(f"{file_label} | {time_label}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return imageio.imread(buf)


def make_gif_for_file(
    nc_path: Path,
    out_dir: Path,
    frame_step: int,
    duration: float,
    ncols: int,
    dpi: int,
    user_vars: list[str] | None,
) -> None:
    with xr.open_dataset(nc_path) as ds0:
        time_name = find_coord_name(ds0, TIME_CANDIDATES)
        lat_name = find_coord_name(ds0, LAT_CANDIDATES)
        lon_name = find_coord_name(ds0, LON_CANDIDATES)

        if time_name is None or lat_name is None or lon_name is None:
            raise ValueError(f"Could not find time/lat/lon in {nc_path.name}")

        ds = ds0.sortby(lat_name)

        plot_vars = pick_plot_vars(
            ds=ds,
            time_name=time_name,
            lat_name=lat_name,
            lon_name=lon_name,
            user_vars=user_vars,
        )
        if not plot_vars:
            raise ValueError(f"No plottable variables found in {nc_path.name}")

        nt = ds.sizes[time_name]
        sample_idx = np.arange(0, nt, frame_step, dtype=int)
        if sample_idx[-1] != nt - 1:
            sample_idx = np.append(sample_idx, nt - 1)

        vlims = {
            v: compute_symmetric_limits(ds[v], time_name, sample_idx)
            for v in plot_vars
        }

        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{nc_path.stem}.gif"

        print(
            f"[MAKE] {nc_path.name} -> {out_path.name} | "
            f"vars={plot_vars} | frames={len(sample_idx)}"
        )

        with imageio.get_writer(out_path, mode="I", duration=duration, loop=0) as writer:
            for tidx in sample_idx:
                frame = render_frame(
                    ds=ds,
                    file_label=nc_path.stem,
                    plot_vars=plot_vars,
                    time_name=time_name,
                    lat_name=lat_name,
                    lon_name=lon_name,
                    tidx=int(tidx),
                    vlims=vlims,
                    ncols=ncols,
                    dpi=dpi,
                )
                writer.append_data(frame)

        print(f"[SAVE] {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Make low-res QC GIFs for ENSO input datasets."
    )
    p.add_argument(
        "--inroot",
        type=Path,
        default=Path("/mnt/d/project/01_ENSO/01_data/03_input"),
    )
    p.add_argument(
        "--outroot",
        type=Path,
        default=Path("/mnt/d/project/01_ENSO/03_output/input_qc_gif"),
    )
    p.add_argument(
        "--glob",
        default="*.nc",
        help="Glob pattern under inroot (default: *.nc)",
    )
    p.add_argument(
        "--frame-step",
        type=int,
        default=3,
        help="Use one frame every N months (default: 3)",
    )
    p.add_argument(
        "--duration",
        type=float,
        default=0.35,
        help="Seconds per frame in GIF (default: 0.35)",
    )
    p.add_argument(
        "--ncols",
        type=int,
        default=4,
        help="Number of subplot columns (default: 4)",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=90,
        help="Figure dpi (default: 90)",
    )
    p.add_argument(
        "--vars",
        nargs="*",
        default=None,
        help="Optional subset of variables, e.g. --vars tos ohc300 uasmax",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    nc_files = sorted([p for p in args.inroot.glob(args.glob) if p.is_file()])
    if not nc_files:
        raise FileNotFoundError(f"No files matched: {args.inroot / args.glob}")

    for nc_path in nc_files:
        try:
            make_gif_for_file(
                nc_path=nc_path,
                out_dir=args.outroot,
                frame_step=args.frame_step,
                duration=args.duration,
                ncols=args.ncols,
                dpi=args.dpi,
                user_vars=args.vars,
            )
        except Exception as e:
            print(f"[FAIL] {nc_path.name}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
