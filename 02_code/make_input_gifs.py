#!/usr/bin/env python3
"""
NetCDF 입력 자료를 GIF로 빠르게 확인하는 스크립트.

기본 아이디어
- 각 .nc 파일의 3차원 변수(time, lat, lon)를 찾아서 GIF 생성
- mode=panel  : 한 파일 안의 변수들을 3x3 패널로 묶어서 1개 GIF 생성
- mode=single : 변수별로 각각 GIF 생성
- mode=both   : panel + single 둘 다 생성

예시 1) 모든 input_*.nc 파일에 대해, 12개월 간격(연 단위) 패널 GIF 생성
python make_input_gifs.py \
  --input-dir /mnt/d/project/01_ENSO/01_data/03_input \
  --output-dir /mnt/d/project/01_ENSO/01_data/03_input/gif \
  --pattern 'input_*.nc' \
  --mode panel \
  --stride 12 \
  --fps 4

예시 2) 특정 파일 하나를 변수별 GIF로 생성
python make_input_gifs.py \
  --input-dir /mnt/d/project/01_ENSO/01_data/03_input \
  --output-dir /mnt/d/project/01_ENSO/01_data/03_input/gif \
  --pattern 'input_ACCESS-CM2_ssp370_r1i1p1f1.nc' \
  --mode single \
  --stride 6 \
  --fps 6

예시 3) ENSO 확인용으로 최근접하게 보고 싶으면 월별 전체 대신 프레임 수 제한
python make_input_gifs.py \
  --input-dir /mnt/d/project/01_ENSO/01_data/03_input \
  --output-dir /mnt/d/project/01_ENSO/01_data/03_input/gif \
  --mode panel \
  --stride 1 \
  --max-frames 180 \
  --fps 10

권장
- time=1032 전체를 월별로 GIF로 만들면 파일 크기가 매우 커집니다.
- 먼저 --stride 12 로 확인하고, 필요한 파일만 --stride 1 또는 3으로 다시 뽑는 편이 좋습니다.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


DEFAULT_CMAPS: Dict[str, str] = {
    "tos": "RdBu_r",
    "ohc300": "RdBu_r",
    "mlotst": "RdBu_r",
    "sos": "RdBu_r",
    "psl": "RdBu_r",
    "uos": "RdBu_r",
    "vos": "RdBu_r",
    "uas": "RdBu_r",
    "vas": "RdBu_r",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NetCDF 입력 자료를 GIF로 생성")
    p.add_argument("--input-dir", type=Path, required=True, help=".nc 파일이 있는 디렉터리")
    p.add_argument("--output-dir", type=Path, required=True, help="GIF 저장 디렉터리")
    p.add_argument("--pattern", type=str, default="input_*.nc", help="검색 패턴 (glob)")
    p.add_argument(
        "--mode",
        choices=["panel", "single", "both"],
        default="panel",
        help="panel=파일당 1개 패널 GIF, single=변수별 GIF, both=둘 다",
    )
    p.add_argument(
        "--vars",
        type=str,
        default="",
        help="쉼표로 구분한 변수명. 비워두면 (time,lat,lon) 3차원 변수를 자동 선택",
    )
    p.add_argument("--stride", type=int, default=12, help="시간축 샘플 간격. 12면 연 단위")
    p.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="최대 프레임 수. 0이면 제한 없음",
    )
    p.add_argument("--fps", type=int, default=4, help="GIF fps")
    p.add_argument(
        "--robust-low",
        type=float,
        default=2.0,
        help="컬러 범위 하위 백분위수 (기본 2)",
    )
    p.add_argument(
        "--robust-high",
        type=float,
        default=98.0,
        help="컬러 범위 상위 백분위수 (기본 98)",
    )
    p.add_argument(
        "--center-zero",
        choices=["all", "auto", "none"],
        default="all",
        help="컬러 범위를 0 중심 대칭으로 둘지 설정",
    )
    p.add_argument(
        "--sample-for-scale",
        type=int,
        default=0,
        help="색상 범위 계산에 사용할 최대 샘플 프레임 수. 0이면 실제 사용 프레임 전부",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="이미 GIF가 있으면 건너뜀",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=120,
        help="렌더링 dpi (클수록 선명하지만 느리고 용량 증가)",
    )
    return p.parse_args()


def discover_files(input_dir: Path, pattern: str) -> List[Path]:
    files = sorted(input_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"패턴에 맞는 파일이 없습니다: {input_dir / pattern}")
    return files


def get_lat_lon_names(ds: xr.Dataset) -> Tuple[str, str]:
    lat_candidates = ["lat", "latitude", "y"]
    lon_candidates = ["lon", "longitude", "x"]
    lat_name = next((n for n in lat_candidates if n in ds.coords or n in ds.variables), None)
    lon_name = next((n for n in lon_candidates if n in ds.coords or n in ds.variables), None)
    if lat_name is None or lon_name is None:
        raise KeyError("lat/lon 좌표를 찾지 못했습니다.")
    return lat_name, lon_name


def infer_vars(ds: xr.Dataset, lat_name: str, lon_name: str, requested: Sequence[str] | None = None) -> List[str]:
    if requested:
        missing = [v for v in requested if v not in ds.data_vars]
        if missing:
            raise KeyError(f"요청한 변수가 파일에 없습니다: {missing}")
        return list(requested)

    out: List[str] = []
    for name, da in ds.data_vars.items():
        if da.ndim != 3:
            continue
        dims = set(da.dims)
        if {"time", lat_name, lon_name}.issubset(dims):
            out.append(name)
    if not out:
        raise ValueError("(time, lat, lon) 3차원 변수를 찾지 못했습니다.")
    return out


def build_time_indices(n_time: int, stride: int, max_frames: int) -> List[int]:
    idx = list(range(0, n_time, max(1, stride)))
    if max_frames and len(idx) > max_frames:
        idx = idx[:max_frames]
    return idx


def maybe_downsample_indices(indices: List[int], sample_for_scale: int) -> List[int]:
    if sample_for_scale <= 0 or len(indices) <= sample_for_scale:
        return indices
    # 균등 간격으로 샘플링
    pos = np.linspace(0, len(indices) - 1, sample_for_scale).round().astype(int)
    return [indices[i] for i in sorted(set(pos.tolist()))]


def format_time_value(value) -> str:
    try:
        ts = pd.to_datetime(value)
        if pd.isna(ts):
            return str(value)
        return ts.strftime("%Y-%m")
    except Exception:
        text = str(value)
        return text[:16]


def compute_limits(da: xr.DataArray, indices: Sequence[int], center_zero: str, low_pct: float, high_pct: float) -> Tuple[float, float]:
    sampled = da.isel(time=list(indices)).load().values
    finite = sampled[np.isfinite(sampled)]
    if finite.size == 0:
        return -1.0, 1.0

    low, high = np.nanpercentile(finite, [low_pct, high_pct])

    zero_center = False
    if center_zero == "all":
        zero_center = True
    elif center_zero == "auto":
        zero_center = (low < 0 < high)
    elif center_zero == "none":
        zero_center = False

    if zero_center:
        lim = float(max(abs(low), abs(high)))
        if lim == 0:
            lim = float(np.nanmax(np.abs(finite))) if finite.size else 1.0
        if lim == 0:
            lim = 1.0
        return -lim, lim

    if low == high:
        span = abs(low) if low != 0 else 1.0
        low -= 0.5 * span
        high += 0.5 * span
    return float(low), float(high)


def get_units(da: xr.DataArray) -> str:
    return str(da.attrs.get("units", ""))


def choose_cmap(var_name: str) -> str:
    return DEFAULT_CMAPS.get(var_name, "RdBu_r")


def draw_canvas_to_rgb(fig: plt.Figure) -> np.ndarray:
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    return buf[..., :3].copy()


def prepare_extent(lat: np.ndarray, lon: np.ndarray) -> List[float]:
    return [float(np.nanmin(lon)), float(np.nanmax(lon)), float(np.nanmin(lat)), float(np.nanmax(lat))]


def orient_array_for_imshow(arr: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    out = np.array(arr, copy=False)
    if lat.ndim == 1 and lat.size >= 2 and lat[0] > lat[-1]:
        out = out[::-1, :]
    if lon.ndim == 1 and lon.size >= 2 and lon[0] > lon[-1]:
        out = out[:, ::-1]
    return out


def open_dataset(path: Path) -> xr.Dataset:
    return xr.open_dataset(path, decode_times=True, mask_and_scale=True)


def save_single_var_gif(
    ds: xr.Dataset,
    file_path: Path,
    out_path: Path,
    var_name: str,
    time_indices: Sequence[int],
    scale_indices: Sequence[int],
    low_pct: float,
    high_pct: float,
    center_zero: str,
    fps: int,
    dpi: int,
) -> None:
    lat_name, lon_name = get_lat_lon_names(ds)
    lat = ds[lat_name].values
    lon = ds[lon_name].values
    time_values = ds["time"].values
    da = ds[var_name]
    vmin, vmax = compute_limits(da, scale_indices, center_zero, low_pct, high_pct)
    units = get_units(da)
    cmap = choose_cmap(var_name)
    extent = prepare_extent(lat, lon)

    arr0 = orient_array_for_imshow(da.isel(time=int(time_indices[0])).load().values, lat, lon)

    fig, ax = plt.subplots(figsize=(8.6, 4.8), dpi=dpi)
    im = ax.imshow(
        arr0,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.92)
    cbar.set_label(f"{var_name} [{units}]" if units else var_name)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    title = ax.set_title("")
    file_text = file_path.name

    with imageio.get_writer(out_path, mode="I", fps=fps, loop=0) as writer:
        for tidx in time_indices:
            arr = orient_array_for_imshow(da.isel(time=int(tidx)).load().values, lat, lon)
            im.set_data(arr)
            tstr = format_time_value(time_values[int(tidx)])
            title.set_text(f"{file_text}\n{var_name} ({units}) | {tstr}")
            writer.append_data(draw_canvas_to_rgb(fig))

    plt.close(fig)


def save_panel_gif(
    ds: xr.Dataset,
    file_path: Path,
    out_path: Path,
    var_names: Sequence[str],
    time_indices: Sequence[int],
    scale_indices: Sequence[int],
    low_pct: float,
    high_pct: float,
    center_zero: str,
    fps: int,
    dpi: int,
) -> None:
    lat_name, lon_name = get_lat_lon_names(ds)
    lat = ds[lat_name].values
    lon = ds[lon_name].values
    time_values = ds["time"].values
    extent = prepare_extent(lat, lon)

    n = len(var_names)
    ncols = 3
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(15.5, 4.7 * nrows), dpi=dpi)
    axes = np.array(axes).reshape(nrows, ncols)

    ims = []
    titles = []
    for i, var_name in enumerate(var_names):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        da = ds[var_name]
        vmin, vmax = compute_limits(da, scale_indices, center_zero, low_pct, high_pct)
        units = get_units(da)
        cmap = choose_cmap(var_name)
        arr0 = orient_array_for_imshow(da.isel(time=int(time_indices[0])).load().values, lat, lon)

        im = ax.imshow(
            arr0,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        cb = fig.colorbar(im, ax=ax, shrink=0.86, pad=0.02)
        cb.set_label(units if units else var_name, fontsize=8)
        ax.set_xlabel("Lon")
        ax.set_ylabel("Lat")
        title = ax.set_title(var_name, fontsize=10)
        ims.append((var_name, im))
        titles.append((var_name, title))

    # 남는 subplot 숨김
    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r, c].axis("off")

    suptitle = fig.suptitle("", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    with imageio.get_writer(out_path, mode="I", fps=fps, loop=0) as writer:
        for tidx in time_indices:
            tstr = format_time_value(time_values[int(tidx)])
            suptitle.set_text(f"{file_path.name} | {tstr}")
            for var_name, im in ims:
                arr = orient_array_for_imshow(ds[var_name].isel(time=int(tidx)).load().values, lat, lon)
                im.set_data(arr)
            writer.append_data(draw_canvas_to_rgb(fig))

    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    requested_vars = [v.strip() for v in args.vars.split(",") if v.strip()] if args.vars else None
    files = discover_files(args.input_dir, args.pattern)

    print(f"[INFO] files found: {len(files)}")
    print(f"[INFO] mode={args.mode}, stride={args.stride}, fps={args.fps}")

    for i, nc_path in enumerate(files, start=1):
        print(f"\n[{i}/{len(files)}] processing: {nc_path.name}")
        ds = open_dataset(nc_path)
        try:
            lat_name, lon_name = get_lat_lon_names(ds)
            var_names = infer_vars(ds, lat_name, lon_name, requested_vars)
            n_time = int(ds.sizes["time"])
            time_indices = build_time_indices(n_time, args.stride, args.max_frames)
            scale_indices = maybe_downsample_indices(time_indices, args.sample_for_scale)

            print(f"  variables: {', '.join(var_names)}")
            print(f"  frames: {len(time_indices)} / total_time={n_time}")

            stem = nc_path.stem

            if args.mode in {"panel", "both"}:
                out_panel = args.output_dir / f"{stem}__panel.gif"
                if args.skip_existing and out_panel.exists():
                    print(f"  skip existing: {out_panel.name}")
                else:
                    print(f"  writing panel GIF -> {out_panel}")
                    save_panel_gif(
                        ds=ds,
                        file_path=nc_path,
                        out_path=out_panel,
                        var_names=var_names,
                        time_indices=time_indices,
                        scale_indices=scale_indices,
                        low_pct=args.robust_low,
                        high_pct=args.robust_high,
                        center_zero=args.center_zero,
                        fps=args.fps,
                        dpi=args.dpi,
                    )

            if args.mode in {"single", "both"}:
                for var_name in var_names:
                    out_single = args.output_dir / f"{stem}__{var_name}.gif"
                    if args.skip_existing and out_single.exists():
                        print(f"  skip existing: {out_single.name}")
                        continue
                    print(f"  writing single GIF -> {out_single.name}")
                    save_single_var_gif(
                        ds=ds,
                        file_path=nc_path,
                        out_path=out_single,
                        var_name=var_name,
                        time_indices=time_indices,
                        scale_indices=scale_indices,
                        low_pct=args.robust_low,
                        high_pct=args.robust_high,
                        center_zero=args.center_zero,
                        fps=args.fps,
                        dpi=args.dpi,
                    )
        finally:
            ds.close()

    print("\n[DONE] GIF generation finished.")


if __name__ == "__main__":
    main()
