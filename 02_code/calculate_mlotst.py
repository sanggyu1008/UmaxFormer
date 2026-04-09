#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr

DEPTH_DIM_CANDIDATES = ["level", "lev", "zlev", "depth", "olevel", "z", "deptht", "st_ocean"]
TIME_DIM_CANDIDATES = ["time", "t"]
LAT_COORD_CANDIDATES = ["lat", "latitude", "yt_ocean", "y", "nav_lat"]
LON_COORD_CANDIDATES = ["lon", "longitude", "xt_ocean", "x", "nav_lon"]


def _default_project_root() -> Path:
    for cand in (
        Path("/home/sanggyu1008/project/01_ENSO"),
        Path("/mnt/d/project/01_ENSO"),
    ):
        if cand.exists():
            return cand
    return Path.cwd()


PROJECT_ROOT = _default_project_root()
DEFAULT_BASE_DIR = PROJECT_ROOT / "01_data/01_raw/soda"


def append_history(ds: xr.Dataset, message: str) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    entry = f"{now}: {message}"
    old = ds.attrs.get("history")
    ds.attrs["history"] = f"{entry}\n{old}" if old else entry


def open_ds(fp: Path, decode_times: bool, use_dask: bool) -> xr.Dataset:
    if not fp.exists():
        raise FileNotFoundError(str(fp))
    chunks = {} if use_dask else None
    if decode_times:
        time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
        return xr.open_dataset(fp, decode_times=time_coder, chunks=chunks)
    return xr.open_dataset(fp, decode_times=False, chunks=chunks)


def _is_numeric_1d_coord(ds: xr.Dataset, dim: str) -> bool:
    if dim not in ds.coords:
        return False
    c = ds[dim]
    if c.ndim != 1:
        return False
    return np.issubdtype(np.asarray(c.values).dtype, np.number)


def find_depth_dim(ds: xr.Dataset, da: xr.DataArray) -> str:
    for d in da.dims:
        dl = d.lower()
        if dl in ("lev", "level", "depth", "z", "st_ocean", "olevel", "zlev", "deptht") and _is_numeric_1d_coord(ds, d):
            return d

    best: tuple[int, str] | None = None
    for d in da.dims:
        if d.lower() in ("time", "t"):
            continue
        if not _is_numeric_1d_coord(ds, d):
            continue
        attrs = ds[d].attrs
        units = str(attrs.get("units", "")).lower()
        axis = str(attrs.get("axis", "")).upper()
        stdn = str(attrs.get("standard_name", "")).lower()
        longn = str(attrs.get("long_name", "")).lower()

        score = 0
        if axis == "Z":
            score += 5
        if "depth" in stdn or "depth" in longn:
            score += 4
        if any(u in units for u in ("m", "meter", "metre", "cm")):
            score += 3

        if best is None or score > best[0]:
            best = (score, d)
    if best is None:
        raise ValueError(f"depth dimension not found for dims={da.dims}")
    return best[1]


def find_time_dim(da: xr.DataArray) -> str:
    for d in TIME_DIM_CANDIDATES:
        if d in da.dims:
            return d
    for d in da.dims:
        if d in da.coords:
            units = str(da[d].attrs.get("units", "")).lower()
            if "since" in units or "days" in units or "hours" in units:
                return d
    raise ValueError(f"time dimension not found for dims={da.dims}")


def _find_lat_lon(da: xr.DataArray) -> tuple[str, str]:
    lat = None
    lon = None
    for c in LAT_COORD_CANDIDATES:
        if c in da.coords:
            lat = c
            break
    for c in LON_COORD_CANDIDATES:
        if c in da.coords:
            lon = c
            break
    if lat is None or lon is None:
        raise ValueError("lat/lon coords not found")
    return lat, lon


def depth_coord_to_m(ds: xr.Dataset, zdim: str) -> xr.Dataset:
    z = ds[zdim]
    units = str(z.attrs.get("units", "")).lower()
    if "cm" in units:
        z_m = (z.astype("float64") / 100.0).assign_attrs(dict(z.attrs))
        z_m.attrs["units"] = "m"
        return ds.assign_coords({zdim: z_m})
    return ds


def ensure_depth_ascending(da: xr.DataArray, zdim: str) -> xr.DataArray:
    z = np.asarray(da[zdim].values, dtype="float64")
    if z.size >= 2 and np.nanmean(np.diff(z)) < 0:
        return da.sortby(zdim)
    return da


def maybe_chunk_4d(
    da: xr.DataArray,
    zdim: str,
    time_dim: str,
    time_chunk: int,
    lat_chunk: int,
    lon_chunk: int,
    use_dask: bool,
) -> xr.DataArray:
    if not use_dask:
        return da

    chunks: dict[str, int] = {zdim: -1}
    if time_dim in da.dims and int(time_chunk) > 0:
        chunks[time_dim] = int(time_chunk)

    try:
        lat_name, lon_name = _find_lat_lon(da)
        if lat_name in da.dims and int(lat_chunk) > 0:
            chunks[lat_name] = int(lat_chunk)
        if lon_name in da.dims and int(lon_chunk) > 0:
            chunks[lon_name] = int(lon_chunk)
    except Exception:
        pass

    return da.chunk(chunks)


def to_degC_if_kelvin(T: xr.DataArray) -> xr.DataArray:
    u = str(T.attrs.get("units", "")).strip().lower()
    if u in ("k", "kelvin") or ("kelvin" in u):
        out = T - 273.15
        out.attrs = dict(T.attrs)
        out.attrs["units"] = "degC"
        return out
    return T


def carry_aux_vars(ds: xr.Dataset, out: xr.Dataset, main_var: str) -> xr.Dataset:
    out_dims = set(out[main_var].dims)
    for name, var in ds.variables.items():
        if name in out.variables or name == main_var or name in ds.coords:
            continue
        if set(var.dims).issubset(out_dims):
            out[name] = var
    return out


def preserve_coord_metadata(src: xr.Dataset, dst: xr.Dataset, coord_names: list[str]) -> xr.Dataset:
    for name in coord_names:
        if name not in src.coords:
            continue
        dst = dst.assign_coords({name: src[name]})
        dst[name].attrs = dict(src[name].attrs)
        try:
            dst[name].encoding = dict(src[name].encoding)
        except Exception:
            pass
    return dst


def maybe_rename_to_match(da: xr.DataArray, old: str, new: str) -> xr.DataArray:
    if old == new:
        return da
    if old in da.dims or old in da.coords:
        return da.rename({old: new})
    return da


def assert_and_harmonize_coords(ref: xr.DataArray, other: xr.DataArray, coord_names: list[str]) -> xr.DataArray:
    updates: dict[str, xr.DataArray] = {}
    for name in coord_names:
        if name not in ref.coords or name not in other.coords:
            continue
        a = np.asarray(ref[name].values)
        b = np.asarray(other[name].values)
        if a.shape != b.shape:
            raise ValueError(f"Coordinate shape mismatch for {name}: {a.shape} vs {b.shape}")
        if np.issubdtype(a.dtype, np.number) and np.issubdtype(b.dtype, np.number):
            if not np.allclose(a, b, equal_nan=True):
                raise ValueError(f"Coordinate values differ for {name}")
            updates[name] = ref[name]
        else:
            if not np.array_equal(a, b):
                raise ValueError(f"Coordinate values differ for {name}")
            updates[name] = ref[name]
    if updates:
        other = other.assign_coords(updates)
    return other


def mld_columnwise_teos10(
    SP: xr.DataArray,
    pt: xr.DataArray,
    zdim: str,
    ref_depth: float,
    delta_sigma: float,
    gsw_float64: bool,
) -> xr.DataArray:
    try:
        import gsw
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "The 'gsw' package is required for make_mlotst.py. Install it first, e.g. 'pip install gsw'."
        ) from e

    lat_name, lon_name = _find_lat_lon(SP)
    pt_c = to_degC_if_kelvin(pt)

    z = SP[zdim].astype("float64")
    if hasattr(z.data, "chunks"):
        z = z.chunk({zdim: -1})

    lat2d, lon2d = xr.broadcast(SP[lat_name], SP[lon_name])

    if hasattr(SP.data, "chunks"):
        ref_chunks = {}
        for d in SP.dims:
            if d == zdim:
                continue
            if d in getattr(SP, "chunksizes", {}):
                ref_chunks[d] = SP.chunksizes[d][0]
        if ref_chunks:
            SP = SP.chunk({**ref_chunks, zdim: -1})
            pt_c = pt_c.chunk({**ref_chunks, zdim: -1})
            lat_chunk_map = {d: SP.chunksizes[d][0] for d in lat2d.dims if d in getattr(SP, "chunksizes", {})}
            if lat_chunk_map:
                lat2d = lat2d.chunk(lat_chunk_map)
                lon2d = lon2d.chunk(lat_chunk_map)

    out_dtype = np.float64 if gsw_float64 else np.float32
    ref_depth_f = float(ref_depth)
    delta_f = float(delta_sigma)

    def _mld_1col(sp_z: np.ndarray, pt_z: np.ndarray, z1d: np.ndarray, lat: float, lon: float) -> np.ndarray:
        sp_z = np.asarray(sp_z, dtype="float64", order="C")
        pt_z = np.asarray(pt_z, dtype="float64", order="C")
        z1d = np.asarray(z1d, dtype="float64", order="C")

        m = np.isfinite(sp_z) & np.isfinite(pt_z) & np.isfinite(z1d)
        if m.sum() < 2:
            return np.asarray(np.nan, dtype=out_dtype)

        sp = sp_z[m]
        ptv = pt_z[m]
        zz = z1d[m]

        if ref_depth_f <= zz.min() or ref_depth_f >= zz.max():
            return np.asarray(np.nan, dtype=out_dtype)

        p = gsw.p_from_z(-zz, float(lat))
        SA = gsw.SA_from_SP(sp, p, float(lon), float(lat))
        CT = gsw.CT_from_pt(SA, ptv)
        sig0 = gsw.sigma0(SA, CT)

        sig_ref = np.interp(ref_depth_f, zz, sig0)
        thr = sig_ref + delta_f
        diff = sig0 - thr

        idx = np.where(diff >= 0)[0]
        if idx.size == 0:
            return np.asarray(np.nan, dtype=out_dtype)

        i = int(idx[0])
        if i == 0:
            return np.asarray(float(zz[0]), dtype=out_dtype)

        d0, d1 = diff[i - 1], diff[i]
        z0, z1 = zz[i - 1], zz[i]
        if (not np.isfinite(d0)) or (not np.isfinite(d1)) or (d1 == d0):
            return np.asarray(float(z1), dtype=out_dtype)

        mld = z0 + (0.0 - d0) * (z1 - z0) / (d1 - d0)
        return np.asarray(float(mld), dtype=out_dtype)

    mld = xr.apply_ufunc(
        _mld_1col,
        SP,
        pt_c,
        z,
        lat2d,
        lon2d,
        input_core_dims=[[zdim], [zdim], [zdim], [], []],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[out_dtype],
        dask_gufunc_kwargs={"allow_rechunk": False},
    )

    mld.name = "mlotst"
    mld.attrs.update(
        {
            "units": "m",
            "long_name": "Mixed layer depth (density criterion)",
            "comment": (
                f"MLD where sigma0(z) >= sigma0({ref_depth_f:g} m) + {delta_f:g} kg m-3; "
                "sigma0 from TEOS-10 (gsw)."
            ),
        }
    )
    return mld


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Create mlotst from temperature and salinity using a TEOS-10 density criterion."
    )
    p.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    p.add_argument("--temp-file", default="SODA_temp.nc")
    p.add_argument("--salt-file", default="SODA_salt.nc")
    p.add_argument("--out-file", default="SODA_mlotst.nc")
    p.add_argument("--temp-var", default="temp")
    p.add_argument("--salt-var", default="salt")
    p.add_argument("--output-var", default="mlotst")
    p.add_argument("--source-name", default="SODA2.2.4")
    p.add_argument("--decode-times", action="store_true")
    p.add_argument("--no-dask", action="store_true")
    p.add_argument("--time-chunk", default=12, type=int)
    p.add_argument("--lat-chunk", default=40, type=int)
    p.add_argument("--lon-chunk", default=80, type=int)
    p.add_argument("--mld-zmax", default=1000.0, type=float)
    p.add_argument("--ref-depth", default=10.0, type=float)
    p.add_argument("--delta-sigma", default=0.03, type=float)
    p.add_argument("--gsw-float64", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    use_dask = not args.no_dask

    base_dir = args.base_dir.expanduser().resolve()
    if not base_dir.exists():
        raise SystemExit(f"Base directory does not exist: {base_dir}")

    temp_fp = base_dir / args.temp_file
    salt_fp = base_dir / args.salt_file
    out_fp = base_dir / args.out_file

    if out_fp.exists() and not args.overwrite:
        print(f"[skip] {out_fp}")
        return

    print(f"[TEMP] {temp_fp}")
    print(f"[SALT] {salt_fp}")
    print(f"[OUT ] {out_fp}")

    dsT = open_ds(temp_fp, decode_times=args.decode_times, use_dask=use_dask)
    dsS = open_ds(salt_fp, decode_times=args.decode_times, use_dask=use_dask)

    if args.temp_var not in dsT.data_vars:
        raise KeyError(f"TEMP_VAR not found: {args.temp_var}")
    if args.salt_var not in dsS.data_vars:
        raise KeyError(f"SALT_VAR not found: {args.salt_var}")

    zdimT = find_depth_dim(dsT, dsT[args.temp_var])
    zdimS = find_depth_dim(dsS, dsS[args.salt_var])

    dsT = depth_coord_to_m(dsT, zdimT)
    dsS = depth_coord_to_m(dsS, zdimS)

    T = dsT[args.temp_var]
    S = dsS[args.salt_var]

    time_dim_T = find_time_dim(T)
    time_dim_S = find_time_dim(S)
    lat_T, lon_T = _find_lat_lon(T)
    lat_S, lon_S = _find_lat_lon(S)

    if zdimS != zdimT:
        S = maybe_rename_to_match(S, zdimS, zdimT)
    if time_dim_S != time_dim_T:
        S = maybe_rename_to_match(S, time_dim_S, time_dim_T)
    if lat_S != lat_T:
        S = maybe_rename_to_match(S, lat_S, lat_T)
    if lon_S != lon_T:
        S = maybe_rename_to_match(S, lon_S, lon_T)

    zdim = zdimT
    time_dim = time_dim_T

    coord_names = [zdim, time_dim, lat_T, lon_T]
    S = assert_and_harmonize_coords(T, S, coord_names)
    T, S = xr.align(T, S, join="exact")

    T = ensure_depth_ascending(T, zdim)
    S = ensure_depth_ascending(S, zdim)

    T = T.sel({zdim: slice(0.0, float(args.mld_zmax))})
    S = S.sel({zdim: slice(0.0, float(args.mld_zmax))})

    T = maybe_chunk_4d(T, zdim, time_dim, args.time_chunk, args.lat_chunk, args.lon_chunk, use_dask)
    S = maybe_chunk_4d(S, zdim, time_dim, args.time_chunk, args.lat_chunk, args.lon_chunk, use_dask)

    mld = mld_columnwise_teos10(
        SP=S.astype("float32"),
        pt=T.astype("float32"),
        zdim=zdim,
        ref_depth=args.ref_depth,
        delta_sigma=args.delta_sigma,
        gsw_float64=args.gsw_float64,
    )
    mld.name = args.output_var

    out = mld.to_dataset(name=args.output_var)
    out = preserve_coord_metadata(dsT, out, [time_dim_T, lat_T, lon_T])
    out.attrs = dict(dsT.attrs)
    out.attrs["variable_id"] = args.output_var
    out = carry_aux_vars(dsT, out, args.output_var)

    out[args.output_var].attrs.update(
        {
            "units": "m",
            "source_temperature_file": temp_fp.name,
            "source_salinity_file": salt_fp.name,
            "source_temperature_variable": args.temp_var,
            "source_salinity_variable": args.salt_var,
            "source_dataset": args.source_name,
            "ref_depth": float(args.ref_depth),
            "delta_sigma": float(args.delta_sigma),
            "mld_zmax": float(args.mld_zmax),
            "density_algorithm": "TEOS-10 sigma0 via gsw",
        }
    )
    append_history(
        out,
        (
            f"Created {args.output_var} from {temp_fp.name} and {salt_fp.name} "
            f"(ref_depth={args.ref_depth:g} m, delta_sigma={args.delta_sigma:g} kg m-3, "
            f"mld_zmax={args.mld_zmax:g} m)"
        ),
    )

    out_fp.parent.mkdir(parents=True, exist_ok=True)
    encoding = {
        args.output_var: {
            "zlib": True,
            "complevel": 4,
            "shuffle": True,
            "dtype": "float32",
            "_FillValue": np.float32(1.0e20),
        }
    }
    out.to_netcdf(out_fp, encoding=encoding)
    print(f"[OK] wrote: {out_fp}")


if __name__ == "__main__":
    main()
