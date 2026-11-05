"""WeatherBench2 ERA5(1.5° 보존형 평균, 6시간, 1959~2023-01)로 지상 7변수 월평균을 만든다. CDS 대기열 없이 수 분.

  python seas5_monthly/wb2_monthly.py OUTDIR [--y0 1991] [--y1 2022]

만드는 파일: OUTDIR/era5_sfc_wb2_{y0}_{y1}.nc  (1.5° 원격자, CDS 월평균과 같은 변수명·단위)
  t2m d2m msl u10 v10 sst : 6시간 값의 월평균
  tp : 6시간 누적의 월평균 × 4 → m/일 (CDS monthly_averaged_reanalysis 와 같은 단위)
영역은 get_cds.AREA_SFC([60N,60E,10N,180E])를 1.5° 격자로 덮는 범위. 위도 내림차순, 차원 이름 valid_time/latitude/longitude.
"""
import argparse, sys, time
from pathlib import Path
import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))
import get_cds as g

SRC = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
VARS = {"2m_temperature": "t2m", "2m_dewpoint_temperature": "d2m", "mean_sea_level_pressure": "msl",
        "10m_u_component_of_wind": "u10", "10m_v_component_of_wind": "v10", "sea_surface_temperature": "sst",
        "total_precipitation_6hr": "tp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("--y0", type=int, default=1991); ap.add_argument("--y1", type=int, default=2022)
    ap.add_argument("--what", default="sfc", choices=["sfc", "pl"])
    a = ap.parse_args()
    out = Path(a.out)
    if a.what == "pl":
        return pl(out, a.y0, a.y1)
    f = out / f"era5_sfc_wb2_{a.y0}_{a.y1}.nc"
    if f.exists():
        print("skip", f); return
    n, w, s, e = g.AREA_SFC
    ds = xr.open_zarr(SRC, storage_options=dict(token="anon"))
    ds = ds[list(VARS)].sel(time=slice(f"{a.y0}-01-01", f"{a.y1}-12-31T18"),
                             latitude=slice(s - 1.5, n + 1.5), longitude=slice(w - 1.5, e + 1.5))
    t0 = time.time()
    res = {}
    for k, v in VARS.items():
        m = ds[k].resample(time="MS").mean().load()
        if v == "tp":
            m = m * 4.0
        res[v] = m.transpose("time", "latitude", "longitude").astype("float32")
        print(f"  {v} {time.time() - t0:.0f}s", flush=True)
    o = xr.Dataset(res).rename(time="valid_time").sortby("latitude", ascending=False)
    o.attrs.update(source="WeatherBench2 ERA5 1.5deg conservative, 6-hourly -> monthly mean", tp_units="m/day")
    o.to_netcdf(f)
    print("ok", f, dict(o.sizes), flush=True)


def pl(out, y0, y1):
    """200 hPa z·u·v [AREA_PL200], 500 hPa z 북반구 [AREA_NH] 월평균 (1.5°)."""
    ds = xr.open_zarr(SRC, storage_options=dict(token="anon"))
    ds = ds.sel(time=slice(f"{y0}-01-01", f"{y1}-12-31T18"))
    jobs = {f"era5_pl200_wb2_{y0}_{y1}.nc": (dict(geopotential="z", u_component_of_wind="u", v_component_of_wind="v"), 200, g.AREA_PL200),
            f"era5_z500_nh_wb2_{y0}_{y1}.nc": (dict(geopotential="z"), 500, g.AREA_NH)}
    for fn, (vs, lev, (n, w, s, e)) in jobs.items():
        f = out / fn
        if f.exists():
            print("skip", fn); continue
        d = ds[list(vs)].sel(level=[lev], latitude=slice(s - 1.5, n + 1.5))
        if w >= 0:
            d = d.sel(longitude=slice(w - 1.5, e + 1.5))
        m = d.resample(time="MS").mean().load().rename(vs).rename(time="valid_time", level="pressure_level")
        if w < 0:
            m = m.assign_coords(longitude=((m.longitude + 180) % 360) - 180).sortby("longitude")
        m = m.transpose("valid_time", "pressure_level", "latitude", "longitude").sortby("latitude", ascending=False).astype("float32")
        m.attrs.update(source="WeatherBench2 ERA5 1.5deg conservative, 6-hourly -> monthly mean")
        m.to_netcdf(f); print("ok", fn, dict(m.sizes), flush=True)


if __name__ == "__main__":
    main()
