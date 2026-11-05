"""SEAS5 요청 묶기: CDS 사용자당 대기열 한도 때문에 초기화·재예보/실황마다 6건 → 3건으로 줄인다.
  slall : 지상 7변수, [60N,50E,10S,120W]  → _sfc / _sst_nino34 / _sst_io 로 잘라 저장
  plall : z·t·q·u·v @200·500·850, [60N,20E,0,180] → _pl(500·850, z·t·q·u·v, 60E~) / _pl200(200, z·u·v, 10N~)
  z500_nh 는 그대로.  잘라낸 파일 이름·영역은 get_cds.py 와 같아 이후 코드는 바뀌지 않는다.
"""
from pathlib import Path
import get_cds as g


def requests(init_months, fc_year):
    out = []
    for im in init_months:
        mm = f"{im:02d}"
        for tag, years in (("hind", g.HIND_YEARS), (f"fc{fc_year}", [str(fc_year)])):
            base = dict(originating_centre="ecmwf", system="51", product_type=["monthly_mean"], year=years,
                        month=[mm], leadtime_month=g.LEADS, data_format="netcdf")
            out += [
                (f"combined/seas5_{tag}_{mm}_slall.nc", "seasonal-monthly-single-levels",
                 dict(base, variable=g.SFC_VARS, area=[60, 50, -10, 240])),
                (f"combined/seas5_{tag}_{mm}_plall.nc", "seasonal-monthly-pressure-levels",
                 dict(base, variable=g.PL_VARS, pressure_level=["200", "500", "850"], area=[60, 20, 0, 180])),
                (f"seas5_{tag}_{mm}_z500_nh.nc", "seasonal-monthly-pressure-levels",
                 dict(base, **g.SEAS5_EXT["z500_nh"][1])),
            ]
    return out


def _box(ds, n, w, s, e):
    import numpy as np
    lon = ds.longitude % 360
    ds = ds.assign_coords(longitude=lon).sortby("longitude")
    w3, e3 = w % 360, e % 360
    if e == 180: e3 = 180
    ds = ds.sel(longitude=slice(w3, e3))
    lat = ds.latitude
    ds = ds.sel(latitude=slice(n, s) if lat[0] > lat[-1] else slice(s, n))
    if w < 0:                                    # 원래 요청처럼 -180~180 경도로
        ds = ds.assign_coords(longitude=((ds.longitude + 180) % 360) - 180)
    return ds


def split(out, name):
    """combined 파일을 get_cds.py 이름의 파일들로 나눈다. 이미 있으면 건너뜀."""
    import xarray as xr
    out = Path(out); src = out / name
    stem = Path(name).name.replace("_slall.nc", "").replace("_plall.nc", "")
    ds = xr.open_dataset(src)
    if name.endswith("_slall.nc"):
        jobs = {"_sfc": (list(ds.data_vars), g.AREA_SFC),
                "_sst_nino34": (["sst"], g.AREA_NINO34), "_sst_io": (["sst"], g.AREA_IO)}
        for suf, (vs, a) in jobs.items():
            f = out / f"{stem}{suf}.nc"
            if not f.exists():
                _box(ds[vs], *a).load().to_netcdf(f)
    else:
        lev = "pressure_level" if "pressure_level" in ds.dims else "isobaricInhPa"
        f = out / f"{stem}_pl.nc"
        if not f.exists():
            _box(ds.sel({lev: [500, 850]}), *g.AREA_PL).load().to_netcdf(f)
        f = out / f"{stem}_pl200.nc"
        if not f.exists():
            v = [x for x in ("z", "u", "v") if x in ds]
            _box(ds[v].sel({lev: [200]}), *g.AREA_PL200).load().to_netcdf(f)
    return [p.name for p in sorted(out.glob(stem + "_*.nc"))]
