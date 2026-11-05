"""ECMWF ARCO(Zarr) 시간별 ERA5로 월평균을 직접 만든다. CDS 월평균 대기열을 피하는 경로.

  CDSAPI_KEY=... python seas5_monthly/arco_monthly.py OUTDIR --year 2026 [--months 1-10] [--workers 16]

만드는 파일(get_cds.py 의 ERA5 부분 파일과 같은 이름·영역·변수명, 1° 격자):
  parts/era5_pl_{Y}.nc        z t q u v @500·850   [60N,60E,0,180]
  parts/era5_pl200_{Y}.nc     z u v @200           [60N,20E,10N,180]
  parts/era5_z500_nh_{Y}.nc   z @500               [90N,180W,20N,180E]
  parts/era5_sfc_{Y}.nc       t2m d2m msl tp u10 v10 sst [60N,60E,10N,180E]
기압면은 6시간 간격(ARCO 원자료 간격), 지상은 6시간 간격(강수 tp만 매시 누적 → m/일).
CDS monthly_averaged_reanalysis 와 거의 같지만 시간 표본이 달라 일주기가 큰 지상 변수에서 약간 차이가 날 수 있다.
그 달 자료가 끝까지 없으면(예: 말일 미공개) 있는 날만 평균하고 attrs['days_used']에 남긴다.
"""
import argparse, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))
import get_cds as g

B = "https://arco.datastores.ecmwf.int"
PL = f"{B}/cadl-arco-time-048/arco/reanalysis_era5_pressure_levels/pl/timeChunked.zarr"
SFC = f"{B}/cadl-arco-time-002/arco/reanalysis_era5_single_levels/sfc/timeChunked.zarr"


def open_store(url):
    from obstore.store import HTTPStore
    from zarr.storage import ObjectStore
    h = HTTPStore(url, client_options={"default_headers": {"Authorization": f"Bearer {os.environ['CDSAPI_KEY']}"},
                                       "timeout": "300s"})
    return xr.open_zarr(ObjectStore(h, read_only=True), consolidated=True, chunks=None)   # chunks=None 필수(OOM 방지)


def monthly(da, times, months, year, workers, label):
    """da(time, lat, lon) 0.25° → 1°(위도 90→-90, 경도 0..359) 월평균 (month, 181, 360)."""
    tv = da.time.values
    out, days = [], []
    for m in months:
        t0 = np.datetime64(f"{year}-{m:02d}-01"); t1 = t0.astype("datetime64[M]") + 1
        idx = [i for i in times if t0 <= tv[i] < t1]
        if not idx:
            break

        def get(i):
            for k in range(5):
                try:
                    x = da.isel(time=i).values[::4, ::4]
                    return x
                except Exception:
                    time.sleep(2 * (k + 1))
            raise
        with ThreadPoolExecutor(workers) as ex:
            acc = sum(ex.map(get, idx)).astype("float64")
        out.append(acc / len(idx)); days.append(len({str(tv[i])[:10] for i in idx}))
        print(f"  {label} {year}-{m:02d} n={len(idx)}", flush=True)
    return np.array(out), days


def to_ds(fields, lat, lon, year, nmon, levels=None):
    vt = np.array([np.datetime64(f"{year}-{m:02d}-01") for m in range(1, nmon + 1)], dtype="datetime64[ns]")
    dv = {}
    for k, a in fields.items():
        if levels is None:
            dv[k] = (("valid_time", "latitude", "longitude"), a.astype("float32"))
        else:
            dv[k] = (("valid_time", "pressure_level", "latitude", "longitude"), a.astype("float32"))
    co = dict(valid_time=vt, latitude=lat, longitude=lon)
    if levels is not None:
        co["pressure_level"] = np.array(levels, dtype="float64")
    return xr.Dataset(dv, coords=co)


def cut(ds, area):
    n, w, s, e = area
    lon = ds.longitude.values
    if w < 0:   # -180..180 표기
        ds = ds.assign_coords(longitude=((lon + 180) % 360) - 180).sortby("longitude")
        ds = ds.sel(longitude=slice(w, e))
    else:
        ds = ds.sel(longitude=slice(w, e))
    return ds.sel(latitude=slice(n, s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("--year", type=int, default=time.gmtime().tm_year)
    ap.add_argument("--months", default="1-10"); ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--only", choices=["pl", "sfc"])
    a = ap.parse_args()
    out = Path(a.out); (out / "parts").mkdir(parents=True, exist_ok=True)
    m0, m1 = map(int, a.months.split("-")); months = list(range(m0, m1 + 1)); Y = a.year
    lat = np.arange(90, -90.1, -1.0); lon = np.arange(0, 360, 1.0)   # 원자료 경도 -180.. → 아래에서 0..359로 재배열
    def roll(arr):   # 원자료(위도 -90→90, 경도 -180..179) → 위도 90→-90, 경도 0..359
        return np.roll(arr[..., ::-1, :], 180, axis=-1)

    if a.only != "sfc":
        f_pl, f_200, f_nh = (out / f"parts/era5_pl_{Y}.nc", out / f"parts/era5_pl200_{Y}.nc", out / f"parts/era5_z500_nh_{Y}.nc")
        if not (f_pl.exists() and f_200.exists() and f_nh.exists()):
            ds = open_store(PL)            # 위도 -90→90 오름차순: 아래 roll에서 뒤집음
            times = list(np.where(ds.time.values >= np.datetime64(f"{Y}-01-01"))[0])
            levs = list(ds.pressureLevel.values)
            res = {}
            for v, ls in (("z", [200, 500, 850]), ("t", [500, 850]), ("q", [500, 850]), ("u", [200, 500, 850]), ("v", [200, 500, 850])):
                for L in ls:
                    arr, days = monthly(ds[v].isel(pressureLevel=levs.index(L)), times, months, Y, a.workers, f"{v}{L}")
                    res[(v, L)] = roll(arr)
            nm = len(days)
            pl = to_ds({v: np.stack([res[(v, 850)], res[(v, 500)]], 1) for v in ("z", "t", "q", "u", "v")}, lat, lon, Y, nm, [850, 500])
            cut(pl, g.AREA_PL).assign_attrs(source="ECMWF ARCO ERA5 6-hourly", days_used=str(days)).to_netcdf(f_pl)
            p2 = to_ds({v: res[(v, 200)][:, None] for v in ("z", "u", "v")}, lat, lon, Y, nm, [200])
            cut(p2, g.AREA_PL200).assign_attrs(source="ECMWF ARCO ERA5 6-hourly", days_used=str(days)).to_netcdf(f_200)
            nh = to_ds({"z": res[("z", 500)][:, None]}, lat, lon, Y, nm, [500])
            cut(nh, g.AREA_NH).assign_attrs(source="ECMWF ARCO ERA5 6-hourly", days_used=str(days)).to_netcdf(f_nh)
            print("ok pl", days, flush=True)

    if a.only != "pl":
        f = out / f"parts/era5_sfc_{Y}.nc"
        if not f.exists():
            ds = open_store(SFC)
            tv = ds.time.values
            allh = list(np.where(tv >= np.datetime64(f"{Y}-01-01"))[0])
            syn = [i for i in allh if tv[i].astype("datetime64[h]").astype(int) % 6 == 0]
            res = {}
            for v in ("t2m", "d2m", "msl", "u10", "v10", "sst"):
                arr, days = monthly(ds[v], syn, months, Y, a.workers, v)
                res[v] = roll(arr)
            arr, days = monthly(ds["tp"], allh, months, Y, a.workers, "tp")
            res["tp"] = roll(arr) * 24.0            # 시간 누적 평균 × 24 → m/일 (CDS 월평균 tp와 같은 단위)
            sf = to_ds(res, lat, lon, Y, len(days))
            cut(sf, g.AREA_SFC).assign_attrs(source="ECMWF ARCO ERA5 (6-hourly; tp hourly)", days_used=str(days)).to_netcdf(f)
            print("ok sfc", days, flush=True)


if __name__ == "__main__":
    main()
