"""ECMWF ARCO 시간별 ERA5 → WeatherBench2와 같은 1.5° 보존형 격자의 월평균 (WB2가 끝나는 2023년 이후 구간용).

  CDSAPI_KEY=... python seas5_monthly/arco_wb2grid.py OUTDIR --years 2023-2026 [--what sfc,pl] [--workers 24]

만드는 파일 (연도별, wb2_monthly.py 출력과 같은 격자·변수명·단위):
  parts/era5_sfc_wb2grid_{Y}.nc      t2m d2m msl u10 v10 sst (6시간 표본), tp (매시 누적 → m/일)
  parts/era5_pl200_wb2grid_{Y}.nc    z u v @200
  parts/era5_z500_nh_wb2grid_{Y}.nc  z @500
1.5° 보존형: 0.25° 점을 셀(±0.75°) 안에서 가장자리 반가중 [.5,1,1,1,1,1,.5]/6 으로 평균 (WB2 conservative 와 거의 같음).
그 달 자료가 끝까지 없으면 있는 날만 평균하고 attrs['days_used']에 남긴다.
"""
import argparse, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))
import get_cds as g
from arco_monthly import open_store, PL, SFC

W = np.array([.5, 1, 1, 1, 1, 1, .5]) / 6.0
LAT15 = np.arange(-90, 90.01, 1.5); LON15 = np.arange(0, 360, 1.5)


def to15(a):
    """a(..., 721 위도 -90→90, 1440 경도 -180..179.75) → (..., 121, 240) 위도 -90→90, 경도 0..358.5"""
    a = np.roll(a, 720, axis=-1)                          # 경도 0..359.75
    ci = np.arange(0, 721, 6)
    la = sum(w * a[..., np.clip(ci + k - 3, 0, 720), :] for k, w in enumerate(W))
    cj = np.arange(0, 1440, 6)
    return sum(w * la[..., (cj + k - 3) % 1440] for k, w in enumerate(W))


def month_mean(da, idx, workers):
    def get(i):
        for k in range(6):
            try:
                return da.isel(time=i).values.astype("float64")
            except Exception:
                time.sleep(3 * (k + 1))
        raise RuntimeError(f"read fail {i}")
    with ThreadPoolExecutor(workers) as ex:
        acc = None
        for x in ex.map(get, idx):
            acc = x if acc is None else acc + x
    return acc / len(idx)


def run(ds, specs, year, months_avail, workers, step6):
    """specs: [(이름, 변수, 기압면 or None, 배율)] → {이름: (nmon, 121, 240)}, days"""
    tv = ds.time.values
    res = {s[0]: [] for s in specs}; days = []
    for m in range(1, 13):
        t0 = np.datetime64(f"{year}-{m:02d}-01", "D"); t1 = (t0.astype("datetime64[M]") + np.timedelta64(1, "M")).astype("datetime64[D]")
        a, b = np.searchsorted(tv, t0), np.searchsorted(tv, t1)
        if a >= b:
            break
        allh = np.arange(a, b)
        for name, v, lev, scale, hourly in specs:
            idx = allh if hourly else [i for i in allh if tv[i].astype("datetime64[h]").astype(int) % 6 == 0]
            da = ds[v] if lev is None else ds[v].isel(pressureLevel=list(ds.pressureLevel.values).index(lev))
            res[name].append(to15(month_mean(da, idx, workers)) * scale)
        days.append(len({str(t)[:10] for t in tv[a:b]}))
        print(f"  {year}-{m:02d} {[s[0] for s in specs]} days={days[-1]}", flush=True)
    return {k: np.array(v) for k, v in res.items()}, days


def save(fields, year, days, f, levels=None, area=None, src=""):
    vt = np.array([np.datetime64(f"{year}-{m:02d}-01") for m in range(1, len(days) + 1)], dtype="datetime64[ns]")
    dims = ("valid_time", "latitude", "longitude") if levels is None else ("valid_time", "pressure_level", "latitude", "longitude")
    dv = {k: (dims, (v if levels is None else v[:, None]).astype("float32")) for k, v in fields.items()}
    co = dict(valid_time=vt, latitude=LAT15, longitude=LON15)
    if levels is not None:
        co["pressure_level"] = np.array(levels, float)
    ds = xr.Dataset(dv, coords=co)
    n, w, s, e = area
    if w < 0:
        ds = ds.assign_coords(longitude=((ds.longitude + 180) % 360) - 180).sortby("longitude")
    ds = ds.sel(latitude=slice(s - 1.5, n + 1.5), longitude=slice(w - 1.5, e + 1.5)).sortby("latitude", ascending=False)
    ds.attrs.update(source=src, days_used=str(days), grid="1.5deg conservative (WB2 equiangular_with_poles)")
    ds.to_netcdf(f)
    print("ok", f.name, dict(ds.sizes), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("--years", default="2023-2026")
    ap.add_argument("--what", default="sfc,pl"); ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()
    out = Path(a.out) / "parts"; out.mkdir(parents=True, exist_ok=True)
    y0, y1 = map(int, a.years.split("-"))
    for Y in range(y0, y1 + 1):
        if "pl" in a.what:
            f2, fn = out / f"era5_pl200_wb2grid_{Y}.nc", out / f"era5_z500_nh_wb2grid_{Y}.nc"
            if not (f2.exists() and fn.exists()):
                ds = open_store(PL)
                r, d = run(ds, [("z200", "z", 200, 1, False), ("u200", "u", 200, 1, False), ("v200", "v", 200, 1, False),
                                ("z500", "z", 500, 1, False)], Y, None, a.workers, True)
                save({"z": r["z200"], "u": r["u200"], "v": r["v200"]}, Y, d, f2, [200], g.AREA_PL200, "ECMWF ARCO ERA5 6-hourly")
                save({"z": r["z500"]}, Y, d, fn, [500], g.AREA_NH, "ECMWF ARCO ERA5 6-hourly")
        if "sfc" in a.what:
            f = out / f"era5_sfc_wb2grid_{Y}.nc"
            if not f.exists():
                ds = open_store(SFC)
                specs = [(v, v, None, 1, False) for v in ("t2m", "d2m", "msl", "u10", "v10", "sst")] + [("tp", "tp", None, 24.0, True)]
                r, d = run(ds, specs, Y, None, a.workers, True)
                save(r, Y, d, f, None, g.AREA_SFC, "ECMWF ARCO ERA5 (6-hourly; tp hourly)")


if __name__ == "__main__":
    main()
