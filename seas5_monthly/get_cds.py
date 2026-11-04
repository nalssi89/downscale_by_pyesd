"""C3S CDS 자료 입수 (SEAS5 월별 3개월 체계용) — 동시 요청 3건 이하로 순차 처리.

  CDSAPI_KEY=... python seas5_monthly/get_cds.py OUTDIR [--init 10,11] [--fc-year 2026] [--only era5|seas5] [--workers 3]

받는 것 (모두 월평균 NetCDF, 1° 격자):
  ERA5 (학습·검증 기준, 1991~현재)
    era5_sfc.nc      t2m·d2m·msl·tp·u10·v10·sst            영역 [60N,60E,10N,180E]
    era5_sst_nino34.nc  sst                                  Niño3.4 상자 [5N,170W,5S,120W]
    era5_pl.nc       z·t·q·u·v @500·850 hPa                 영역 [60N,60E,0N,180E]
  SEAS5 system 51 (초기화 달마다): 재예보 1993~2016 (25멤버) + 실황 예보(51멤버), 리드 1~4개월
    seas5_{hind|fc<year>}_{mm}_sfc.nc / _sst_nino34.nc / _pl.nc   (변수·영역은 ERA5와 동일)

인증키는 환경변수 CDSAPI_KEY 로만 읽는다. 이미 있는 파일은 건너뛴다.
"""
import argparse, os, sys, time, traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CDS_URL = "https://cds.climate.copernicus.eu/api"
AREA_SFC = [60, 60, 10, 180]
AREA_PL = [60, 60, 0, 180]
AREA_NINO34 = [5, -170, -5, -120]
SFC_VARS = ["2m_temperature", "2m_dewpoint_temperature", "mean_sea_level_pressure", "total_precipitation",
            "10m_u_component_of_wind", "10m_v_component_of_wind", "sea_surface_temperature"]
PL_VARS = ["geopotential", "temperature", "specific_humidity", "u_component_of_wind", "v_component_of_wind"]
LEVELS = ["500", "850"]
LEADS = ["1", "2", "3", "4"]
HIND_YEARS = [str(y) for y in range(1993, 2017)]
MONTHS = [f"{m:02d}" for m in range(1, 13)]


def era5_requests(y0=1991, y1=None):
    y1 = y1 or time.gmtime().tm_year
    years = [str(y) for y in range(y0, y1 + 1)]
    base = dict(product_type=["monthly_averaged_reanalysis"], year=years, month=MONTHS, time=["00:00"],
                data_format="netcdf", download_format="unarchived", grid=[1.0, 1.0])
    return [
        ("era5_sfc.nc", "reanalysis-era5-single-levels-monthly-means", dict(base, variable=SFC_VARS, area=AREA_SFC)),
        ("era5_sst_nino34.nc", "reanalysis-era5-single-levels-monthly-means",
         dict(base, variable=["sea_surface_temperature"], area=AREA_NINO34)),
        ("era5_pl.nc", "reanalysis-era5-pressure-levels-monthly-means",
         dict(base, variable=PL_VARS, pressure_level=LEVELS, area=AREA_PL)),
    ]


def seas5_requests(init_months, fc_year):
    reqs = []
    for im in init_months:
        mm = f"{im:02d}"
        for tag, years in (("hind", HIND_YEARS), (f"fc{fc_year}", [str(fc_year)])):
            base = dict(originating_centre="ecmwf", system="51", product_type=["monthly_mean"], year=years,
                        month=[mm], leadtime_month=LEADS, data_format="netcdf")
            reqs += [
                (f"seas5_{tag}_{mm}_sfc.nc", "seasonal-monthly-single-levels", dict(base, variable=SFC_VARS, area=AREA_SFC)),
                (f"seas5_{tag}_{mm}_sst_nino34.nc", "seasonal-monthly-single-levels",
                 dict(base, variable=["sea_surface_temperature"], area=AREA_NINO34)),
                (f"seas5_{tag}_{mm}_pl.nc", "seasonal-monthly-pressure-levels",
                 dict(base, variable=PL_VARS, pressure_level=LEVELS, area=AREA_PL)),
            ]
    return reqs


def fetch(out, name, dataset, req):
    import cdsapi
    f = out / name
    if f.exists() and f.stat().st_size > 0:
        return f"skip {name}"
    t0 = time.time()
    log = out / f"log_{name}.txt"
    try:
        c = cdsapi.Client(url=CDS_URL, key=os.environ["CDSAPI_KEY"], quiet=True, progress=False)
        tmp = f.with_suffix(".part")
        c.retrieve(dataset, req, str(tmp))
        tmp.rename(f)
        msg = f"ok {name} {f.stat().st_size // 1024} KB {time.time() - t0:.0f}s"
    except Exception as e:  # noqa: BLE001
        msg = f"FAIL {name} {time.time() - t0:.0f}s {e}"
        log.write_text(traceback.format_exc(), encoding="utf-8")
    print(msg, flush=True)
    return msg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--init", default="10,11", help="SEAS5 초기화 달(쉼표)")
    ap.add_argument("--fc-year", type=int, default=time.gmtime().tm_year)
    ap.add_argument("--only", choices=["era5", "seas5"])
    ap.add_argument("--no-fc", action="store_true", help="실황 예보는 받지 않음(재예보만)")
    ap.add_argument("--workers", type=int, default=3, help="동시 요청 수 (CDS 예절상 2~3)")
    a = ap.parse_args()
    if "CDSAPI_KEY" not in os.environ:
        sys.exit("CDSAPI_KEY 환경변수가 없습니다")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    reqs = []
    if a.only != "seas5":
        reqs += era5_requests()
    if a.only != "era5":
        r = seas5_requests([int(m) for m in a.init.split(",")], a.fc_year)
        reqs += [x for x in r if not (a.no_fc and "_fc" in x[0])]
    print(f"{len(reqs)} requests, {a.workers} at a time", flush=True)
    with ThreadPoolExecutor(max_workers=min(a.workers, 3)) as ex:
        list(ex.map(lambda r: fetch(out, *r), reqs))


if __name__ == "__main__":
    main()
