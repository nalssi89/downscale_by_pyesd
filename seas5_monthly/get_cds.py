"""C3S CDS 자료 입수 (SEAS5 월별 3개월 체계용) — 동시 요청 3건 이하로 순차 처리.

  CDSAPI_KEY=... python seas5_monthly/get_cds.py OUTDIR [--init 10,11] [--fc-year 2026] [--only era5|seas5] [--workers 3]

받는 것 (모두 월평균 NetCDF, 1° 격자):
  ERA5 (학습·검증 기준, 1991~현재; CDS 비용 한도 때문에 연도 구간별 parts/ 로 받아 합침)
    era5_sfc.nc      t2m·d2m·msl·tp·u10·v10·sst            영역 [60N,60E,10N,180E]
    era5_sst_nino34.nc  sst                                  Niño3.4 상자 [5N,170W,5S,120W]
    era5_pl.nc       z·t·q·u·v @500·850 hPa                 영역 [60N,60E,0N,180E]
    확장(기후지수용, --no-ext 로 생략): era5_pl200.nc z·u·v @200 [60N,20E,10N,180E], era5_sst_io.nc 인도양 SST [10N,50E,10S,110E],
                                        era5_z500_nh.nc z500 북반구 [90N,180W,20N,180E]
  SEAS5 system 51 (초기화 달마다): 재예보 1993~2016 (25멤버) + 실황 예보(51멤버), 리드 1~4개월
    seas5_{hind|fc<year>}_{mm}_sfc.nc / _sst_nino34.nc / _pl.nc / _pl200.nc / _sst_io.nc / _z500_nh.nc  (변수·영역은 ERA5와 동일)

인증키는 환경변수 CDSAPI_KEY 로만 읽는다. 이미 있는 파일은 건너뛴다.
"""
import argparse, os, sys, time, traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CDS_URL = "https://cds.climate.copernicus.eu/api"
AREA_SFC = [60, 60, 10, 180]
AREA_PL = [60, 60, 0, 180]
AREA_NINO34 = [5, -170, -5, -120]
AREA_IO = [10, 50, -10, 110]            # 인도양 (DMI)
AREA_NH = [90, -180, 20, 180]           # 북반구 z500 (NAO·AO·EU)
AREA_PL200 = [60, 20, 10, 180]          # 200 hPa z·u·v (CGT·SAH·EAJ·SRP; SRP 영역 30~130E)
SFC_VARS = ["2m_temperature", "2m_dewpoint_temperature", "mean_sea_level_pressure", "total_precipitation",
            "10m_u_component_of_wind", "10m_v_component_of_wind", "sea_surface_temperature"]
PL_VARS = ["geopotential", "temperature", "specific_humidity", "u_component_of_wind", "v_component_of_wind"]
LEVELS = ["500", "850"]
LEADS = ["1", "2", "3", "4"]
HIND_YEARS = [str(y) for y in range(1993, 2017)]
MONTHS = [f"{m:02d}" for m in range(1, 13)]


ERA5_PARTS = {"era5_sfc.nc": [(1991, 2008), (2009, 2025)], "era5_pl.nc": [(1991, 2025)], "era5_sst_nino34.nc": [(1991, 2025)],
              "era5_pl200.nc": [(1991, 2025)], "era5_sst_io.nc": [(1991, 2025)], "era5_z500_nh.nc": [(1991, 2008), (2009, 2025)]}
ERA5_SPEC = {"era5_sfc.nc": ("reanalysis-era5-single-levels-monthly-means", dict(variable=SFC_VARS, area=AREA_SFC)),
             "era5_sst_nino34.nc": ("reanalysis-era5-single-levels-monthly-means",
                                    dict(variable=["sea_surface_temperature"], area=AREA_NINO34)),
             "era5_pl.nc": ("reanalysis-era5-pressure-levels-monthly-means",
                            dict(variable=PL_VARS, pressure_level=LEVELS, area=AREA_PL)),
             # 확장(기후지수용): 2026 여름 기후인자 분석과 같은 정의의 지수를 계산하기 위한 추가 장
             "era5_pl200.nc": ("reanalysis-era5-pressure-levels-monthly-means",
                               dict(variable=["geopotential", "u_component_of_wind", "v_component_of_wind"], pressure_level=["200"], area=AREA_PL200)),
             "era5_sst_io.nc": ("reanalysis-era5-single-levels-monthly-means",
                                dict(variable=["sea_surface_temperature"], area=AREA_IO)),
             "era5_z500_nh.nc": ("reanalysis-era5-pressure-levels-monthly-means",
                                 dict(variable=["geopotential"], pressure_level=["500"], area=AREA_NH))}
SEAS5_EXT = {"pl200": ("seasonal-monthly-pressure-levels",
                       dict(variable=["geopotential", "u_component_of_wind", "v_component_of_wind"], pressure_level=["200"], area=AREA_PL200)),
             "sst_io": ("seasonal-monthly-single-levels", dict(variable=["sea_surface_temperature"], area=AREA_IO)),
             "z500_nh": ("seasonal-monthly-pressure-levels", dict(variable=["geopotential"], pressure_level=["500"], area=AREA_NH))}


def _era5_base(years, months=MONTHS):
    return dict(product_type=["monthly_averaged_reanalysis"], year=[str(y) for y in years], month=list(months),
                time=["00:00"], data_format="netcdf", download_format="unarchived", grid=[1.0, 1.0])


def era5_requests(y0=1991, ext=True):
    """CDS 비용 한도(단일면 20000)를 넘지 않도록 연도 구간별 부분 파일로 나눔. 올해 분은 1~(현재달-1) → 실패 시 한 달 더 줄여 재시도.
    반환 [(부분파일명, dataset, 요청, 합칠 최종파일명), ...]"""
    now = time.gmtime(); cur = now.tm_year
    reqs = []
    for final, (ds, spec) in ERA5_SPEC.items():
        if not ext and final in ("era5_pl200.nc", "era5_sst_io.nc", "era5_z500_nh.nc"):
            continue
        for a, b in ERA5_PARTS[final]:
            reqs.append((f"parts/{final[:-3]}_{a}_{b}.nc", ds, dict(_era5_base(range(a, b + 1)), **spec), final))
        nm = max(now.tm_mon - 1, 1)                      # 월평균은 보통 전달까지 공개
        reqs.append((f"parts/{final[:-3]}_{cur}.nc", ds, [dict(_era5_base([cur], MONTHS[:k]), **spec) for k in (nm, nm - 1) if k >= 1], final))
    return reqs


def merge_era5(out, final, parts):
    import xarray as xr
    f = out / final
    if f.exists():
        return f"skip merge {final}"
    fs = [out / p for p in parts]
    miss = [p for p in fs if not p.exists()]
    if miss:
        return f"merge skip {final}: missing {[m.name for m in miss]}"
    ds = xr.open_mfdataset(fs, combine="by_coords", compat="override", coords="minimal")
    ds.load().to_netcdf(f)
    return f"merged {final} {f.stat().st_size // 1024} KB"


def seas5_requests(init_months, fc_year, ext=True):
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
            if ext:
                reqs += [(f"seas5_{tag}_{mm}_{suf}.nc", ds, dict(base, **spec)) for suf, (ds, spec) in SEAS5_EXT.items()]
    return reqs


def fetch(out, name, dataset, req):
    import cdsapi
    f = out / name
    if f.exists() and f.stat().st_size > 0:
        return f"skip {name}"
    t0 = time.time()
    log = out / f"log_{name}.txt"
    f.parent.mkdir(parents=True, exist_ok=True)
    msg = ""
    for r in (req if isinstance(req, list) else [req]):          # 리스트면 앞에서부터 시도(올해 분 폴백)
        try:
            c = cdsapi.Client(url=CDS_URL, key=os.environ["CDSAPI_KEY"], quiet=True, progress=False)
            tmp = f.with_suffix(".part")
            c.retrieve(dataset, r, str(tmp))
            tmp.rename(f)
            msg = f"ok {name} {f.stat().st_size // 1024} KB {time.time() - t0:.0f}s"
            break
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
    ap.add_argument("--no-hind", action="store_true", help="재예보는 받지 않음(실황만; 운영 전망용)")
    ap.add_argument("--workers", type=int, default=3, help="동시 요청 수 (CDS 예절상 2~3)")
    ap.add_argument("--no-ext", action="store_true", help="확장 장(200 hPa·인도양 SST·북반구 z500)은 받지 않음")
    a = ap.parse_args()
    if "CDSAPI_KEY" not in os.environ:
        sys.exit("CDSAPI_KEY 환경변수가 없습니다")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    reqs, merges = [], {}
    if a.only != "seas5":
        for part, ds, req, final in era5_requests(ext=not a.no_ext):
            merges.setdefault(final, []).append(part)
            if not (out / final).exists():
                reqs.append((part, ds, req))
    if a.only != "era5":
        r = seas5_requests([int(m) for m in a.init.split(",")], a.fc_year, ext=not a.no_ext)
        reqs += [x for x in r if not (a.no_fc and "_fc" in x[0]) and not (a.no_hind and "_hind_" in x[0])]
    print(f"{len(reqs)} requests, {a.workers} at a time", flush=True)
    with ThreadPoolExecutor(max_workers=min(a.workers, 3)) as ex:
        list(ex.map(lambda r: fetch(out, *r), reqs))
    for final, parts in merges.items():
        print(merge_era5(out, final, parts), flush=True)


if __name__ == "__main__":
    main()
