"""C3S CDS의 SEAS5(system 51) 월평균: 재예보(1993~2016) 모형기후와 2026 예보 멤버.
Open-Meteo SEAS5는 편향보정이 없으므로, 모형 기후값(같은 시작월·리드)을 빼서 편차를 만든다.
  CDSAPI_KEY=... python get_seas5_cds.py OUTDIR hind|fc INIT_MONTH [YEAR]
"""
import os, sys, cdsapi
out, what, im = sys.argv[1], sys.argv[2], int(sys.argv[3])
c = cdsapi.Client(url="https://cds.climate.copernicus.eu/api", key=os.environ["CDSAPI_KEY"])
years = [str(y) for y in range(1993, 2017)] if what == "hind" else [sys.argv[4]]
req = {"originating_centre": "ecmwf", "system": "51",
       "variable": ["2m_temperature", "2m_dewpoint_temperature", "mean_sea_level_pressure", "total_precipitation"],
       "product_type": ["monthly_mean"], "year": years, "month": [f"{im:02d}"],
       "leadtime_month": ["1", "2", "3", "4", "5", "6"], "data_format": "netcdf",
       "area": [46.5, 113.5, 27.5, 141.5]}
f = f"{out}/seas5_{what}_{years[0] if what == 'fc' else 'hind'}{im:02d}.nc"
c.retrieve("seasonal-monthly-single-levels", req, f)
print(f)
