"""SEAS5 월별 체계의 예측인자 — PyESD 원본 구성에 맞춤.

PyESD                      여기
Predictor + RegionalAverage  대상점 반경 250 km 가우스 가중평균 (지상 7 + 상층 6 변수, 1° 격자)
teleconnection indices      장에서 직접 계산: NINO34, WNPSH(z500), SH(msl 시베리아고기압), EAJ(u500 동아시아 제트), KSST(한반도 주변 SST)
eof_analysis                z500·msl·t2m 영역 EOF 상위 3 PC (ERA5로 적합, SEAS5는 투영)
MonthlyStandardizer         ERA5: 1991~2020 달별 기후값. SEAS5: 자기 재예보(1993~2016, 초기화달·리드별) 기후값 + ERA5 기후 차이 보정

자료: get_cds.py 산출 (era5_sfc.nc, era5_pl.nc, era5_sst_nino34.nc, seas5_{hind|fcYYYY}_{mm}_{sfc|pl|sst_nino34}.nc)
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pyesd"))
import esd  # noqa: E402

CLIM = (1991, 2020)
HIND = (1993, 2016)
SFC = ["t2m", "d2m", "msl", "tp", "u10", "v10", "sst"]
PL = ["z500", "z850", "t850", "q850", "u850", "v850"]
LOCAL = SFC[:6] + PL                       # 지역평균 인자 12개 (sst는 지수로만)
INDICES = ["NINO34", "WNPSH", "SH", "EAJ", "KSST"]
EOF_VARS = {"z500": 3, "msl": 3, "t2m": 3}
BOX = {"WNPSH": ("z500", 15, 30, 115, 150), "SH": ("msl", 40, 60, 80, 120),
       "EAJ": ("u500", 30, 40, 120, 160), "KSST": ("sst", 30, 40, 120, 135)}
PRED_FEATURES = LOCAL + INDICES + [f"{v}_pc{i+1}" for v, n in EOF_VARS.items() for i in range(n)]
NAME = {"2t": "t2m", "2d": "d2m", "10u": "u10", "10v": "v10", "tprate": "tp"}


# ---------------------------------------------------------------- 파일 읽기
def _std(ds):
    ren = {}
    for a, b in (("valid_time", "time"), ("forecast_reference_time", "init"), ("forecastMonth", "lead"),
                 ("latitude", "lat"), ("longitude", "lon"), ("pressure_level", "level"), ("isobaricInhPa", "level"),
                 ("number", "member")):
        if a in ds.dims or a in ds.coords:
            ren[a] = b
    ds = ds.rename(ren)
    ds = ds.rename({k: NAME[k] for k in ds.data_vars if k in NAME})
    if "lat" in ds.dims and ds.lat.values[0] > ds.lat.values[-1]:
        ds = ds.isel(lat=slice(None, None, -1))
    return ds.drop_vars([c for c in ("expver", "surface", "step", "valid_time") if c in ds.coords and c not in ds.dims])


def _units(ds, kind):
    """물리단위 통일: t2m·d2m·sst °C, msl hPa, tp mm/일(sqrt는 나중), z m, t850 °C, q g/kg"""
    out = {}
    for v in ds.data_vars:
        x = ds[v]
        if v in ("t2m", "d2m", "sst"):
            x = x - 273.15
        elif v == "msl":
            x = x / 100.0
        elif v == "tp":
            x = x * 86400 * 1000 if kind == "seas5" else x * 1000      # SEAS5 tprate m/s, ERA5 월평균 m/일
        out[v] = x
    return out


def _split_levels(fields):
    out = {}
    for v, x in fields.items():
        if "level" in x.dims:
            for lv in x.level.values:
                y = x.sel(level=lv).drop_vars("level")
                nm = {"z": "z", "t": "t", "q": "q", "u": "u", "v": "v"}[v] + f"{int(lv)}"
                if nm.startswith("z"):
                    y = y / 9.80665
                elif nm.startswith("t"):
                    y = y - 273.15
                elif nm.startswith("q"):
                    y = y * 1000
                out[nm] = y
        else:
            out[v] = x
    return out


def load_era5(d):
    d = Path(d)
    sfc = _units(_std(xr.open_dataset(d / "era5_sfc.nc")), "era5")
    pl = _units(_std(xr.open_dataset(d / "era5_pl.nc")), "era5")
    nino = _units(_std(xr.open_dataset(d / "era5_sst_nino34.nc")), "era5")["sst"]
    F = {**sfc, **_split_levels(pl)}
    F = {k: v.transpose("time", "lat", "lon") for k, v in F.items()}
    F["nino_sst"] = nino.transpose("time", "lat", "lon")
    t = pd.DatetimeIndex(F["t2m"].time.values).to_period("M").to_timestamp()
    for k in F:
        F[k] = F[k].assign_coords(time=t)
    return F


def load_seas5(d, tag, mm):
    """tag = 'hind' | 'fc2026'. 반환 dict var -> DataArray(member, init, lead, lat, lon)"""
    d = Path(d); mm = f"{int(mm):02d}"
    sfc = _units(_std(xr.open_dataset(d / f"seas5_{tag}_{mm}_sfc.nc")), "seas5")
    pl = _units(_std(xr.open_dataset(d / f"seas5_{tag}_{mm}_pl.nc")), "seas5")
    nino = _units(_std(xr.open_dataset(d / f"seas5_{tag}_{mm}_sst_nino34.nc")), "seas5")["sst"]
    F = {**sfc, **_split_levels(pl)}
    F = {k: v.transpose("member", "init", "lead", "lat", "lon") for k, v in F.items()}
    F["nino_sst"] = nino.transpose("member", "init", "lead", "lat", "lon")
    return F


# ---------------------------------------------------------------- 편차
def era5_clim(F, years=CLIM):
    c = {}
    for k, x in F.items():
        s = x.sel(time=slice(f"{years[0]}-01-01", f"{years[1]}-12-31"))
        c[k] = s.groupby("time.month").mean("time")
    return c


def era5_anoms(F, clim):
    """ERA5 편차 (강수는 sqrt 공간에서 편차). 반환 dict var -> (time, lat, lon)"""
    A = {}
    for k, x in F.items():
        if k == "tp":
            x = np.sqrt(x.clip(min=0)); c = np.sqrt(clim[k].clip(min=0))
        else:
            c = clim[k]
        A[k] = (x.groupby("time.month") - c).drop_vars("month", errors="ignore")
    return A


def seas5_anoms(Fh, Ff, clim9120, clim9316):
    """SEAS5 편차: 자기 재예보 기후(초기화달·리드별) 기준 + (ERA5 93~16 기후 − 91~20 기후) 보정.
    Fh 재예보, Ff 적용 대상(재예보 자신 또는 실황). 반환 dict var -> (member, init, lead, lat, lon)와 검증월 목록"""
    A = {}
    for k in Ff:
        xh, xf = Fh[k], Ff[k]
        if k == "tp":
            xh, xf = np.sqrt(xh.clip(min=0)), np.sqrt(xf.clip(min=0))
        mc = xh.mean(("member", "init"))                                  # (lead, lat, lon)
        a = xf - mc
        # 리드별 검증월에 해당하는 ERA5 기후 차이 (같은 격자로 보간)
        init0 = pd.Timestamp(xf.init.values[0])
        adj = []
        for ld in xf.lead.values:
            vm = (init0 + pd.DateOffset(months=int(ld) - 1)).month
            c1 = clim9316[k].sel(month=vm); c2 = clim9120[k].sel(month=vm)
            if k == "tp":
                c1, c2 = np.sqrt(c1.clip(min=0)), np.sqrt(c2.clip(min=0))
            adj.append((c1 - c2).interp(lat=xf.lat, lon=xf.lon, method="nearest"))
        adj = xr.concat(adj, dim="lead").assign_coords(lead=xf.lead.values)
        A[k] = a + adj
    return A


# ---------------------------------------------------------------- 인자 계산
def box_mean(x, lat0, lat1, lon0, lon1):
    s = x.sel(lat=slice(lat0, lat1), lon=slice(lon0, lon1))
    w = np.cos(np.deg2rad(s.lat))
    return s.weighted(w).mean(("lat", "lon"))


def indices(A):
    """A: 편차 dict (ERA5: (time,…) / SEAS5: (member,init,lead,…)). 반환 dict name -> DataArray(샘플 차원…)"""
    I = {"NINO34": A["nino_sst"].mean(("lat", "lon"))}
    for nm, (v, la0, la1, lo0, lo1) in BOX.items():
        I[nm] = box_mean(A[v], la0, la1, lo0, lo1)
    return I


class Weights:
    """대상점 × 1° 격자점 반경 가중 (PyESD RegionalAverage)."""

    def __init__(self, lat2d, lon2d, tlat, tlon, radius=250.0):
        self.plat, self.plon = lat2d.ravel(), lon2d.ravel()
        self.W = esd.radius_weights(self.plat, self.plon, np.asarray(tlat), np.asarray(tlon), radius)  # (npt, ngrid)

    def apply(self, x):
        """x: (..., lat, lon) -> (..., npt)"""
        v = np.nan_to_num(np.asarray(x).reshape(x.shape[:-2] + (-1,)))
        return v @ self.W.T


class EOFs:
    """변수별 표준화 편차장 EOF (ERA5로 적합). transform 은 ERA5·SEAS5 공통."""

    def __init__(self, nvar=EOF_VARS):
        self.nvar = nvar

    def fit(self, A2d):          # A2d: {v: (n, ngrid)}
        self.mu, self.sd, self.eof, self.pcsd, self.evr = {}, {}, {}, {}, {}
        for v, n in self.nvar.items():
            X = np.nan_to_num(A2d[v]); self.mu[v] = X.mean(0); self.sd[v] = X.std(0) + 1e-6
            Z = (X - self.mu[v]) / self.sd[v]
            U, S, Vt = np.linalg.svd(Z, full_matrices=False)
            self.eof[v] = Vt[:n]; self.pcsd[v] = (Z @ Vt[:n].T).std(0) + 1e-9
            self.evr[v] = S[:n] ** 2 / (S ** 2).sum()
        return self

    def transform(self, A2d):
        return np.concatenate([(np.nan_to_num(A2d[v]) - self.mu[v]) / self.sd[v] @ self.eof[v].T / self.pcsd[v]
                               for v in self.nvar], -1)


def feature_table(A, Wt, eofs, sample_dims):
    """편차 dict -> (n, p, npt) 설계 입력. p = LOCAL(점별) + INDICES(공통) + PC(공통)
    sample_dims: ERA5 ('time',) / SEAS5 ('member','init','lead')"""
    def flat(x):
        return np.asarray(x.transpose(*sample_dims, "lat", "lon")).reshape(-1, x.sizes["lat"] * x.sizes["lon"])
    loc = np.stack([Wt.apply(flat(A[v])) for v in LOCAL], 1)                      # (n, 12, npt)
    I = indices(A)
    idx = np.stack([np.asarray(I[k].transpose(*[d for d in sample_dims if d in I[k].dims])).ravel() for k in INDICES], 1)  # (n, 5)
    pcs = eofs.transform({v: flat(A[v]) for v in EOF_VARS})                        # (n, 9)
    common = np.concatenate([idx, pcs], 1)
    n, npt = loc.shape[0], loc.shape[2]
    X = np.concatenate([loc, np.broadcast_to(common[:, :, None], common.shape + (npt,))], 1)
    return X                                                                       # (n, p, npt)
