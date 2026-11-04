"""SEAS5 월별 체계의 예측인자 — PyESD 원본 구성에 맞춤.

PyESD                      여기
Predictor + RegionalAverage  대상점 반경 250 km 가우스 가중평균 (지상 7 + 상층 6 변수, 1° 격자)
teleconnection indices      장에서 직접 계산 (INDEX_DEFS·PATTERN_DEFS; 2026 여름 기후인자 분석과 같은 정의, 월평균 기준):
                            상자평균 NINO34·WNPSH(z850)·SH·KH·OKH·KSST·WP·EAJ(u200; 없으면 u500)·CGT·SAH·DMI·EU,
                            EOF 패턴 PJ(850 와도)·NAO·AO·SRP(v200). 자료가 없는 지수는 자동 제외
eof_analysis                z500·msl·t2m 영역 EOF 상위 3 PC (ERA5로 적합, SEAS5는 투영)
MonthlyStandardizer         ERA5: 1991~2020 달별 기후값. SEAS5: 자기 재예보(1993~2016, 초기화달·리드별) 기후값 + ERA5 기후 차이 보정

자료: get_cds.py 산출 (era5_sfc.nc, era5_pl.nc, era5_sst_nino34.nc, seas5_{hind|fcYYYY}_{mm}_{sfc|pl|sst_nino34}.nc)
      확장(있으면 사용): *_pl200.nc (z·u·v 200 hPa), *_sst_io.nc (인도양 SST → DMI), *_z500_nh.nc (북반구 z500 → NAO·AO·EU)
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
EOF_VARS = {"z500": 3, "msl": 3, "t2m": 3}
EXT_FILES = {"pl200": None, "sst_io": "io_sst", "z500_nh": "z500_nh"}   # 접미사 -> 변수명(None: 층별 분리)
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


def _common_grid(F, ref):
    """상층(0~60N)과 지상(10~60N) 영역이 달라 지상 격자 범위로 맞춤."""
    la0, la1 = float(ref.lat.min()), float(ref.lat.max())
    return {k: v.sel(lat=slice(la0, la1)) for k, v in F.items()}


def load_era5(d):
    d = Path(d)
    sfc = _units(_std(xr.open_dataset(d / "era5_sfc.nc")), "era5")
    pl = _units(_std(xr.open_dataset(d / "era5_pl.nc")), "era5")
    nino = _units(_std(xr.open_dataset(d / "era5_sst_nino34.nc")), "era5")["sst"]
    F = _common_grid({**sfc, **_split_levels(pl)}, sfc["t2m"])
    F["nino_sst"] = nino
    F.update(_ext(d, "era5_{suf}.nc", "era5"))
    F = {k: v.transpose("time", "lat", "lon") for k, v in F.items()}
    t = pd.DatetimeIndex(F["t2m"].time.values).to_period("M").to_timestamp()
    for k in F:
        F[k] = F[k].assign_coords(time=t)
    return F


def _ext(d, pattern, kind):
    """확장 파일(200 hPa, 인도양 SST, 북반구 z500)이 있으면 읽음."""
    out = {}
    for suf, nm in EXT_FILES.items():
        f = d / pattern.format(suf=suf)
        if not f.exists():
            continue
        ds = _units(_std(xr.open_dataset(f)), kind)
        if nm is None:
            out.update({k: v for k, v in _split_levels(ds).items() if k.endswith("200")})
        else:
            v = list(ds.values())[0]
            if nm == "z500_nh" and "level" in v.dims:
                v = v.sel(level=500).drop_vars("level") / 9.80665
            if nm == "z500_nh" and float(v.lon.max()) > 180:          # 경도 -180~180 으로 정렬 (NAO 상자)
                v = v.assign_coords(lon=((v.lon + 180) % 360) - 180).sortby("lon")
            out[nm] = v
    return out


def load_seas5(d, tag, mm):
    """tag = 'hind' | 'fc2026'. 반환 dict var -> DataArray(member, init, lead, lat, lon)"""
    d = Path(d); mm = f"{int(mm):02d}"
    sfc = _units(_std(xr.open_dataset(d / f"seas5_{tag}_{mm}_sfc.nc")), "seas5")
    pl = _units(_std(xr.open_dataset(d / f"seas5_{tag}_{mm}_pl.nc")), "seas5")
    nino = _units(_std(xr.open_dataset(d / f"seas5_{tag}_{mm}_sst_nino34.nc")), "seas5")["sst"]
    F = _common_grid({**sfc, **_split_levels(pl)}, sfc["t2m"])
    F["nino_sst"] = nino
    F.update(_ext(d, f"seas5_{tag}_{mm}_{{suf}}.nc", "seas5"))
    return {k: v.transpose("member", "init", "lead", "lat", "lon") for k, v in F.items()}


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


def hind_clim(Fh):
    """재예보 기후값 (초기화달 고정, 리드별): var -> (lead, lat, lon). 강수는 sqrt 공간."""
    c = {}
    for k, x in Fh.items():
        if k == "tp":
            x = np.sqrt(x.clip(min=0))
        c[k] = x.mean(("member", "init"))
    return c


def seas5_anoms(hc, Ff, clim9120, clim9316):
    """SEAS5 편차: 자기 재예보 기후(초기화달·리드별, hind_clim) 기준 + (ERA5 93~16 기후 − 91~20 기후) 보정.
    Ff 적용 대상(재예보 자신 또는 실황). 반환 dict var -> (member, init, lead, lat, lon)"""
    A = {}
    for k in Ff:
        xf = Ff[k]
        if k == "tp":
            xf = np.sqrt(xf.clip(min=0))
        a = xf - hc[k]
        init0 = pd.Timestamp(xf.init.values[0])
        adj = []
        for ld in xf.lead.values:
            vm = (init0 + pd.DateOffset(months=int(ld) - 1)).month
            c1 = clim9316[k].sel(month=vm); c2 = clim9120[k].sel(month=vm)
            if k == "tp":
                c1, c2 = np.sqrt(c1.clip(min=0)), np.sqrt(c2.clip(min=0))
            adj.append((c1 - c2).interp(lat=xf.lat, lon=xf.lon, method="nearest", kwargs={"fill_value": "extrapolate"}))
        adj = xr.concat(adj, dim="lead").assign_coords(lead=xf.lead.values)
        A[k] = a + adj
    return A


def save_clims(path, c9120, c9316, hc=None):
    """ERA5 기후값(1991~2020, 1993~2016)과 재예보 기후값을 변수별 그룹으로 저장 (격자가 변수마다 달라 그룹 분리).
    운영 전망은 이 파일 + 실황 SEAS5 파일만 있으면 된다."""
    path = Path(path)
    if path.exists():
        path.unlink()
    for i, k in enumerate(c9120):
        xr.Dataset({"c9120": c9120[k], "c9316": c9316[k]}).to_netcdf(path, group=k, mode="w" if i == 0 else "a")
        if hc and k in hc:                                   # 재예보 기후는 SEAS5 격자(x.5°)라 별도 하위 그룹
            xr.Dataset({"hc": hc[k]}).to_netcdf(path, group=f"{k}/hc", mode="a")


def load_clims(path):
    import netCDF4
    with netCDF4.Dataset(path) as nc:
        names = list(nc.groups)
    c9120, c9316, hc = {}, {}, {}
    for k in names:
        ds = xr.open_dataset(path, group=k).load()
        c9120[k], c9316[k] = ds["c9120"], ds["c9316"]
        with netCDF4.Dataset(path) as nc:
            has = "hc" in nc.groups[k].groups
        if has:
            hc[k] = xr.open_dataset(path, group=f"{k}/hc").load()["hc"]
    return c9120, c9316, hc


# ---------------------------------------------------------------- 인자 계산
def box_mean(x, lat0, lat1, lon0, lon1):
    s = x.sel(lat=slice(lat0, lat1), lon=slice(lon0, lon1))
    w = np.cos(np.deg2rad(s.lat))
    return s.weighted(w).mean(("lat", "lon"))


def _pt(x, la, lo):
    return x.sel(lat=la, lon=lo, method="nearest").drop_vars(("lat", "lon"))


def _bx(A, v, la0, la1, lo0, lo1):
    return box_mean(A[v], la0, la1, lo0, lo1)


# 상자·점 지수: 이름 -> (필요 변수, 함수).  z 단위 m, u m/s, sst °C, msl hPa (편차장 입력)
INDEX_DEFS = {
    "NINO34": (["nino_sst"], lambda A: A["nino_sst"].mean(("lat", "lon"))),                 # Niño3.4 SST
    "WNPSH": (["z850"], lambda A: _bx(A, "z850", 15, 25, 115, 150)),                        # 서태평양고기압 (Wang 2013)
    "SH": (["msl"], lambda A: _bx(A, "msl", 40, 60, 80, 120)),                              # 시베리아고기압
    "KH": (["z500"], lambda A: _bx(A, "z500", 30, 45, 120, 135)),                           # 한반도 상공 고기압
    "OKH": (["z500"], lambda A: _bx(A, "z500", 50, 60, 130, 150)),                          # 오호츠크해 고기압 (영역 상한 60N)
    "KSST": (["sst"], lambda A: _bx(A, "sst", 25, 40, 120, 135)),                           # 한반도 주변 SST
    "WP": (["z500"], lambda A: 0.5 * (_pt(A["z500"], 60, 155) - _pt(A["z500"], 30, 155))),  # 서태평양 패턴 (WG81)
    "EAJ": (["u200"], lambda A: _bx(A, "u200", 40, 50, 100, 140) - _bx(A, "u200", 30, 37.5, 100, 140)),  # 동아시아 제트 위치
    "EAJ500": (["u500"], lambda A: _bx(A, "u500", 40, 50, 100, 140) - _bx(A, "u500", 30, 37.5, 100, 140)),  # (u200 없을 때 대용)
    "CGT": (["z200"], lambda A: _bx(A, "z200", 35, 40, 60, 70)),                            # 순환 전구 원격상관 (Ding & Wang 2005)
    "SAH": (["z200"], lambda A: _bx(A, "z200", 25, 35, 100, 125)),                          # 티베트고기압 동쪽 확장
    "DMI": (["io_sst"], lambda A: _bx(A, "io_sst", -10, 10, 50, 70) - _bx(A, "io_sst", -10, 0, 90, 110)),  # IOD
    "EU": (["z500_nh"], lambda A: -0.25 * _pt(A["z500_nh"], 55, 20) + 0.5 * _pt(A["z500_nh"], 55, 75)
           - 0.25 * _pt(A["z500_nh"], 40, 145)),                                            # 유라시아 패턴 (WG81)
}
# EOF 패턴 지수: 이름 -> (필요 변수, 장 함수, 영역, 부호 규칙 함수(패턴 DataArray -> 양이어야 함))
def _vort850(A):
    u, v = A["u850"], A["v850"]
    R = 6.371e6; cos = np.cos(np.deg2rad(u.lat))
    return (v.differentiate("lon") / (R * cos) - (u * cos).differentiate("lat") / (R * cos)) * 180 / np.pi * 1e6


PATTERN_DEFS = {
    "PJ": (["u850", "v850"], _vort850, (10, 60, 100, 160),
           lambda e: float(e.sel(lat=slice(10, 22.5), lon=slice(115, 145)).mean())),      # 태평양-일본 패턴 (Kosaka & Nakamura)
    "NAO": (["z500_nh"], lambda A: A["z500_nh"], (20, 80, -90, 40),
            lambda e: float(e.sel(lat=slice(35, 47.5), lon=slice(-40, 0)).mean() - e.sel(lat=slice(57.5, 72.5), lon=slice(-40, 0)).mean())),
    "AO": (["z500_nh"], lambda A: A["z500_nh"], (20, 90, -180, 180),
           lambda e: -float(e.sel(lat=slice(70, 90)).mean())),
    "SRP": (["v200"], lambda A: A["v200"], (20, 60, 30, 130),
            lambda e: float(e.sel(lat=slice(35, 45), lon=slice(55, 75)).mean())),           # 실크로드 패턴 (부호: CGT 중심 양)
}


def available_indices(A):
    have = set(A)
    names = [k for k, (need, _) in INDEX_DEFS.items() if set(need) <= have]
    if "EAJ" in names and "EAJ500" in names:
        names.remove("EAJ500")
    return names


class Patterns:
    """EOF1 패턴 지수 (ERA5 편차장으로 적합, SEAS5는 투영). sqrt(cos) 가중, PC는 적합기간 표준편차로 정규화."""

    def __init__(self):
        self.pat = {}            # name -> dict(e (ngrid,), mu (ngrid,), sd float, lat, lon)

    @staticmethod
    def _field(A, name, sample_dims):
        need, fn, (la0, la1, lo0, lo1), _ = PATTERN_DEFS[name]
        x = fn(A).sel(lat=slice(la0, la1), lon=slice(lo0, lo1))
        w = np.sqrt(np.cos(np.deg2rad(x.lat))).clip(min=1e-3)
        X = np.asarray((x * w).transpose(*sample_dims, "lat", "lon")).reshape(-1, x.sizes["lat"] * x.sizes["lon"])
        return np.nan_to_num(X), x.lat.values, x.lon.values, np.asarray(w)

    def fit(self, A, sample_dims=("time",)):
        for name in PATTERN_DEFS:
            if not set(PATTERN_DEFS[name][0]) <= set(A):
                continue
            X, lat, lon, w = self._field(A, name, sample_dims)
            mu = X.mean(0)
            _, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
            e = Vt[0]
            pat = xr.DataArray((e.reshape(len(lat), len(lon)) / w[:, None]), coords={"lat": lat, "lon": lon})
            if PATTERN_DEFS[name][3](pat) < 0:
                e = -e
            pc = (X - mu) @ e
            self.pat[name] = dict(e=e, mu=mu, sd=pc.std() + 1e-9, lat=lat, lon=lon, evr=S[0] ** 2 / (S ** 2).sum())
        return self

    def transform(self, A, sample_dims):
        out = {}
        for name, p in self.pat.items():
            X, _, _, _ = self._field(A, name, sample_dims)
            out[name] = (X - p["mu"]) @ p["e"] / p["sd"]
        return out

    def to_npz(self, d):
        for k, p in self.pat.items():
            d[f"pat_{k}_e"], d[f"pat_{k}_mu"], d[f"pat_{k}_sd"], d[f"pat_{k}_lat"], d[f"pat_{k}_lon"] = p["e"], p["mu"], p["sd"], p["lat"], p["lon"]
        return d

    @classmethod
    def from_npz(cls, M):
        self = cls()
        for k in PATTERN_DEFS:
            if f"pat_{k}_e" in M:
                self.pat[k] = dict(e=M[f"pat_{k}_e"], mu=M[f"pat_{k}_mu"], sd=float(M[f"pat_{k}_sd"]), lat=M[f"pat_{k}_lat"], lon=M[f"pat_{k}_lon"])
        return self


def indices(A, names=None, patterns=None, sample_dims=("time",)):
    """A: 편차 dict. 반환 dict name -> 1D ndarray (표본 순서 = sample_dims 평탄화 순서)"""
    names = names or available_indices(A)
    I = {}
    for k in names:
        x = INDEX_DEFS[k][1](A)
        I[k] = np.asarray(x.transpose(*[d for d in sample_dims if d in x.dims])).ravel()
    if patterns is not None:
        I.update(patterns.transform(A, sample_dims))
    return I


class Weights:
    """대상점 × 1° 격자점 반경 가중 (PyESD RegionalAverage)."""

    def __init__(self, lat2d, lon2d, tlat, tlon, radius=250.0):
        self.plat, self.plon = lat2d.ravel(), lon2d.ravel()
        self.W = esd.radius_weights(self.plat, self.plon, np.asarray(tlat), np.asarray(tlon), radius)  # (npt, ngrid)

    def apply(self, x):
        """x: (..., ngrid) 평탄화된 장 -> (..., npt)"""
        return np.nan_to_num(np.asarray(x)) @ self.W.T


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


def feature_names(idx_names, patterns):
    return LOCAL + list(idx_names) + list(patterns.pat if patterns else []) + [f"{v}_pc{i+1}" for v, n in EOF_VARS.items() for i in range(n)]


def feature_table(A, Wt, eofs, sample_dims, idx_names, patterns):
    """편차 dict -> (n, p, npt) 설계 입력. p = LOCAL(점별) + 지수(공통) + 패턴지수(공통) + PC(공통)
    sample_dims: ERA5 ('time',) / SEAS5 ('member','init','lead')"""
    def flat(x):
        return np.asarray(x.transpose(*sample_dims, "lat", "lon")).reshape(-1, x.sizes["lat"] * x.sizes["lon"])
    loc = np.stack([Wt.apply(flat(A[v])) for v in LOCAL], 1)                      # (n, 12, npt)
    I = indices(A, idx_names, patterns, sample_dims)
    idx = np.stack([I[k] for k in list(idx_names) + list(patterns.pat if patterns else [])], 1) if I else np.zeros((loc.shape[0], 0))
    pcs = eofs.transform({v: flat(A[v]) for v in EOF_VARS})                        # (n, 9)
    common = np.nan_to_num(np.concatenate([idx, pcs], 1))
    npt = loc.shape[2]
    return np.concatenate([loc, np.broadcast_to(common[:, :, None], common.shape + (npt,))], 1)   # (n, p, npt)


def regrid(A, ref):
    """SEAS5(격자 중심 x.5°) 편차장을 변수별 ERA5 격자(ref[k]의 lat/lon)로 선형보간. 경계 밖은 최근접으로 채움."""
    out = {}
    for k, x in A.items():
        lat, lon = ref[k].lat, ref[k].lon
        y = x.interp(lat=lat, lon=lon, method="linear", kwargs={"fill_value": None})
        if bool(y.isnull().any()):
            y = y.fillna(x.interp(lat=lat, lon=lon, method="nearest", kwargs={"fill_value": "extrapolate"}))
        out[k] = y
    return out


def valid_months(init, leads):
    """초기화 시각 + 리드(1=초기화 달) -> 검증 달 Timestamp 목록"""
    init = pd.Timestamp(init)
    return [init + pd.DateOffset(months=int(ld) - 1) for ld in leads]
