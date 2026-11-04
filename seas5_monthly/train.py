"""SEAS5 월별 체계 전이함수 학습 — PyESD 원본형 (점별 ARD 회귀, 자동 인자선택).

  KDS_WORK=... CDS_DIR=... python seas5_monthly/train.py [grid|asos] [T|P]

입력: ERA5 1° 예측인자(CDS_DIR/era5_*.nc, predictors.py), 예측대상 월평균(KDS_WORK/target_grid.nc 412격자, target_asos.nc 66지점)
방법: 달별(±1개월 창) 표본에 대해 EOF 적합 → 설계행렬 (지역평균 12 + 기후지수(자료에 따라 7~16) + EOF PC 9) 열 표준화 → 점별 ARDRegression
      기후지수 중 EOF 패턴형(PJ·NAO·AO·SRP)은 1991~2020 전체 달로 한 번 적합해 저장
검증: 연도 5묶음 블록 교차검증(EOF·회귀 폴드별 재적합) 1991~2025, 교차검증 잔차 σ·3분위 경계 저장
산출: seas5_monthly/models/{kind}_{var}.npz, KDS_WORK/cv_seas5_{kind}_{var}.nc
"""
import os, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd, xarray as xr
from joblib import Parallel, delayed
from sklearn.linear_model import ARDRegression

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "pyesd"))
import esd, predictors as pr  # noqa: E402

WORK = Path(os.environ.get("KDS_WORK", "/home/user/kds/work"))
CDS = Path(os.environ.get("CDS_DIR", "/mnt/project-files/korea_downscaling/data/seas5_cds"))
MODELS = HERE / "models"
TRAIN = (1991, 2025)
BLOCK = 5                     # 교차검증 연도 묶음


def load_target(kind, var):
    """월평균 예측대상 편차 (강수 sqrt), 1991~2020 기후값. 반환 DataArray(time, pt), lat, lon, meta"""
    if kind == "grid":
        ds = xr.open_dataset(WORK / "target_grid.nc")
        st = ds[{"T": "t2m", "P": "tp"}[var]].stack(pt=("lat", "lon"))
        st = st.isel(pt=np.where(ds.mask.stack(pt=("lat", "lon")).values)[0])
        lat, lon = st.lat.values, st.lon.values
        da = xr.DataArray(st.values, dims=("time", "pt"), coords={"time": st.time.values}); meta = {}
    else:
        ds = xr.open_dataset(WORK / "target_asos.nc")
        da = ds[var].rename(stn="pt"); lat, lon = ds.lat.values, ds.lon.values
        meta = dict(stn=ds.stn.values, name=ds.name.values.astype(str), jeju=ds.jeju.values)
        da = da.drop_vars([c for c in da.coords if c != "time"])
    mon = esd.transform(esd.aggregate(da, "month"), var)
    ya, yclim = esd.anomalies(mon, "month")
    ya = ya.assign_coords(time=pd.DatetimeIndex(ya.time.values).to_period("M").to_timestamp())
    return ya, lat, lon, meta, np.asarray(yclim(np.arange(1, 13))).reshape(12, -1), mon


def load_era5_anoms():
    F = pr.load_era5(CDS)
    c9120 = pr.era5_clim(F, pr.CLIM)
    A = pr.era5_anoms(F, c9120)
    return A, F["t2m"].lat.values, F["t2m"].lon.values


def standardize(X):
    mu = X.mean(0); sd = X.std(0) + 1e-9
    return (X - mu) / sd, mu, sd


def fit_point(X, y):
    """X (n,p) 표준화, y (n,). ARD → 원 단위 계수 (p+1,) [coef..., intercept]"""
    ok = np.isfinite(y)
    if ok.sum() < 20:
        return np.full(X.shape[1] + 1, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = ARDRegression(max_iter=300, threshold_lambda=1e4, fit_intercept=True).fit(X[ok], y[ok])
    return np.r_[m.coef_, m.intercept_]


def fit_points(X, Y, n_jobs=-1):
    """X (n,p,npt) 원 단위, Y (n,npt). 열 표준화 후 점별 ARD. 반환 coef (npt,p+1) 원 단위"""
    n, p, npt = X.shape
    def one(j):
        Z, mu, sd = standardize(X[:, :, j])
        b = fit_point(Z, Y[:, j])
        c = b[:-1] / sd
        return np.r_[c, b[-1] - (c * mu).sum()]
    return np.stack(Parallel(n_jobs=n_jobs, prefer="threads")(delayed(one)(j) for j in range(npt)))


def predict_points(coef, X):
    """coef (npt,p+1), X (n,p,npt) -> (n,npt)"""
    return np.einsum("npj,jp->nj", X, coef[:, :-1]) + coef[:, -1][None]


def run(kind, var):
    t0 = time.time()
    ya, lat, lon, meta, yclim, mon = load_target(kind, var)
    A, plat, plon = load_era5_anoms()
    t = ya.time.to_index().intersection(pd.DatetimeIndex(A["t2m"].time.values))
    t = t[(t.year >= TRAIN[0]) & (t.year <= TRAIN[1])]
    ya = ya.sel(time=t); A = {k: v.sel(time=t) for k, v in A.items()}
    Y = ya.values; yrs = t.year.values; mons = t.month.values
    lon2d, lat2d = np.meshgrid(plon, plat)
    Wt = pr.Weights(lat2d, lon2d, lat, lon)
    flat = {v: np.asarray(A[v]).reshape(len(t), -1) for v in pr.EOF_VARS}
    idx_names = pr.available_indices(A)
    clim_rows = np.where((yrs >= pr.CLIM[0]) & (yrs <= pr.CLIM[1]))[0]
    patterns = pr.Patterns().fit({k: v.isel(time=clim_rows) for k, v in A.items()})
    feats = pr.feature_names(idx_names, patterns)
    print(f"  features {len(feats)}: 지수 {idx_names} 패턴 {list(patterns.pat)}", flush=True)

    def features(mask, eofs):
        sub = {k: v.isel(time=np.where(mask)[0]) for k, v in A.items()}
        return pr.feature_table(sub, Wt, eofs, ("time",), idx_names, patterns)

    cv = np.full_like(Y, np.nan)
    out = {"lat": lat, "lon": lon, "plat": plat, "plon": plon, "W": Wt.W, "yclim": yclim,
           "features": np.array(feats), "local": np.array(pr.LOCAL), "idx": np.array(idx_names)}
    patterns.to_npz(out)
    out.update({k: np.asarray(v) for k, v in meta.items()})
    blocks = [(y0, min(y0 + BLOCK - 1, TRAIN[1])) for y0 in range(TRAIN[0], TRAIN[1] + 1, BLOCK)]
    for m in range(1, 13):
        win = esd.season_window(t, m, 1); cen = mons == m
        for b0, b1 in blocks:                                   # 블록 교차검증 (EOF도 폴드별)
            te = cen & (yrs >= b0) & (yrs <= b1); fit = win & ~((yrs >= b0) & (yrs <= b1))
            e = pr.EOFs().fit({v: flat[v][fit] for v in pr.EOF_VARS})
            c = fit_points(features(fit, e), Y[fit])
            cv[te] = predict_points(c, features(te, e))
        e = pr.EOFs().fit({v: flat[v][win] for v in pr.EOF_VARS})
        c = fit_points(features(win, e), Y[win])
        res = Y[cen] - cv[cen]
        out[f"m{m}_coef"] = c
        for v in pr.EOF_VARS:
            out[f"m{m}_mu_{v}"], out[f"m{m}_sd_{v}"], out[f"m{m}_eof_{v}"], out[f"m{m}_pcsd_{v}"] = e.mu[v], e.sd[v], e.eof[v], e.pcsd[v]
        out[f"m{m}_evr"] = np.concatenate([e.evr[v] for v in pr.EOF_VARS])
        out[f"m{m}_sigma"] = np.nanstd(res, 0)
        out[f"m{m}_q33"] = np.nanpercentile(Y[cen], 100 / 3, 0); out[f"m{m}_q67"] = np.nanpercentile(Y[cen], 200 / 3, 0)
        oc, cc = Y[cen], cv[cen]
        out[f"m{m}_r_cv"] = np.array([np.corrcoef(oc[np.isfinite(oc[:, j]), j], cc[np.isfinite(oc[:, j]), j])[0, 1]
                                      if np.isfinite(oc[:, j]).sum() > 5 else np.nan for j in range(Y.shape[1])])
        print(f"  seas5 {kind} {var} month {m:2d} r_cv={np.nanmedian(out[f'm{m}_r_cv']):.2f} {time.time()-t0:.0f}s", flush=True)
    if var == "P":                                             # 평년 강수 (mm/일, 월별) — 1991~2020 실제 평균
        pm = mon.sel(time=slice("1991", "2020")) ** 2
        out["normal"] = pm.groupby("time.month").mean().values
    MODELS.mkdir(exist_ok=True)
    np.savez_compressed(MODELS / f"{kind}_{var}.npz", **out)
    ds = xr.Dataset({"obs": (("time", "pt"), Y), "cv": (("time", "pt"), cv)},
                    coords={"time": t, "lat": ("pt", lat), "lon": ("pt", lon)})
    if var == "P":
        ds["clim_sqrt"] = (("time", "pt"), yclim[mons - 1])
    ds.to_netcdf(WORK / f"cv_seas5_{kind}_{var}.nc")
    print("done", kind, var, f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    k, v = (sys.argv[1:3] + [None, None])[:2]
    for kk in [k] if k else ["asos", "grid"]:
        for vv in [v] if v else ["T", "P"]:
            run(kk, vv)
