"""2단계 보강망(AWS) 전이함수 학습 — 이웃 기준망 계수 쪽으로 수축하는 능형회귀.

  KDS_WORK=... CDS_DIR=... python seas5_monthly/train_aws.py [T|P]

입력: KDS_WORK/target_base.nc(기준망 83, training/build_base_network.py), KDS_WORK/target_aws.nc(보강망, training/build_aws_network.py)
      ERA5 1° 예측인자(train.py 와 같음)
방법 (달별 ±1개월 창, 연도 5묶음 블록 교차검증, EOF·기준망 계수도 폴드별 재적합):
  기준망: train.py 와 같은 점별 ARD.
  보강망 C(운영): 설계행렬 열 표준화 후  min |y - Zb|² + λ|b - b0|²,  b0 = 가장 가까운 기준망 지점 계수(원 단위)를 이 지점의
          열 표준편차·편차 표준편차 비로 옮긴 값, λ 는 GCV(1~1000). 자료가 짧거나 잡음이 크면 이웃 계수에 가까워진다.
  비교용 A: 보강망 지점 독립 ARD (1단계 방식 그대로 적용).
  비교용 D: 1단계만 쓸 때 — 기준망 예측을 가까운 4지점 역거리제곱 가중으로 AWS 위치에 내삽.
평년: 보강망은 1991년부터 자료가 없으므로 가장 가까운 '온전한' 기준망 지점과의 공통 기간 차(기온)·비(강수 sqrt)로 1991~2020 평년 환산.
산출: seas5_monthly/models/aws_{var}.npz (train.py 와 같은 형식, 계수 = C), KDS_WORK/cv_seas5_aws_{var}.nc (obs, cv_C, cv_A, cv_D)
"""
import sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd, xarray as xr
from joblib import Parallel, delayed

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "pyesd"))
import esd, predictors as pr  # noqa: E402
import train as tr  # noqa: E402

LAMBDAS = np.array([1, 3, 10, 30, 100, 300, 1000.0])
K_IDW = 4


def dist(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.deg2rad, (la1, lo1, la2, lo2))
    return 6371 * np.arccos(np.clip(np.sin(la1) * np.sin(la2) + np.cos(la1) * np.cos(la2) * np.cos(lo1 - lo2), -1, 1))


def load_aws(var, base_mon, blat, blon):
    ds = xr.open_dataset(tr.WORK / "target_aws.nc")
    da = ds[var].rename(stn="pt"); da = da.drop_vars([c for c in da.coords if c != "time"])
    mon = esd.transform(da, var)
    lat, lon = ds.lat.values, ds.lon.values
    # 기준망의 온전한 지점(1991~2020 달별 27년 이상)
    cb = base_mon.sel(time=slice(*esd.CLIM)); mb = cb["time.month"].values
    full = np.stack([np.isfinite(cb.values[mb == k]).sum(0) for k in range(1, 13)]).min(0) >= 27
    bclim = np.stack([np.nanmean(cb.values[mb == k], 0) for k in range(1, 13)])
    d = dist(lat[:, None], lon[:, None], blat[None], blon[None]); d[:, ~full] = np.inf
    nb = d.argmin(1)
    ma = mon["time.month"].values
    clim = np.full((12, len(lat)), np.nan)
    a, b = mon.values, base_mon.reindex(time=mon.time).values
    for j in range(len(lat)):
        n = nb[j]
        for k in range(1, 13):
            both = (ma == k) & np.isfinite(a[:, j]) & np.isfinite(b[:, n])
            if both.sum() < 5:
                continue
            x, y = a[both, j].mean(), b[both, n].mean()
            clim[k - 1, j] = bclim[k - 1, n] * (x / y) if (var == "P" and y > 0) else bclim[k - 1, n] + (x - y)
    mclim = xr.DataArray(clim, dims=("month", "pt"), coords={"month": np.arange(1, 13)})
    ya = mon.groupby("time.month") - mclim
    meta = dict(stn=ds.stn.values, name=ds.name.values.astype(str), jeju=ds.jeju.values)
    return ya, lat, lon, meta, clim, mon


def shrink_fit(X, y, c0, ysd0):
    """X (n,p) 원 단위, y (n,), c0 이웃 기준망 원 단위 계수 (p+1,), ysd0 이웃 편차 표준편차 → 원 단위 계수 (p+1,)"""
    ok = np.isfinite(y)
    if ok.sum() < 12 or not np.isfinite(c0).all():
        return np.full(X.shape[1] + 1, np.nan)
    X, y = X[ok], y[ok]
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd; ym = y.mean(); yc = y - ym
    ratio = yc.std() / ysd0 if ysd0 > 0 else 1.0
    b0 = c0[:-1] * sd * ratio
    G = Z.T @ Z; zy = Z.T @ yc; n, p = Z.shape
    best = None
    for lam in LAMBDAS:
        A = G + lam * np.eye(p)
        b = np.linalg.solve(A, zy + lam * b0)
        r = yc - Z @ b
        tr_h = np.trace(np.linalg.solve(A, G))
        g = (r @ r) / max(n - tr_h, 1.0) ** 2 * n
        if best is None or g < best[0]:
            best = (g, b)
    c = best[1] / sd
    return np.r_[c, ym - (c * mu).sum()]


def idw(pred_b, d_ab):
    """기준망 예측 (n, nb) → AWS 위치 (n, na), 가까운 K_IDW 지점 역거리제곱."""
    out = np.zeros((pred_b.shape[0], d_ab.shape[0]))
    for j in range(d_ab.shape[0]):
        nn = np.argsort(d_ab[j])[:K_IDW]; w = 1 / np.maximum(d_ab[j, nn], 1.0) ** 2
        p = pred_b[:, nn]; ok = np.isfinite(p)
        out[:, j] = np.nansum(p * w, 1) / np.maximum((ok * w).sum(1), 1e-9)
    return out


def run(var):
    t0 = time.time()
    yb, blat, blon, bmeta, bclim, bmon = tr.load_target("base", var)
    ya, alat, alon, ameta, aclim, amon = load_aws(var, bmon, blat, blon)
    A, plat, plon = tr.load_era5_anoms()
    t = yb.time.to_index().intersection(pd.DatetimeIndex(A["t2m"].time.values))
    t = t[(t.year >= tr.TRAIN[0]) & (t.year <= tr.TRAIN[1])]
    ya = ya.assign_coords(time=pd.DatetimeIndex(ya.time.values).to_period("M").to_timestamp())
    Yb = yb.sel(time=t).values; Ya = ya.sel(time=t).values
    A = {k: v.sel(time=t) for k, v in A.items()}
    yrs, mons = t.year.values, t.month.values
    nb_, na_ = len(blat), len(alat)
    lat, lon = np.r_[blat, alat], np.r_[blon, alon]
    lon2d, lat2d = np.meshgrid(plon, plat)
    Wt = pr.Weights(lat2d, lon2d, lat, lon)
    flat = {v: np.asarray(A[v]).reshape(len(t), -1) for v in pr.EOF_VARS}
    idx_names = pr.available_indices(A)
    clim_rows = np.where((yrs >= pr.CLIM[0]) & (yrs <= pr.CLIM[1]))[0]
    patterns = pr.Patterns().fit({k: v.isel(time=clim_rows) for k, v in A.items()})
    feats = pr.feature_names(idx_names, patterns)
    d_ab = dist(alat[:, None], alon[:, None], blat[None], blon[None])
    nbr = d_ab.argmin(1)
    print(f"  {var}: 기준망 {nb_} + 보강망 {na_}, features {len(feats)}", flush=True)

    def features(mask, eofs):
        sub = {k: v.isel(time=np.where(mask)[0]) for k, v in A.items()}
        return pr.feature_table(sub, Wt, eofs, ("time",), idx_names, patterns)

    def fit_C(X, Ya_, cb, Yb_):
        ysd = np.nanstd(Yb_, 0)
        return np.stack(Parallel(n_jobs=-1, prefer="threads")(
            delayed(shrink_fit)(X[:, :, nb_ + j], Ya_[:, j], cb[nbr[j]], ysd[nbr[j]]) for j in range(na_)))

    cvC, cvA, cvD = (np.full_like(Ya, np.nan) for _ in range(3))
    blocks = [(y0, min(y0 + tr.BLOCK - 1, tr.TRAIN[1])) for y0 in range(tr.TRAIN[0], tr.TRAIN[1] + 1, tr.BLOCK)]
    W_idx, W_val, ngrid = pr.pack_W(Wt.W)
    out = {"lat": alat, "lon": alon, "plat": plat, "plon": plon, "features": np.array(feats), "local": np.array(pr.LOCAL),
           "idx": np.array(idx_names), "yclim": aclim, "nbr_base_stn": bmeta["stn"][nbr]}
    patterns.to_npz(out); out.update({k: np.asarray(v) for k, v in ameta.items()})
    Wa = pr.Weights(lat2d, lon2d, alat, alon); out["W_idx"], out["W_val"], out["W_ngrid"] = pr.pack_W(Wa.W)
    for m in range(1, 13):
        win = esd.season_window(t, m, 1); cen = mons == m
        for b0, b1 in blocks:
            te = cen & (yrs >= b0) & (yrs <= b1); fit = win & ~((yrs >= b0) & (yrs <= b1))
            e = pr.EOFs().fit({v: flat[v][fit] for v in pr.EOF_VARS})
            Xf, Xt = features(fit, e), features(te, e)
            cb = tr.fit_points(Xf[:, :, :nb_], Yb[fit])
            pb = tr.predict_points(cb, Xt[:, :, :nb_])
            cvD[te] = idw(pb, d_ab)
            cC = fit_C(Xf, Ya[fit], cb, Yb[fit]); cvC[te] = tr.predict_points(cC, Xt[:, :, nb_:])
            cA = tr.fit_points(Xf[:, :, nb_:], Ya[fit]); cvA[te] = tr.predict_points(cA, Xt[:, :, nb_:])
        e = pr.EOFs().fit({v: flat[v][win] for v in pr.EOF_VARS})
        Xw = features(win, e)
        cb = tr.fit_points(Xw[:, :, :nb_], Yb[win])
        c = fit_C(Xw, Ya[win], cb, Yb[win])
        out[f"m{m}_coef"] = c
        for v in pr.EOF_VARS:
            out[f"m{m}_mu_{v}"], out[f"m{m}_sd_{v}"] = e.mu[v].astype(np.float32), e.sd[v].astype(np.float32)
            out[f"m{m}_eof_{v}"], out[f"m{m}_pcsd_{v}"] = e.eof[v].astype(np.float32), e.pcsd[v]
        out[f"m{m}_evr"] = np.concatenate([e.evr[v] for v in pr.EOF_VARS])
        out[f"m{m}_sigma"] = np.nanstd(Ya[cen] - cvC[cen], 0)
        yc = Ya[cen & (yrs >= pr.CLIM[0]) & (yrs <= pr.CLIM[1])]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for nm_, qq in (("q10", 10), ("q33", 100 / 3), ("q50", 50), ("q67", 200 / 3), ("q90", 90)):
                out[f"m{m}_{nm_}"] = np.nanpercentile(yc, qq, 0)
        oc = Ya[cen]
        r = lambda cc: np.nanmedian([np.corrcoef(oc[np.isfinite(oc[:, j]) & np.isfinite(cc[:, j]), j],
                                                 cc[np.isfinite(oc[:, j]) & np.isfinite(cc[:, j]), j])[0, 1]
                                     for j in range(na_) if (np.isfinite(oc[:, j]) & np.isfinite(cc[:, j])).sum() > 5])
        print(f"  aws {var} month {m:2d} r_cv C={r(cvC[cen]):.2f} A={r(cvA[cen]):.2f} D={r(cvD[cen]):.2f} {time.time()-t0:.0f}s", flush=True)
    if var == "P":
        pm = amon.sel(time=slice("1991", "2020")) ** 2
        out["normal"] = (xr.DataArray(aclim, dims=("month", "pt")) ** 2).values   # 환산 평년(mm/일)
    tr.MODELS.mkdir(exist_ok=True)
    np.savez_compressed(tr.MODELS / f"aws_{var}.npz", **out)
    xr.Dataset({"obs": (("time", "pt"), Ya), "cv_C": (("time", "pt"), cvC), "cv_A": (("time", "pt"), cvA),
                "cv_D": (("time", "pt"), cvD), "clim": (("month", "pt"), aclim)},
               coords={"time": t, "lat": ("pt", alat), "lon": ("pt", alon), "stn": ("pt", ameta["stn"]),
                       "nbr_base": ("pt", bmeta["stn"][nbr]), "nbr_km": ("pt", d_ab.min(1))}
               ).to_netcdf(tr.WORK / f"cv_seas5_aws_{var}.nc")
    print("done aws", var, f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    for v in (sys.argv[1:2] or ["T", "P"]):
        run(v)
