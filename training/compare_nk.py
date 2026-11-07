"""북한 27지점 점검: (1) ERA5가 북한 관측을 얼마나 재현하는가(격자 목표자료의 신뢰도), (2) 북한 지점 SEAS5 월별 전이함수 교차검증.
남한 ASOS 66을 같은 방식으로 계산해 기준으로 삼는다.

  KDS_WORK=/home/user/kds/nk BASE_WORK=/home/user/kds/base python training/compare_nk.py OUTDIR
입력: KDS_WORK/target_nk.nc, cv_seas5_nk_{T,P}.nc / BASE_WORK/target_base.nc, cv_seas5_base_{T,P}.nc / CDS_DIR/era5_sfc.nc (1°)
출력: OUTDIR/nk_station.csv, nk_summary.csv, nk_skill_map.png, nk_era5_scatter.png
"""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd, xarray as xr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mapbase  # noqa: E402

WORK = Path(os.environ.get("KDS_WORK", "/home/user/kds/nk")); BASE = Path(os.environ.get("BASE_WORK", "/home/user/kds/base"))
CDS = Path(os.environ.get("CDS_DIR", "/mnt/project-files/korea_downscaling/data/seas5_cds"))
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else WORK / "compare"); OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "WenQuanYi Zen Hei", "axes.unicode_minus": False, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})
CN, CS = "#d6452a", "#2a78d6"


def msess(o, p):
    k = np.isfinite(o) & np.isfinite(p)
    return 1 - ((o[k] - p[k]) ** 2).sum() / (o[k] ** 2).sum() if k.sum() > 24 else np.nan


def corr(o, p):
    k = np.isfinite(o) & np.isfinite(p)
    return np.corrcoef(o[k], p[k])[0, 1] if k.sum() > 24 else np.nan


def anom(a, t):
    """a (time, n), 1991~2020 달별 평균 제거"""
    a = np.array(a, float); m = t.month.values; c = (t.year >= 1991) & (t.year <= 2020)
    out = a.copy(); clim = np.zeros((12, a.shape[1]))
    for k in range(1, 13):
        clim[k - 1] = np.nanmean(a[c & (m == k)], 0); out[m == k] -= clim[k - 1]
    return out, clim


e5 = xr.open_dataset(CDS / "era5_sfc.nc").rename(valid_time="time", latitude="lat", longitude="lon")[["t2m", "tp"]]
e5 = e5.sel(lat=slice(46, 30), lon=slice(120, 135)).load()
groups = {}
for lab, f, cvdir, kind in (("북한", WORK / "target_nk.nc", WORK, "nk"), ("남한 ASOS 66", BASE / "target_base.nc", BASE, "base")):
    ds = xr.open_dataset(f)
    if "asos66" in ds:
        ds = ds.isel(stn=np.where(ds.asos66.values == 1)[0])
    ds = ds.sel(time=slice("1991", "2025"))
    t = pd.DatetimeIndex(ds.time.values)
    x = e5.sel(time=t).interp(lat=xr.DataArray(ds.lat.values, dims="stn"), lon=xr.DataArray(ds.lon.values, dims="stn"))
    rows = []
    for var, ev, conv in (("T", "t2m", lambda v: v - 273.15), ("P", "tp", lambda v: v * 1000)):
        o, oc = anom(ds[var].values, t); m, mc = anom(conv(x[ev].values), t)
        cv = xr.open_dataset(cvdir / f"cv_seas5_{kind}_{var}.nc")
        cvsel = cv.isel(pt=np.where(xr.open_dataset(f).asos66.values == 1)[0]) if kind == "base" else cv
        for j in range(ds.sizes["stn"]):
            rows.append(dict(group=lab, var=var, stn=int(ds.stn.values[j]), name=str(ds.name.values[j]), lat=float(ds.lat.values[j]),
                             lon=float(ds.lon.values[j]), alt=float(ds.alt.values[j]),
                             era5_r=corr(o[:, j], m[:, j]),
                             era5_bias=np.nanmean(mc[:, j] - oc[:, j]) if var == "T" else np.nansum(mc[:, j]) / np.nansum(oc[:, j]),
                             cv_msess=msess(cvsel.obs.values[:, j], cvsel.cv.values[:, j]), cv_r=corr(cvsel.obs.values[:, j], cvsel.cv.values[:, j]),
                             miss=float(np.isnan(ds[var].values[:, j]).mean())))
    groups[lab] = rows
st = pd.DataFrame([r for v in groups.values() for r in v])
st.to_csv(OUT / "nk_station.csv", index=False)
summ = st.groupby(["group", "var"])[["era5_r", "era5_bias", "cv_msess", "cv_r", "miss"]].median().round(3)
summ.to_csv(OUT / "nk_summary.csv"); print(summ)

# 지도: ERA5 재현 상관, 교차검증 MSESS (기온·강수)
fig, axs = plt.subplots(2, 2, figsize=(10, 11))
for i, var in enumerate(("T", "P")):
    for k, (col, ttl, vr) in enumerate((("era5_r", "ERA5(1°) 월편차 상관", (0.3, 1.0) if var == "P" else (0.8, 1.0)),
                                       ("cv_msess", "SEAS5 전이함수 교차검증 MSESS", (0.2, 0.8) if var == "P" else (0.7, 0.95)))):
        ax = axs[i, k]; s = st[st["var"] == var]
        mapbase.draw(ax, box=(124.0, 33.0, 131.0, 43.0), color="#888", lw=0.4)
        sc = ax.scatter(s.lon, s.lat, c=s[col], s=np.where(s.group == "북한", 60, 30), cmap="viridis", vmin=vr[0], vmax=vr[1],
                        edgecolor=np.where(s.group == "북한", "k", "none"), lw=0.6)
        plt.colorbar(sc, ax=ax, shrink=0.7)
        nk, sk = s[s.group == "북한"][col].median(), s[s.group != "북한"][col].median()
        ax.set_title(f"{'기온' if var == 'T' else '강수'} {ttl}\n중앙값 북한 {nk:.2f} · 남한 {sk:.2f}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("북한 27지점(테두리, 2001~2025) vs 남한 ASOS 66(1991~2025) 월자료", fontsize=12)
fig.tight_layout(); fig.savefig(OUT / "nk_skill_map.png", dpi=130); plt.close(fig)
print("saved", OUT)
