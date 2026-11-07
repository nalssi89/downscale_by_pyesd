"""1단계(기준망 83) vs 2단계(기준망 + 보강망 AWS) 비교.

  KDS_WORK=... python training/compare_networks.py OUTDIR

AWS 지점에서 (교차검증 기준)
  · 편차 예측: C 보강망 수축회귀 / A 보강망 독립 ARD / D 1단계 기준망 예측 내삽(역거리제곱 4지점)
  · 평년(1991~2020): 2단계 = 그 지점 환산 평년 / 1단계 = 기준망 평년 내삽(기온은 고도 감률 6.5 °C/km 보정, 강수는 역거리 가중)
  · 전체값(평년+편차) RMSE: 1단계(내삽 평년 + D) vs 2단계(자기 평년 + C)
5 km 격자 지도: 표본 달의 편차 내삽(1단계·2단계)과 가장 가까운 지점까지 거리(공백).
출력: OUTDIR/compare_*.png, compare_station.csv, compare_summary.csv
"""
import json, os, sys
from pathlib import Path
import numpy as np, pandas as pd, xarray as xr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MP

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "pyesd")); sys.path.insert(0, str(HERE.parent / "seas5_monthly"))
import mapbase  # noqa: E402
import esd  # noqa: E402

WORK = Path(os.environ.get("KDS_WORK", "/home/user/kds/aws"))
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else WORK / "compare"); OUT.mkdir(parents=True, exist_ok=True)
GEO = HERE.parent / "site" / "data" / "geo" / "korea.geojson"
plt.rcParams.update({"font.family": "WenQuanYi Zen Hei", "axes.unicode_minus": False, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})
C1, C2, C3 = "#eb6834", "#2a78d6", "#1baf7a"


def dist(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.deg2rad, (la1, lo1, la2, lo2))
    return 6371 * np.arccos(np.clip(np.sin(la1) * np.sin(la2) + np.cos(la1) * np.cos(la2) * np.cos(lo1 - lo2), -1, 1))


def idw_w(d, k=4):
    W = np.zeros_like(d)
    for j in range(d.shape[0]):
        nn = np.argsort(d[j])[:k]; W[j, nn] = 1 / np.maximum(d[j, nn], 1.0) ** 2
    return W / W.sum(1, keepdims=True)


def msess(o, p):
    k = np.isfinite(o) & np.isfinite(p)
    return 1 - ((o[k] - p[k]) ** 2).sum() / (o[k] ** 2).sum() if k.sum() > 10 else np.nan


def corr(o, p):
    k = np.isfinite(o) & np.isfinite(p)
    return np.corrcoef(o[k], p[k])[0, 1] if k.sum() > 10 else np.nan


tb = xr.open_dataset(WORK / "target_base.nc"); ta = xr.open_dataset(WORK / "target_aws.nc")
bst = pd.read_csv(HERE / "base_stations.csv"); ast = pd.read_csv(HERE / "aws_stations.csv")
d_ab = dist(ast.lat.values[:, None], ast.lon.values[:, None], bst.lat.values[None], bst.lon.values[None])
W = idw_w(d_ab)
rows, summ = [], []
for var in ("T", "P"):
    cv = xr.open_dataset(WORK / f"cv_seas5_aws_{var}.nc")
    cb = xr.open_dataset(WORK / f"cv_seas5_base_{var}.nc")
    o = cv.obs.values; mon = cv.time.dt.month.values
    # 평년: 기준망 1991~2020 (자기 평년, 보정 포함) — base cv 파일에는 없으므로 target에서 다시 계산
    bm = esd.transform(tb[var], var).sel(time=slice("1991", "2020"))
    bclim = bm.groupby("time.month").mean().values                       # (12, nb) — 15지점은 자기 기간 평균(근사)
    aclim = cv.clim.values                                                # (12, na) 환산 평년
    if var == "T":
        alt_b, alt_a = bst.alt.values, ast.alt.values
        interp = np.stack([(W * (bclim[k][None] - 0.0065 * (alt_a[:, None] - alt_b[None]))).sum(1) for k in range(12)])
    else:                                                                 # 강수 없는 기준망 지점(공항 5곳)은 빼고 가중
        interp = np.stack([np.nansum(W * bclim[k][None], 1) / np.sum(W * np.isfinite(bclim[k])[None], 1) for k in range(12)])
    tot_o = o + aclim[mon - 1]
    t1 = interp[mon - 1] + cv.cv_D.values; t2 = aclim[mon - 1] + cv.cv_C.values
    th = aclim[mon - 1] + cv.cv_D.values                                  # 절충: 기준망 편차 내삽 + AWS 평년
    if var == "P":                                                        # sqrt → mm/일 로 되돌려 비교
        tot_o, t1, t2, th = (np.clip(x, 0, None) ** 2 for x in (tot_o, t1, t2, th))
    for j, s in ast.iterrows():
        r = dict(var=var, stn=s.STN, name=s["name"], lat=s.lat, lon=s.lon, alt=s.alt, nbr_km=float(cv.nbr_km[j]),
                 msess_C=msess(o[:, j], cv.cv_C.values[:, j]), msess_A=msess(o[:, j], cv.cv_A.values[:, j]),
                 msess_D=msess(o[:, j], cv.cv_D.values[:, j]),
                 r_C=corr(o[:, j], cv.cv_C.values[:, j]), r_D=corr(o[:, j], cv.cv_D.values[:, j]),
                 normal_err=float(np.nanmean(np.abs(interp[:, j] - aclim[:, j]))) if var == "T"
                 else float(np.nanmean(np.abs(interp[:, j] ** 2 / np.maximum(aclim[:, j] ** 2, 1e-6) - 1))),
                 rmse_tot1=float(np.sqrt(np.nanmean((t1[:, j] - tot_o[:, j]) ** 2))),
                 rmse_tot2=float(np.sqrt(np.nanmean((t2[:, j] - tot_o[:, j]) ** 2))),
                 rmse_hyb=float(np.sqrt(np.nanmean((th[:, j] - tot_o[:, j]) ** 2))))
        rows.append(r)
R = pd.DataFrame(rows); R.to_csv(OUT / "compare_station.csv", index=False)
S = R.groupby("var")[["msess_C", "msess_A", "msess_D", "r_C", "r_D", "normal_err", "rmse_tot1", "rmse_tot2", "rmse_hyb"]].median()
S["n"] = R.groupby("var").size(); S.round(3).to_csv(OUT / "compare_summary.csv"); print(S.round(3).to_string())

# ---- 그림 1: AWS 지점 성능 비교 (편차 MSESS, 전체값 RMSE)
fig, axs = plt.subplots(1, 3, figsize=(13, 4), dpi=150)
for ax, var, unit in zip(axs[:2], ("T", "P"), ("기온", "강수(√)")):
    x = R[R["var"] == var]
    ax.scatter(x.msess_D, x.msess_C, s=10, c=C2, alpha=0.6, edgecolors="none")
    lo = min(x.msess_D.min(), x.msess_C.min(), 0); ax.plot([lo, 1], [lo, 1], color="#999", lw=1)
    ax.set_xlabel("1단계: 기준망 예측 내삽 MSESS"); ax.set_ylabel("2단계: 보강망 수축회귀 MSESS")
    ax.set_title(f"{unit} 편차 (AWS {len(x)}곳, 교차검증)", loc="left", fontsize=11)
    ax.text(0.03, 0.95, f"2단계가 나은 곳 {int((x.msess_C > x.msess_D).mean() * 100)}%", transform=ax.transAxes, va="top", fontsize=9)
ax = axs[2]
x = R[R["var"] == "T"]
ax.scatter(x.rmse_tot1, x.rmse_tot2, s=10, c=C1, alpha=0.6, edgecolors="none")
m = max(x.rmse_tot1.quantile(0.99), x.rmse_tot2.max()); ax.plot([0, m], [0, m], color="#999", lw=1); ax.set_xlim(0, m); ax.set_ylim(0, m)
ax.set_xlabel("1단계 (내삽 평년 + 내삽 편차) RMSE °C"); ax.set_ylabel("2단계 (자기 평년 + 수축회귀) RMSE °C")
ax.set_title("기온 월평균값 자체의 오차", loc="left", fontsize=11)
plt.tight_layout(); plt.savefig(OUT / "compare_skill.png"); plt.close()

# ---- 그림 2: 평년 내삽 오차 지도 (1단계로는 놓치는 국지 평년)
x = R[R["var"] == "T"]
fig, axs = plt.subplots(1, 2, figsize=(11, 6.2), dpi=150)
for ax, col, tt, cm, vmax in ((axs[0], "normal_err", "기온 평년 내삽 오차 |°C| (1단계, 고도 보정 포함)", "Oranges", 1.5),
                              (axs[1], None, "강수 평년 내삽 오차 |비율-1| (1단계)", "Blues", 0.3)):
    xx = x if col else R[R["var"] == "P"]
    mapbase.draw(ax, box=(124.5, 33, 131.2, 38.7), color="#777", lw=0.4)
    sc = ax.scatter(xx.lon, xx.lat, c=xx.normal_err, s=14, cmap=cm, vmin=0, vmax=vmax, edgecolors="#555", linewidths=0.2, zorder=3)
    ax.scatter(bst.lon, bst.lat, s=18, marker="^", c="k", zorder=4, label="기준망")
    plt.colorbar(sc, ax=ax, shrink=0.7); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(tt, loc="left", fontsize=10)
    ax.legend(loc="lower right", fontsize=8)
plt.tight_layout(); plt.savefig(OUT / "compare_normal_error.png"); plt.close()

# ---- 그림 3: 5 km 격자 — 공백(가장 가까운 지점 거리)과 표본 달 편차 내삽
polys = []
for f in json.load(open(GEO, encoding="utf-8"))["features"]:
    if f["properties"].get("name") != "South Korea":
        continue
    g = f["geometry"]
    for poly in ([g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]):
        polys.append(MP(np.asarray(poly[0])))
glat = np.arange(33.1, 38.65, 0.045); glon = np.arange(124.6, 131.0, 0.057)          # 약 5 km
LO, LA = np.meshgrid(glon, glat); pts = np.c_[LO.ravel(), LA.ravel()]
inside = np.zeros(len(pts), bool)
for P_ in polys:
    inside |= P_.contains_points(pts)
mask = inside.reshape(LA.shape)                                                    # 남한 육지
d1 = dist(pts[:, 1][:, None], pts[:, 0][:, None], bst.lat.values[None], bst.lon.values[None]).min(1).reshape(LA.shape)
allst = pd.concat([bst[["lat", "lon"]], ast[["lat", "lon"]]])
d2 = dist(pts[:, 1][:, None], pts[:, 0][:, None], allst.lat.values[None], allst.lon.values[None]).min(1).reshape(LA.shape)
cvT = xr.open_dataset(WORK / "cv_seas5_aws_T.nc"); cbT = xr.open_dataset(WORK / "cv_seas5_base_T.nc")
tsel = pd.Timestamp("2025-03-01")
pb = cbT.cv.sel(time=tsel).values; pa = cvT.cv_C.sel(time=tsel).values


def grid_idw(vals, lat, lon, k=6):
    dd = dist(pts[:, 1][:, None], pts[:, 0][:, None], lat[None], lon[None])
    ok = np.isfinite(vals); dd = dd[:, ok]; v = vals[ok]
    nn = np.argsort(dd, 1)[:, :k]; w = 1 / np.maximum(np.take_along_axis(dd, nn, 1), 1.0) ** 2
    return ((v[nn] * w).sum(1) / w.sum(1)).reshape(LA.shape)


g1 = grid_idw(pb, bst.lat.values, bst.lon.values)
g2 = grid_idw(np.r_[pb, pa], np.r_[bst.lat.values, ast.lat.values], np.r_[bst.lon.values, ast.lon.values])
fig, axs = plt.subplots(2, 2, figsize=(11, 11.5), dpi=150)
box = (124.5, 33, 131.2, 38.7)
vm = float(np.nanpercentile(np.abs(np.where(mask, g1, np.nan)), 98)); dv = float(np.nanpercentile(np.abs(np.where(mask, g2 - g1, np.nan)), 98))
for ax, z, tt, cm, vmin, vmax in ((axs[0, 0], d1, f"1단계 공백: 가장 가까운 지점까지 거리(km), 20 km 초과 {int((d1[mask] > 20).mean() * 100)}%", "magma_r", 0, 40),
                                  (axs[0, 1], d2, f"2단계 공백: 같은 지표, 20 km 초과 {int((d2[mask] > 20).mean() * 100)}%", "magma_r", 0, 40),
                                  (axs[1, 0], g1, f"1단계 5 km 내삽: {tsel:%Y년 %m월} 기온 편차 예측(°C)", "RdBu_r", -vm, vm),
                                  (axs[1, 1], g2 - g1, "2단계 - 1단계 차이(°C): 보강망이 바꾸는 부분", "PuOr_r", -dv, dv)):
    zz = np.where(mask, z, np.nan)
    pc = ax.pcolormesh(glon, glat, zz, cmap=cm, vmin=vmin, vmax=vmax, shading="auto")
    mapbase.draw(ax, box=box, color="#555", lw=0.4); plt.colorbar(pc, ax=ax, shrink=0.7)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(tt, loc="left", fontsize=10)
axs[0, 0].scatter(bst.lon, bst.lat, s=6, c="w", edgecolors="k", linewidths=0.3)
axs[0, 1].scatter(allst.lon, allst.lat, s=3, c="w", edgecolors="k", linewidths=0.2)
plt.tight_layout(); plt.savefig(OUT / "compare_map_5km.png"); plt.close()
pd.DataFrame({"gap20_step1": [(d1[mask] > 20).mean()], "gap20_step2": [(d2[mask] > 20).mean()],
              "gap_med_step1": [np.median(d1[mask])], "gap_med_step2": [np.median(d2[mask])]}).round(3).to_csv(OUT / "compare_gap.csv", index=False)
print("gap", (d1[mask] > 20).mean(), (d2[mask] > 20).mean(), np.median(d1[mask]), np.median(d2[mask]))
