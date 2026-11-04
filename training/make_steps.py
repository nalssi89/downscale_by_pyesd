"""PyESD 단계별 적용 과정 그림 (fig_s1 ~ fig_s6).
  python make_steps.py
입력: WORK의 model_*.pkl, pred.nc, cv_*.nc, target_asos.nc + 예보 JSON(KDS_FC)
"""
import os, sys, json, pickle
from pathlib import Path
import numpy as np, pandas as pd, xarray as xr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pyesd"))
import plots, mapbase, esd
from plots import WORK, FIG, INK, INK2, GRID, C_ESD, C_RAW, DIV_T, DIV_P, SEQ, mapax, field, BOX

FC = Path(os.environ.get("KDS_FC", "/mnt/project-files/korea_downscaling/forecast/openmeteo_20261104/downscaling.json"))
WIDE = (113.0, 27.0, 142.0, 47.0)
PVN = {"t2m": "기온 t2m", "d2m": "이슬점 d2m", "msl": "해면기압 msl", "tp": "강수 tp (sqrt)"}
NMON = ["1월", "2월", "3월", "4월", "5월", "6월", "7월", "8월", "9월", "10월", "11월", "12월"]


def load(kind, var, scale):
    return pickle.load(open(WORK / f"model_{kind}_{var}_{scale}.pkl", "rb"))


def wideax(ax, title=None):
    mapbase.draw(ax, box=WIDE, color="#55554f", lw=0.5)
    ax.set_xlim(WIDE[0], WIDE[2]); ax.set_ylim(WIDE[1], WIDE[3])
    ax.set_xticks([115, 120, 125, 130, 135, 140]); ax.set_yticks([30, 35, 40, 45])
    ax.tick_params(labelsize=7)
    ax.add_patch(plt.Rectangle((BOX[0], BOX[1]), BOX[2] - BOX[0], BOX[3] - BOX[1], fill=False, ec=C_ESD, lw=1))
    if title:
        ax.set_title(title, loc="left", color=INK)


def coarse(ax, vals, plat, plon, cmap, vmin, vmax):
    la, lo = np.unique(plat), np.unique(plon)
    G = np.full((len(la), len(lo)), np.nan)
    G[np.searchsorted(la, plat), np.searchsorted(lo, plon)] = vals
    d = 0.75
    return ax.pcolormesh(np.r_[lo - d, lo[-1] + d], np.r_[la - d, la[-1] + d], G, cmap=cmap,
                         norm=mcolors.Normalize(vmin, vmax), shading="flat", alpha=0.9)


# ---------------------------------------------------------------- S1 예측대상(predictand)
def fig_s1():
    mT, mP = load("grid", "T", "week"), load("grid", "P", "week")
    asos = xr.open_dataset(WORK / "target_asos.nc")
    fig, axs = plt.subplots(1, 4, figsize=(13, 4.2))
    for ax, doy, ttl in zip(axs[:2], [15, 196], ["(a) 1월 기후값 기온 (°C)", "(b) 7월 기후값 기온 (°C)"]):
        mapax(ax, ttl)
        m = field(ax, mT["yclim"][doy], mT["lat"], mT["lon"], DIV_T, -15, 30)
        plt.colorbar(m, ax=ax, shrink=0.8, pad=0.02).ax.tick_params(labelsize=7)
    mapax(axs[2], "(c) 7월 기후값 강수 (mm/일)")
    m = field(axs[2], mP["yclim"][196] ** 2, mP["lat"], mP["lon"], SEQ, 0, 14)
    plt.colorbar(m, ax=axs[2], shrink=0.8, pad=0.02).ax.tick_params(labelsize=7)
    mapax(axs[3], "(d) ASOS 62지점 연평균기온 (°C)")
    tm = asos["T"].sel(time=slice("1991", "2020")).mean("time").values
    sc = axs[3].scatter(asos.lon, asos.lat, c=tm, cmap=DIV_T, vmin=5, vmax=17, s=28, ec="k", lw=0.3)
    plt.colorbar(sc, ax=axs[3], shrink=0.8, pad=0.02).ax.tick_params(labelsize=7)
    fig.suptitle("1단계 예측대상: ERA5 0.25° 육지 351격자(북한 포함) + ASOS 62지점, 1991~2020 조화함수 기후값", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(); fig.savefig(FIG / "fig_s1_predictand.png"); plt.close(fig)


# ---------------------------------------------------------------- S2 예측인자(predictor) 편차장 + 반경 평균
def fig_s2():
    m = load("grid", "T", "week")
    ds = xr.open_dataset(WORK / "pred.nc")
    t1 = pd.Timestamp("2026-10-25")
    wk = ds.sel(time=slice(t1 - pd.Timedelta(days=3), t1 + pd.Timedelta(days=3))).mean("time")
    doy = t1.dayofyear
    plat, plon = m["plat"], m["plon"]
    fig, axs = plt.subplots(1, 5, figsize=(16, 3.9))
    spec = {"t2m": (DIV_T, -4, 4, "°C"), "d2m": (DIV_T, -4, 4, "°C"), "msl": (DIV_T, -6, 6, "hPa"), "tp": (DIV_P, -1.5, 1.5, "sqrt(mm/일)")}
    for ax, v in zip(axs[:4], esd.PVARS):
        raw = wk[v].values.ravel()
        raw = np.sqrt(np.clip(raw, 0, None)) if v == "tp" else raw
        a = raw - m["xclim"][v][doy]
        if v == "msl":
            a = a / 100.0
        cmap, lo, hi, unit = spec[v]
        wideax(ax, f"{PVN[v]} 편차 ({unit})")
        c = coarse(ax, a, plat, plon, cmap, lo, hi)
        plt.colorbar(c, ax=ax, shrink=0.75, pad=0.02).ax.tick_params(labelsize=7)
    # 반경 250 km 가우스 가중치 예시 (평양 부근 격자)
    ax = axs[4]
    i = int(np.argmin((m["lat"] - 39.0) ** 2 + (m["lon"] - 125.75) ** 2))
    w = m["W"][i]
    wideax(ax, "반경 250 km 가우스 가중 (예: 39°N 125.75°E)")
    ax.scatter(plon, plat, s=6, c="#c3c2b7")
    k = w > 0
    ax.scatter(plon[k], plat[k], s=20 + 600 * w[k], c=w[k], cmap=SEQ, ec="k", lw=0.3)
    ax.plot(m["lon"][i], m["lat"][i], marker="*", ms=12, c=C_RAW)
    fig.suptitle("2단계 예측인자: ERA5 1.5° 247점 지상 4변수 — 7일 평균 편차장 (2026-10-22~28) + 대상점별 지역평균 가중치", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(); fig.savefig(FIG / "fig_s2_predictors.png"); plt.close(fig)


# ---------------------------------------------------------------- S3 EOF 주성분
def fig_s3():
    m = load("grid", "T", "week"); mm = m["models"][11]["eof"]
    plat, plon = m["plat"], m["plon"]
    fig, axs = plt.subplots(3, 3, figsize=(11, 9.5))
    for r, v in enumerate(["t2m", "msl", "tp"]):
        for c in range(3):
            ax = axs[r, c]
            e = mm.eof[v][c]
            if e[np.argmax(np.abs(e))] < 0:
                e = -e
            wideax(ax, f"{PVN[v]} EOF{c+1}  설명분산 {100*mm.evr[v][c]:.0f}%")
            col = coarse(ax, e / np.abs(e).max(), plat, plon, DIV_T, -1, 1)
    fig.colorbar(col, ax=axs, shrink=0.5, pad=0.02, label="정규화 적재값")
    fig.suptitle("3단계 EOF 주성분 (11월 주별 모형, ±1개월 창 1991~2025): 변수별 상위 6개 PC → 공통 예측인자", x=0.01, ha="left", fontsize=11, color=INK)
    fig.savefig(FIG / "fig_s3_eof.png", bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------- S4 전이함수 (능형회귀)
def fig_s4():
    mT, mP = load("grid", "T", "week"), load("grid", "P", "week")
    fig, axs = plt.subplots(1, 4, figsize=(14, 4.2))
    lev = mcolors.ListedColormap(["#b7d3f6", "#6da7ec", "#2a78d6", "#0d366b"])
    for ax, m, ttl in zip(axs[:2], [mT, mP], ["(a) 기온: 선택된 α (11월)", "(b) 강수: 선택된 α (11월)"]):
        mapax(ax, ttl)
        a = np.log10(m["models"][11]["alpha"])
        c = field(ax, a, m["lat"], m["lon"], lev, -0.5, 3.5)
        cb = plt.colorbar(c, ax=ax, shrink=0.8, pad=0.02, ticks=[0, 1, 2, 3]); cb.ax.set_yticklabels(["1", "10", "100", "1000"]); cb.ax.tick_params(labelsize=7)
    # (c) 11월 모형 계수 크기: 설계행렬 열 그룹별 평균 |표준화 계수|
    ax = axs[2]
    import train
    valsT = []; valsP = []
    for m, store, var, local in [(mT, valsT, "T", ["t2m", "d2m"]), (mP, valsP, "P", ["tp", "d2m", "msl"])]:
        mod = m["models"][11]; coef = mod["coef"]; n = mod["eof"].n
        D = train.prepare("grid", var, "week"); times = D["ya"].time.to_index()
        win = esd.season_window(times, 11, 1)
        Xd = {v: D["X"][v].values[win] for v in esd.PVARS}
        Lsd = np.nanstd(train.local_feats(Xd, D["W"], var), 0)          # (m, npt)
        groups = []
        for v in esd.PVARS:
            groups.append((f"EOF {v}", np.nanmean(np.abs(coef[:, len(groups) * n:(len(groups) + 1) * n]))))  # PC는 표준화(sd=1)
        off = 4 * n
        for j, v in enumerate(local):
            groups.append((f"지역평균 {v}", np.nanmean(np.abs(coef[:, off + j]) * Lsd[j])))
        store.append(dict(groups))
    keys = ["EOF t2m", "EOF d2m", "EOF msl", "EOF tp", "지역평균 t2m", "지역평균 d2m", "지역평균 tp", "지역평균 msl"]
    dT, dP = valsT[0], valsP[0]
    y = np.arange(len(keys))
    ax.barh(y - 0.18, [dT.get(k, 0) / max(dT.values()) for k in keys], 0.36, color=C_ESD, label="기온 모형")
    ax.barh(y + 0.18, [dP.get(k, 0) / max(dP.values()) for k in keys], 0.36, color=C_RAW, label="강수 모형")
    ax.set_yticks(y); ax.set_yticklabels(keys, fontsize=8); ax.invert_yaxis()
    ax.set_xlabel("상대 기여 (|β|×인자 표준편차의 그룹 평균, 최대=1)", fontsize=8); ax.legend(fontsize=8, frameon=False)
    ax.set_title("(c) 11월 모형 설계행렬 열별 기여", loc="left", color=INK)
    # (d) 잔차 표준편차 σ (확률 산출에 사용)
    ax = axs[3]; mapax(ax, "(d) 기온 CV 잔차 σ (°C, 11월)")
    c = field(ax, mT["models"][11]["resid_sd"], mT["lat"], mT["lon"], SEQ, 0, 1.5)
    plt.colorbar(c, ax=ax, shrink=0.8, pad=0.02).ax.tick_params(labelsize=7)
    fig.suptitle("4단계 전이함수: 점별·달별 능형회귀 [EOF 24 PC | 지역평균 2~3 | 절편], α는 연도 5묶음 CV로 선택", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(); fig.savefig(FIG / "fig_s4_transfer.png"); plt.close(fig)


# ---------------------------------------------------------------- S5 교차검증 시계열 예시
def fig_s5():
    ds = xr.open_dataset(WORK / "cv_asos_T_week.nc"); dp = xr.open_dataset(WORK / "cv_asos_P_week.nc")
    asos = xr.open_dataset(WORK / "target_asos.nc")
    names = asos["name"].values if "name" in asos else None
    i = int(np.argmin((asos.lat.values - 37.57) ** 2 + (asos.lon.values - 126.97) ** 2))
    fig, axs = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    sl = slice("2023-01-01", "2026-10-28")
    for ax, d, ttl, unit in zip(axs, [ds, dp], ["서울 주평균 기온 편차", "서울 주평균 강수 (sqrt 공간 편차)"], ["°C", "sqrt(mm/일)"]):
        o = d["obs"].isel(pt=i).sel(time=sl); p = d["cv"].isel(pt=i).sel(time=sl); b = d["base"].isel(pt=i).sel(time=sl)
        ax.axhline(0, color=GRID, lw=0.8)
        ax.plot(o.time, o, color=INK, lw=1.2, label="관측(ERA5/ASOS)")
        ax.plot(b.time, b, color=C_RAW, lw=0.9, alpha=0.8, label="원시 1.5° 지역평균")
        ax.plot(p.time, p, color=C_ESD, lw=1.0, label="다운스케일(leave-one-year-out)")
        r, s = plots.skill(o.values, p.values); r0, s0 = plots.skill(o.values, b.values)
        ax.set_title(f"{ttl}  — 독립 2023~2026: r {r:.2f} (원시 {r0:.2f}), MSESS {s:.2f} (원시 {s0:.2f})", loc="left", color=INK)
        ax.set_ylabel(unit); ax.legend(fontsize=8, frameon=False, ncol=3, loc="upper left")
    fig.suptitle("5단계 검증: 연도별 교차검증(EOF도 폴드마다 재적합) — 학습에 쓰지 않은 2023~2026 독립 기간 예시", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(); fig.savefig(FIG / "fig_s5_cv_series.png"); plt.close(fig)


# ---------------------------------------------------------------- S6 예보모형 적용: 리드별 요약 + 모형기후 보정
def fig_s6():
    d = json.load(open(FC))
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    for ax, prod, ttl in zip(axs[:2], ["week", "month"], ["EC46 주별 (1~6주)", "SEAS5 월별 (1~6개월)"]):
        P = d["products"][prod]["periods"]
        lab = [p["label"] for p in P]
        ta = np.array([np.mean(p["grid_T"]["anom"]) for p in P]); sp = np.array([np.mean(p["grid_T"]["spread"]) for p in P])
        pa = np.array([np.mean(p["grid_T"]["above"]) for p in P])
        x = np.arange(len(P))
        ax.axhline(0, color=GRID, lw=0.8)
        ax.fill_between(x, ta - sp, ta + sp, color=C_ESD, alpha=0.15, label="멤버 퍼짐 ±1σ")
        ax.plot(x, ta, "o-", color=C_ESD, lw=1.5, ms=5, label="격자평균 기온 편차")
        for xi, yi, pi in zip(x, ta, pa):
            ax.annotate(f"{yi:+.1f}°C\n높음 {pi:.0f}%", (xi, yi), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=7.5, color=INK2)
        ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=8, rotation=20); ax.set_ylabel("°C"); ax.set_ylim(-1.5, 4)
        ax.set_title(f"{ttl} 한반도 평균 기온 편차", loc="left", color=INK); ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax = axs[2]
    rows = []
    for prod in ["week", "month"]:
        for p in d["products"][prod]["periods"]:
            b = p["bias"]
            rows.append([p["label"]] + [b[k].split()[-1] for k in ["t2m", "d2m", "msl", "tp"]])
    ax.axis("off")
    tb = ax.table(cellText=rows, colLabels=["기간", "t2m", "d2m", "msl(hPa)", "tp"], loc="center", cellLoc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(7.5); tb.scale(1, 1.25)
    for (r, c), cell in tb.get_celld().items():
        cell.set_edgecolor(GRID)
        if r == 0:
            cell.set_text_props(color=INK, weight="bold")
    ax.set_title("모형기후 보정량 (Open-Meteo anomaly로 역산)", loc="left", color=INK)
    fig.suptitle(f"6단계 예보모형 적용: 발표 {d['products']['week']['issued']} — 보정된 EC46·SEAS5 대규모장을 학습 EOF·계수에 투영", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(); fig.savefig(FIG / "fig_s6_forecast_apply.png"); plt.close(fig)


if __name__ == "__main__":
    for f in [fig_s1, fig_s2, fig_s3, fig_s4, fig_s5, fig_s6]:
        f(); print(f.__name__, "ok", flush=True)
