"""SEAS5(CDS) 3개월 상세전망 산출 — 격자 412 + ASOS 66, 리드별 편차·3분위 확률.

  KDS_WORK=... CDS_DIR=... python seas5_monthly/forecast.py [--init 10] [--year 2026] [--leads 2,3,4] [--out out/seas5_monthly] [--maps]

SEAS5 실황(51멤버) 편차 = 재예보(1993~2016) 기후 기준 + ERA5 기후 차이 보정 → ERA5 1° 격자 보간 → 학습 EOF·ARD 계수 투영
→ 멤버별 편차(51개) + 교차검증 잔차 σ 정규혼합 → 3분위 확률과 10·25·50·75·90 백분위 예측범위(p10~p90).
JSON 구조는 ec46_weekly/forecast.py 와 동일(products.month) + p10~p90, south_mean 에 T_p10/T_p90·P_p10/P_p90.
"""
import argparse, datetime as dt, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "pyesd"))
import predictors as pr, apply as ap, products  # noqa: E402
from train import CDS  # noqa: E402

KOR = ["1월", "2월", "3월", "4월", "5월", "6월", "7월", "8월", "9월", "10월", "11월", "12월"]


def downscale(kind, var, A, lead, month, ndays):
    M = ap.load_model(kind, var); A_ = ap.Applier(M)
    mu = A_.predict(A, lead, month)[:, 0, :]                      # (member, npt) 멤버별 투영 편차
    q = {k: M[f"m{month}_{k}"] for k in ("q10", "q33", "q50", "q67", "q90")}
    r = products.fields(mu, M[f"m{month}_sigma"], q, M["yclim"][month - 1], var, "month", ndays,
                        normal=M["normal"][month - 1] if var == "P" else None)
    r["r_cv"] = M[f"m{month}_r_cv"]
    return r, M


def to_json(out, P, kinds=("grid", "asos")):
    meta = {}
    if "grid" in kinds:
        g = ap.load_model("grid", "T"); meta["grid"] = dict(lat=g["lat"].round(2).tolist(), lon=g["lon"].round(2).tolist())
    a = ap.load_model("asos", "T")
    meta["asos"] = dict(stn=a["stn"].tolist(), name=a["name"].tolist(), lat=a["lat"].round(4).tolist(),
                        lon=a["lon"].round(4).tolist(), jeju=[int(v) for v in a["jeju"]])
    south = np.asarray(meta["asos"]["jeju"]) == 0
    per = []
    for it in P["periods"]:
        d = {k: it[k] for k in ("period", "label", "start", "end", "lead", "kind", "ndays")}
        for k in [f"{kd}_{v}" for kd in kinds for v in ("T", "P")]:
            d[k] = products.to_jsonable(it[k], k[-1])
        T, Pp = it["asos_T"], it["asos_P"]
        nm = lambda x: float(np.nanmean(np.asarray(x)[south]))  # noqa: E731
        d["south_mean"] = dict(n=int(south.sum()), T_anom=round(nm(T["anom"]), 2), T_above=int(round(100 * nm(T["above"]))),
                               T_below=int(round(100 * nm(T["below"]))), P_pct=int(round(nm(Pp["pct"]))),
                               P_above=int(round(100 * nm(Pp["above"]))), P_below=int(round(100 * nm(Pp["below"]))),
                               T_p10=round(nm(T["p10"]), 2), T_p90=round(nm(T["p90"]), 2),
                               P_p10=int(round(nm(Pp["p10"]))), P_p90=int(round(nm(Pp["p90"]))))
        per.append(d)
    js = dict(meta=meta, generated=dt.datetime.now(dt.timezone.utc).isoformat(timespec="minutes"),
              products={"month": dict(issued=P["issued"], model=P["model"], periods=per)})
    (out / "downscaling.json").write_text(json.dumps(js, ensure_ascii=False, separators=(",", ":")))
    rows = ["period,lead,stn,name,jeju,T_anom,T_value,T_below,T_near,T_above,T_p10,T_p25,T_p75,T_p90,"
            "P_value,P_pct,P_below,P_near,P_above,P_p10,P_p25,P_p75,P_p90"]
    for it in P["periods"]:
        T, Pp = it["asos_T"], it["asos_P"]
        for j, s in enumerate(meta["asos"]["stn"]):
            rows.append(",".join(map(str, [it["period"], it["lead"], s, meta["asos"]["name"][j], meta["asos"]["jeju"][j],
                f"{T['anom'][j]:.2f}", f"{T['value'][j]:.2f}", f"{T['below'][j]:.2f}", f"{T['near'][j]:.2f}", f"{T['above'][j]:.2f}",
                *[f"{T[q][j]:.2f}" for q in ("p10", "p25", "p75", "p90")],
                f"{Pp['value'][j]:.2f}", f"{Pp['pct'][j]:.0f}", f"{Pp['below'][j]:.2f}", f"{Pp['near'][j]:.2f}", f"{Pp['above'][j]:.2f}",
                *[f"{Pp[q][j]:.0f}" for q in ("p10", "p25", "p75", "p90")]])))
    (out / "asos66_downscaled.csv").write_text("\n".join(rows) + "\n", encoding="utf-8-sig")


def draw(out, P):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(HERE.parent / "ec46_weekly")); import maps  # noqa: E402
    g = ap.load_model("grid", "T"); lat, lon = g["lat"], g["lon"]
    per = P["periods"]; nb = len(per)
    fig, axs = plt.subplots(4, nb, figsize=(1.9 * nb + 1.3, 9.6), squeeze=False)
    for b, it in enumerate(per):
        T, Pp = it["grid_T"], it["grid_P"]
        i1 = maps.field(axs[0, b], T["anom"], lat, lon, maps.DIV_T, np.arange(-3, 3.01, 0.5)); maps.base(axs[0, b], it["label"])
        dom = np.where(T["above"] >= T["below"], T["above"], -T["below"]) * 100
        i2 = maps.field(axs[1, b], dom, lat, lon, maps.DIV_T, [-100, -70, -60, -50, -40, 40, 50, 60, 70, 100]); maps.base(axs[1, b])
        i3 = maps.field(axs[2, b], Pp["pct"], lat, lon, maps.DIV_P, [0, 40, 60, 80, 90, 110, 120, 150, 200, 300]); maps.base(axs[2, b])
        domp = np.where(Pp["above"] >= Pp["below"], Pp["above"], -Pp["below"]) * 100
        i4 = maps.field(axs[3, b], domp, lat, lon, maps.DIV_P, [-100, -70, -60, -50, -40, 40, 50, 60, 70, 100]); maps.base(axs[3, b])
    for ax, im, lab in ((axs[0], i1, "기온 편차 (°C)"), (axs[1], i2, "기온 우세범주 확률 (%)\n-낮음 / +높음"),
                        (axs[2], i3, "강수 평년비 (%)"), (axs[3], i4, "강수 우세범주 확률 (%)\n-적음 / +많음")):
        fig.colorbar(im, ax=ax, shrink=0.85, label=lab)
    fig.suptitle(f"한반도 0.25° 상세예측 — 월별(SEAS5 CDS, {P['model']}), 발표 {P['issued']}", x=0.01, ha="left", fontsize=11, color=maps.INK)
    fig.savefig(Path(out) / "map_month.png", bbox_inches="tight"); plt.close(fig)


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--init", type=int, default=10); a_.add_argument("--year", type=int, default=dt.date.today().year)
    a_.add_argument("--leads", default="2,3,4"); a_.add_argument("--out", default=str(HERE.parent / "out" / "seas5_monthly"))
    a_.add_argument("--maps", action="store_true"); a_.add_argument("--kinds", default="grid,asos")
    a = a_.parse_args()
    t0 = time.time()
    clim = ap.MODELS / f"clim_init{a.init:02d}.nc"
    if clim.exists():                                   # hindcast_verify.py 가 저장한 기후값 → ERA5·재예보 원자료 불필요
        c9120, c9316, hc = pr.load_clims(clim)
    else:
        F = pr.load_era5(CDS); c9120 = pr.era5_clim(F, pr.CLIM); c9316 = pr.era5_clim(F, pr.HIND)
        hc = pr.hind_clim(pr.load_seas5(CDS, "hind", a.init))
    Ff = pr.load_seas5(CDS, f"fc{a.year}", a.init)
    A = ap.seas5_anoms_on_grid(hc, Ff, c9120, c9316)
    init = pd.Timestamp(Ff["t2m"].init.values[0])
    kinds = tuple(a.kinds.split(",")); pers = []
    for ld in [int(x) for x in a.leads.split(",")]:
        vt = init + pd.DateOffset(months=ld - 1); m = vt.month
        nd = int((vt + pd.offsets.MonthEnd(0)).day)
        it = dict(period=vt.strftime("%Y-%m"), label=f"{vt.year}년 {KOR[m-1]}", start=vt.strftime("%Y-%m-%d"),
                  end=(vt + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d"), lead=ld, kind="month", ndays=nd)
        for kind in kinds:
            for var in ("T", "P"):
                it[f"{kind}_{var}"], _ = downscale(kind, var, A, ld, m, nd)
        pers.append(it)
        k0 = kinds[0]
        print(f"  lead {ld} {it['label']} {k0} 기온 편차 {np.nanmean(it[f'{k0}_T']['anom']):+.2f}°C "
              f"[p10 {np.nanmean(it[f'{k0}_T']['p10']):+.2f}, p90 {np.nanmean(it[f'{k0}_T']['p90']):+.2f}], 강수 {np.nanmean(it[f'{k0}_P']['pct']):.0f}% {time.time()-t0:.0f}s", flush=True)
    P = dict(issued=init.strftime("%Y-%m"), model=f"SEAS5 v51 {Ff['t2m'].sizes['member']}멤버, {init.strftime('%Y-%m')} 초기화", periods=pers)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    to_json(out, P, kinds)
    if a.maps and "grid" in kinds:
        draw(out, P)
    print("done", out, f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
