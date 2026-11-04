"""SEAS5 재예보(1993~2016)로 리드별 검증 — 상세화 vs 원 SEAS5 지역평균.

  KDS_WORK=... CDS_DIR=... python seas5_monthly/hindcast_verify.py [--init 10] [--kind grid,asos] [--out docs/seas5]

점수: 앙상블평균 ACC(시간상관, 점별), RPSS(3분위, 기후확률 1/3 기준), 10~90 % 예측범위 포함률, 신뢰도표(above/below), 리드 1~4.
      멤버(25개)마다 따로 투영한 뒤 확률·백분위를 계산한다.
기준선 '원 SEAS5': 같은 반경 가중 지역평균 t2m(또는 sqrt 강수) 편차, 멤버 분포 → 3분위.
산출: OUT/hindcast_{kind}_{var}_init{mm}.nc, OUT/hindcast_summary.csv, OUT/fig_hindcast_init{mm}.png
"""
import argparse, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd, xarray as xr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "pyesd"))
import esd, predictors as pr, apply as ap  # noqa: E402
from train import load_target, WORK, CDS  # noqa: E402


def acc(pred, obs):
    """시간상관 (n, npt) -> (npt,)"""
    ok = np.isfinite(obs) & np.isfinite(pred)
    out = np.full(obs.shape[1], np.nan)
    for j in range(obs.shape[1]):
        k = ok[:, j]
        if k.sum() > 5:
            out[j] = np.corrcoef(pred[k, j], obs[k, j])[0, 1]
    return out


def rpss(pb, pn, pa, obs, q33, q67):
    """RPS 대비 기후(1/3) — (n, npt) 확률, obs 편차 -> (npt,)"""
    cat = np.where(obs < q33, 0, np.where(obs > q67, 2, 1))
    P = np.stack([pb, pn, pa], -1).cumsum(-1)
    O = (np.arange(3)[None, None] >= cat[..., None]).astype(float)
    rps = ((P - O) ** 2).sum(-1)
    C = np.cumsum(np.ones(3) / 3)
    rpsc = ((C[None, None] - O) ** 2).sum(-1)
    ok = np.isfinite(obs)
    return 1 - np.nansum(np.where(ok, rps, np.nan), 0) / np.nansum(np.where(ok, rpsc, np.nan), 0)


def reliability(p, hit, bins=(0, .1, .2, .3, .4, .5, .6, .7, .8, 1.0001)):
    """예보확률 p 와 실제 발생(0/1) -> 구간별 (평균 예보확률, 관측빈도, 표본수)"""
    p, hit = np.asarray(p).ravel(), np.asarray(hit).ravel(); ok = np.isfinite(p) & np.isfinite(hit)
    rows = []
    for a, b in zip(bins[:-1], bins[1:]):
        k = ok & (p >= a) & (p < b)
        rows.append((float(p[k].mean()) if k.any() else np.nan, float(hit[k].mean()) if k.any() else np.nan, int(k.sum())))
    return np.array(rows)


def verify(kind, var, mm, Fh, A, c9120):
    M = ap.load_model(kind, var); A_ = ap.Applier(M)
    ya, lat, lon, meta, yclim, mon = load_target(kind, var)
    inits = pd.DatetimeIndex(Fh["t2m"].init.values)
    res = {}
    for ld in Fh["t2m"].lead.values:
        m = ap.lead_month(inits[0], ld)
        vt = pd.DatetimeIndex([i + pd.DateOffset(months=int(ld) - 1) for i in inits])
        obs = ya.reindex(time=vt).values                                         # (ninit, npt)
        mu = A_.predict(A, ld, m)                                                # (member, ninit, npt)
        pb, pn, pa = A_.probs(mu, m)
        em = mu.mean(0)
        # 기준선: 원 SEAS5 지역평균 편차 (t2m 또는 sqrt 강수), 멤버 분포 3분위 (재예보 자체 분위)
        v = "t2m" if var == "T" else "tp"
        raw = A[v].sel(lead=ld).transpose("member", "init", "lat", "lon").values.reshape(mu.shape[0], mu.shape[1], -1)
        raw = np.nan_to_num(raw) @ M["W"].T                                      # (member, ninit, npt)
        rq33, rq67 = np.nanpercentile(raw, 100 / 3, (0, 1)), np.nanpercentile(raw, 200 / 3, (0, 1))
        rpb = (raw < rq33).mean(0); rpa = (raw > rq67).mean(0); rpn = 1 - rpb - rpa
        q33, q67 = M[f"m{m}_q33"], M[f"m{m}_q67"]
        Q = np.stack([A_.quantiles(mu[:, i], m, q=(10, 90)) for i in range(mu.shape[1])], 1)   # (2, ninit, npt)
        cover = np.nanmean((obs >= Q[0]) & (obs <= Q[1]), 0)                                  # 10~90 범위 포함률 (목표 0.8)
        rel = np.stack([reliability(pa, obs > q67), reliability(pb, obs < q33)])             # (2, nbin, 3) above/below
        res[int(ld)] = dict(month=m, acc_ds=acc(em, obs), acc_raw=acc(raw.mean(0), obs),
                            rpss_ds=rpss(pb, pn, pa, obs, q33, q67), rpss_raw=rpss(rpb, rpn, rpa, obs, q33, q67),
                            cover1090=cover, reliab=rel, ensmean=em, obs=obs, p_above=pa, p_below=pb, valid=vt)
        print(f"  {kind} {var} init{mm:02d} lead{ld} ({m}월) ACC ds {np.nanmedian(res[int(ld)]['acc_ds']):.2f} raw {np.nanmedian(res[int(ld)]['acc_raw']):.2f}"
              f" RPSS ds {np.nanmedian(res[int(ld)]['rpss_ds']):.2f} raw {np.nanmedian(res[int(ld)]['rpss_raw']):.2f}", flush=True)
    leads = sorted(res)
    ds = xr.Dataset(coords={"lead": leads, "init": inits, "pt": np.arange(len(lat)), "lat": ("pt", lat), "lon": ("pt", lon)})
    for k in ("acc_ds", "acc_raw", "rpss_ds", "rpss_raw", "cover1090"):
        ds[k] = (("lead", "pt"), np.stack([res[l][k] for l in leads]))
    ds["reliab"] = (("lead", "cat", "bin", "stat"), np.stack([res[l]["reliab"] for l in leads]))
    ds = ds.assign_coords(cat=["above", "below"], stat=["p_fcst", "obs_freq", "n"])
    for k in ("ensmean", "obs", "p_above", "p_below"):
        ds[k] = (("lead", "init", "pt"), np.stack([res[l][k] for l in leads]))
    ds["month"] = ("lead", [res[l]["month"] for l in leads])
    if kind == "asos":
        for k, v in meta.items():
            ds[k] = ("pt", np.asarray(v))
    return ds


def figure(out, mm, tabs):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(HERE.parent / "ec46_weekly")); import maps  # noqa: E402
    fig, axs = plt.subplots(2, 2, figsize=(9, 6.2))
    for ax, (kind, var) in zip(axs.ravel(), [(k, v) for k in ("grid", "asos") for v in ("T", "P")]):
        ds = tabs.get((kind, var))
        if ds is None:
            ax.axis("off"); continue
        ld = ds.lead.values
        for k, lab, c in (("acc_ds", "상세화 ACC", "#c0392b"), ("acc_raw", "원 SEAS5 ACC", "#7f8c8d"),
                          ("rpss_ds", "상세화 RPSS", "#2471a3"), ("rpss_raw", "원 SEAS5 RPSS", "#aab7b8")):
            med = ds[k].median("pt").values; q1 = ds[k].quantile(0.25, "pt").values; q3 = ds[k].quantile(0.75, "pt").values
            ax.plot(ld, med, "-o", color=c, label=lab, ms=4); ax.fill_between(ld, q1, q3, color=c, alpha=0.12)
        ax.axhline(0, color="#999", lw=0.6)
        ax.set_xticks(ld); ax.set_xticklabels([f"리드 {l}\n({int(m)}월)" for l, m in zip(ld, ds.month.values)])
        ax.set_title(f"{'격자 412' if kind == 'grid' else 'ASOS 66'} — {'기온' if var == 'T' else '강수'}", loc="left", fontsize=9)
        ax.set_ylim(-0.4, 1); ax.grid(alpha=0.3)
    for ax in axs.ravel():
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7, ncol=2); break
    fig.suptitle(f"SEAS5 {mm}월 초기화 재예보(1993~2016) 리드별 검증 — 점별 중앙값과 사분위 범위", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(); fig.savefig(out / f"fig_hindcast_init{mm:02d}.png", dpi=140); plt.close(fig)


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--init", type=int, default=10)
    ap_.add_argument("--kind", default="grid,asos")
    ap_.add_argument("--out", default=str(HERE.parent / "docs" / "seas5"))
    a = ap_.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    F = pr.load_era5(CDS); c9120 = pr.era5_clim(F, pr.CLIM); c9316 = pr.era5_clim(F, pr.HIND)
    Fh = pr.load_seas5(CDS, "hind", a.init); hc = pr.hind_clim(Fh)
    ap.MODELS.mkdir(exist_ok=True)
    pr.save_clims(ap.MODELS / f"clim_init{a.init:02d}.nc", c9120, c9316, hc)     # 운영 전망용 (실황 파일만으로 가능)
    A = ap.seas5_anoms_on_grid(hc, Fh, c9120, c9316)
    print(f"loaded {time.time()-t0:.0f}s", flush=True)
    tabs, rows = {}, []
    for kind in a.kind.split(","):
        for var in ("T", "P"):
            ds = verify(kind, var, a.init, Fh, A, c9120)
            ds.to_netcdf(out / f"hindcast_{kind}_{var}_init{a.init:02d}.nc"); tabs[(kind, var)] = ds
            for l in ds.lead.values:
                r = dict(init=a.init, kind=kind, var=var, lead=int(l), month=int(ds.month.sel(lead=l)))
                for k in ("acc_ds", "acc_raw", "rpss_ds", "rpss_raw", "cover1090"):
                    r[k] = round(float(ds[k].sel(lead=l).median()), 3)
                rows.append(r)
    csv = out / "hindcast_summary.csv"
    df = pd.DataFrame(rows)
    if csv.exists():
        old = pd.read_csv(csv); df = pd.concat([old[~((old.init == a.init) & old.kind.isin(df.kind))], df])
    df.to_csv(csv, index=False)
    figure(out, a.init, tabs)
    print("done", f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
