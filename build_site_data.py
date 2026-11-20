"""상세예측 산출(downscaling.json) → 사이트 자료 (objective_climate_outlook 과 같은 형식).

  python site/build_site_data.py --ec46 out/ec46_weekly/downscaling.json --seas5 out/seas5_monthly/downscaling.json --out site/data

생성: data/grid.json (0.25° 412격자: step·lats·lons·points·fill_index), data/latest/manifest.json,
      map_1m.json·map_3m.json (격자 field), stations.json (ASOS 66 field + 남한 62지점 평균 korea 블록),
      points/<i>.json·stations/<stn>.json (지점 상세: 기간별 통계·분포 경계·임계값 확률)
요소: t2m(°C, 기간 평균), precip(mm, 기간 누적). 확률은 정수 %.
"""
import argparse, datetime as dt, json
from pathlib import Path
import numpy as np

VARS = {"t2m": {"name": "평균기온", "unit": "°C", "agg": "mean", "mode": "add", "thresholds": {"anom": [-2, -1, 1, 2]}},
        "precip": {"name": "강수량", "unit": "mm", "agg": "sum", "mode": "mul",
                   "thresholds": {"ratio": [50, 150, 200], "abs": {"week": [10, 30, 50, 100], "month": [50, 100, 200, 300]}}}}
PRODUCT_META = {"1m": ("1개월 전망", "ECMWF EC46 (51 멤버) · PyESD 상세화"), "3m": ("3개월 전망", "ECMWF SEAS5 (51 멤버) · PyESD 상세화")}
NAN = None


def grid_json(lat, lon, step=0.25):
    lats = np.round(np.arange(lat.min(), lat.max() + 1e-6, step), 3); lons = np.round(np.arange(lon.min(), lon.max() + 1e-6, step), 3)
    pts = [[float(a), float(b)] for a, b in zip(lat, lon)]
    arr = np.array(pts); fill = []
    for la in lats:
        for lo in lons:
            d = (arr[:, 0] - la) ** 2 + ((arr[:, 1] - lo) * np.cos(np.radians(la))) ** 2
            fill.append(int(d.argmin()))
    return {"step": step, "lats": lats.tolist(), "lons": lons.tolist(), "points": pts, "fill_index": fill}


def arr(x):
    return np.array([np.nan if v is None else v for v in x], float)


def field(r, var, ndays, kind):
    """downscaling.json 의 한 field dict(격자 또는 지점) -> 사이트 field. precip 은 mm/일 → 기간 누적 mm"""
    f = lambda k: arr(r[k])  # noqa: E731
    r2 = lambda a, nd=2: [None if not np.isfinite(v) else round(float(v), nd) for v in a]  # noqa: E731
    pc = lambda a: [None if not np.isfinite(v) else int(round(float(v))) for v in a]  # noqa: E731
    if var == "T":
        mean, clim = f("value"), f("clim") if "clim" in r else f("value") - f("anom")
        out = dict(mean=r2(mean), clim=r2(clim), anom=r2(mean - clim), spread=r2(f("spread")),
                   p10=r2(clim + f("p10")), p25=r2(clim + f("p25")), p50=r2(clim + f("p50")), p75=r2(clim + f("p75")), p90=r2(clim + f("p90")))
        if "b10" in r:
            out["bounds"] = [r2(f(k)) for k in ("b10", "b33", "b50", "b67", "b90")]
    else:
        nm = f("normal"); mean = f("value") * ndays; clim = nm * ndays
        out = dict(mean=r2(mean, 1), clim=r2(clim, 1), anom=r2(mean - clim, 1), anom_pct=r2(f("pct"), 0), spread=r2(f("spread") * ndays, 1))
        for q in ("p10", "p25", "p50", "p75", "p90"):
            out[q] = r2(f(q) / 100 * clim, 1)
        if "b10" in r:
            out["bounds"] = [r2(f(k) * ndays, 1) for k in ("b10", "b33", "b50", "b67", "b90")]
    for k in ("below", "near", "above", "gt_median", "lt_p10", "gt_p90"):
        if k in r:
            out[k] = pc(f(k))
    out["thr"] = {k: pc(arr(v)) for k, v in r.get("thr", {}).items()}
    out["r_cv"] = r2(f("r_cv")) if "r_cv" in r else None
    return out


def period_entry(it, i, key):
    kind = it.get("kind", "week" if key == "1m" else "month")
    return dict(id=f"{'w' if kind == 'week' else 'm'}{i+1}", kind=kind, label=it["label"] if kind == "month" else f"{i+1}주",
                start=it["start"], end=it["end"], ndays=it.get("ndays") or ((dt.date.fromisoformat(it["end"]) - dt.date.fromisoformat(it["start"])).days + 1))


def build(ec46, seas5, out):
    out = Path(out); latest = out / "latest"; (latest / "points").mkdir(parents=True, exist_ok=True); (latest / "stations").mkdir(exist_ok=True)
    srcs = {}
    if ec46 and Path(ec46).exists():
        srcs["1m"] = json.load(open(ec46))
    if seas5 and Path(seas5).exists():
        srcs["3m"] = json.load(open(seas5))
    if not srcs:
        raise SystemExit("입력 없음")
    meta = next(iter(srcs.values()))["meta"]
    glat, glon = np.array(meta["grid"]["lat"]), np.array(meta["grid"]["lon"])
    json.dump(grid_json(glat, glon), open(out / "grid.json", "w"), separators=(",", ":"))
    stns = meta["asos"]; south = np.array(stns["jeju"]) == 0
    alt = {}
    csv = Path(__file__).resolve().parents[1] / "training" / "asos66_stations.csv"
    if csv.exists():                                                   # 지점 고도 (표시용)
        for ln in csv.read_text(encoding="utf-8-sig").splitlines()[1:]:
            c = ln.split(","); alt[int(c[0])] = float(c[4])
    products, maps, stfields, korea = {}, {}, {}, {}
    pts_rec = [dict(lat=float(a), lon=float(b)) for a, b in zip(glat, glon)]
    stn_rec = [dict(lat=stns["lat"][j], lon=stns["lon"][j], stn=stns["stn"][j], name=stns["name"][j], ht=alt.get(stns["stn"][j]), kind="station", missing=[])
               for j in range(len(stns["stn"]))]
    issued = None
    for key, js in srcs.items():
        pk = "week" if key == "1m" else "month"
        P = js["products"][pk]
        pers = [period_entry(it, i, key) for i, it in enumerate(P["periods"])]
        name, model = PRODUCT_META[key]
        products[key] = dict(name=name, model=P["model"] if key == "3m" else model, periods=pers, start=pers[0]["start"], end=pers[-1]["end"],
                             members={"t2m": 51, "precip": 51}, issued=P["issued"], skill=None)
        issued = issued or js["generated"]
        fields = {"t2m": {}, "precip": {}}; sf = {"t2m": {}, "precip": {}}; kor = {}
        for p, it in zip(pers, P["periods"]):
            for vk, var in (("t2m", "T"), ("precip", "P")):
                fields[vk][p["id"]] = field(it[f"grid_{var}"], var, p["ndays"], p["kind"])
                sf[vk][p["id"]] = field(it[f"asos_{var}"], var, p["ndays"], p["kind"])
            # 남한 평균 (제주 제외 62지점)
            kor[p["id"]] = {}
            for vk in ("t2m", "precip"):
                s = sf[vk][p["id"]]; m = lambda k: float(np.nanmean(arr(s[k])[south]))  # noqa: E731
                b = s.get("bounds")
                kor[p["id"]][vk] = dict(normal=round(m("clim"), 1), lower=round(float(np.nanmean(arr(b[1])[south])), 1) if b else None,
                                        upper=round(float(np.nanmean(arr(b[3])[south])), 1) if b else None, mean=round(m("mean"), 2),
                                        below=int(round(m("below"))), near=int(round(m("near"))), above=int(round(m("above"))), n=int(south.sum()), src="obs")
        maps[key] = dict(key=key, periods=pers, fields=fields)
        stfields[key] = dict(fields=sf, korea=kor)
        # 지점 상세 레코드
        for recs, F in ((pts_rec, fields), (stn_rec, sf)):
            for j, rec in enumerate(recs):
                rec[key] = {"periods": {vk: {p["id"]: row(F[vk][p["id"]], j) for p in pers} for vk in ("t2m", "precip")}}
    for key, mp in maps.items():
        json.dump(mp, open(latest / f"map_{key}.json", "w"), separators=(",", ":"))
    json.dump(dict(source="PyESD 상세화 · ASOS 66지점 관측 평년(1991–2020)", stations=[dict(stn=s["stn"], name=s["name"], lat=s["lat"], lon=s["lon"], ht=s["ht"], missing=[]) for s in stn_rec],
                   products=stfields), open(latest / "stations.json", "w"), separators=(",", ":"), ensure_ascii=False)
    for j, rec in enumerate(pts_rec):
        json.dump(rec, open(latest / "points" / f"{j}.json", "w"), separators=(",", ":"), ensure_ascii=False)
    for rec in stn_rec:
        json.dump(rec, open(latest / "stations" / f"{rec['stn']}.json", "w"), separators=(",", ":"), ensure_ascii=False)
    man = dict(issued=issued, demo=False, climatology="ERA5 0.25° · ASOS 1991–2020 관측 평년", variables=VARS, products=products,
               stations=len(stn_rec), grid=dict(step=0.25, cells=len(pts_rec)), method="PyESD Perfect-Prognosis 상세화 (downscale_by_pyesd)")
    json.dump(man, open(latest / "manifest.json", "w"), separators=(",", ":"), ensure_ascii=False)
    print("site data:", list(maps), f"grid {len(pts_rec)} stations {len(stn_rec)}")


def row(F, j):
    """field dict 의 j 번째 점 -> 지점 상세 행"""
    g = lambda k: (F[k][j] if F.get(k) is not None else None)  # noqa: E731
    r = {k: g(k) for k in ("mean", "clim", "anom", "anom_pct", "spread", "below", "near", "above", "gt_median", "lt_p10", "gt_p90") if k in F}
    r["q"] = [g(k) for k in ("p10", "p25", "p50", "p75", "p90")]
    r["bounds"] = [b[j] for b in F["bounds"]] if F.get("bounds") else [None] * 5
    r["thr"] = {k: v[j] for k, v in F["thr"].items()}
    r["r_cv"] = g("r_cv")
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ec46"); ap.add_argument("--seas5"); ap.add_argument("--out", default="site/data")
    a = ap.parse_args()
    build(a.ec46, a.seas5, a.out)
