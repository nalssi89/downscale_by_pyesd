"""2단계 보강망(AWS) 월자료 만들기 — 기상청 API허브 기후통계(sts_ta, sts_rn) 월자료.

보강망 = 기준망(training/base_stations.csv)에 없는 AWS 가운데
  · 마지막 이전(좌표 2 km 이상 또는 고도 30 m 이상 변화) 이후 구간만 사용하고
  · 그 구간에서 1991~현재 12개 달 각각 유효 연수 15년 이상, 결측률 10% 미만인 지점.
이전 이력은 연도별 호출(tm2 = 그 해 12월)로 시기별 좌표가 달리 나오는 것을 이용해 찾는다.
강수는 30 km 안 기준망 지점과 월강수 상관이 0.85 미만이면 쓰지 않는다(기온만 사용).

  KMA_AUTHKEY=... python training/build_aws_network.py [--select] OUTDIR
출력: training/aws_stations.csv, OUTDIR/target_aws.nc (time, stn)  T °C, P mm/일, 이전 전 자료는 NaN
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_base_network import fetch, station_info, T0  # noqa: E402

MIN_YEARS, MAX_MISS, MOVE_KM, MOVE_ALT = 15, 0.10, 2.0, 30.0
P_KM, P_MIN_R = 30.0, 0.85


def _dist(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.deg2rad, (la1, lo1, la2, lo2))
    return 6371 * np.arccos(np.clip(np.sin(la1) * np.sin(la2) + np.cos(la1) * np.cos(la2) * np.cos(lo1 - lo2), -1, 1))


def history(tm2):
    """1997~현재 연도별 호출(stn_id=0)로 지점·월별 당시 좌표를 모은다."""
    from kma_util import _get  # noqa: F401  (fetch 가 사용)
    out = []
    y1 = int(tm2[:4])
    for y in range(1997, y1 + 1):
        d = fetch_all("ta", f"{y}01", f"{y}12" if y < y1 else tm2)
        out.append(d[["TM", "STN", "LAT", "LON", "ALT"]])
    return pd.concat(out).sort_values("TM")


def fetch_all(var, tm1, tm2):
    from io import StringIO
    from kma_util import _get
    txt = _get(f"sts_{var}.php", dict(tm1=tm1, tm2=tm2, stn_id=0, disp=0, help=0))
    rows = [ln.rstrip("=").rstrip(",") for ln in txt.splitlines() if ln[:1].isdigit()]
    d = pd.read_csv(StringIO("\n".join(rows)), header=None).iloc[:, :6]
    d.columns = ["TM", "STN", "LAT", "LON", "ALT", "VAL"]
    d.loc[d.VAL <= -99, "VAL"] = np.nan
    return d


def select(tm2):
    base = pd.read_csv(HERE / "base_stations.csv")
    h = history(tm2)
    cur = h.groupby("STN")[["LAT", "LON", "ALT"]].last()
    cand = [s for s in cur.index if s not in set(base.STN) and s >= 90 and h[h.STN == s].TM.max() >= int(tm2) - 100]
    ta = fetch("ta", cand, tm2)                       # 전체 기간(tm2 최근)
    ta = ta[ta.TM <= int(tm2)]
    asos = set(pd.read_csv(HERE / "asos66_stations.csv").STN)
    names = station_info()
    rows = []
    for s in cand:
        g = h[h.STN == s]
        c = cur.loc[s]
        moved = (_dist(g.LAT.values, g.LON.values, c.LAT, c.LON) >= MOVE_KM) | (np.abs(g.ALT.values - c.ALT) >= MOVE_ALT)
        brk = int(g.TM.values[np.where(moved)[0][-1]]) if moved.any() else 0
        v = ta[(ta.STN == s) & (ta.TM > brk) & ta.VAL.notna() & (ta.TM >= int(T0))]
        if v.empty:
            continue
        my = v.groupby(v.TM % 100).apply(lambda x: (x.TM // 100).nunique()).reindex(range(1, 13)).fillna(0).min()
        first = str(v.TM.min())
        n = (pd.Period(f"{tm2[:4]}-{tm2[4:]}", "M") - pd.Period(f"{first[:4]}-{first[4:]}", "M")).n + 1
        miss = 1 - len(v) / n
        if my >= MIN_YEARS and miss < MAX_MISS and s not in asos:
            rows.append(dict(STN=s, name=names.get(s, ""), lat=c.LAT, lon=c.LON, alt=c.ALT, start=int(first),
                             break_ym=brk, min_years=int(my), miss=round(miss, 3),
                             jeju=int(c.LAT < 34.0 and c.LON < 127.2)))
    st = pd.DataFrame(rows).sort_values("STN")
    st.to_csv(HERE / "aws_stations.csv", index=False)
    print("보강망", len(st), "지점 (이전 후 구간 사용", int((st.break_ym > 0).sum()), ")")
    return st


def precip_check(P, st, base, Pb):
    """30 km 안 기준망 지점과 월강수 상관(공통 기간) 0.85 이상이면 사용."""
    ok = np.ones(len(st), int)
    d = _dist(st.lat.values[:, None], st.lon.values[:, None], base.lat.values[None], base.lon.values[None])
    for j in range(len(st)):
        n = int(np.argmin(d[j]))
        if d[j, n] > P_KM:
            continue
        x, y = P[:, j], Pb[:, n]; k = np.isfinite(x) & np.isfinite(y)
        r = np.corrcoef(x[k], y[k])[0, 1] if k.sum() > 24 else np.nan
        ok[j] = int(np.isfinite(r) and r >= P_MIN_R)
    return ok


def build(outdir, tm2):
    st = pd.read_csv(HERE / "aws_stations.csv")
    base = pd.read_csv(HERE / "base_stations.csv")
    t = pd.period_range(f"{T0[:4]}-{T0[4:]}", f"{tm2[:4]}-{tm2[4:]}", freq="M")
    out = {}
    for var, nm, sts in (("ta", "T", st), ("rn", "P", st), ("rn", "Pb", base)):
        d = fetch(var, sts.STN.tolist(), tm2)
        d["per"] = pd.PeriodIndex([f"{x // 100}-{x % 100:02d}" for x in d.TM], freq="M")
        a = d.pivot_table(index="per", columns="STN", values="VAL", aggfunc="first").reindex(index=t, columns=sts.STN.tolist())
        if nm != "Pb":
            brk = sts.set_index("STN").break_ym
            for s in a.columns:                     # 이전 전 구간 버림
                if brk[s] > 0:
                    a.loc[a.index <= pd.Period(f"{brk[s] // 100}-{brk[s] % 100:02d}", "M"), s] = np.nan
        out[nm] = a
    st["p_ok"] = precip_check(out["P"].values, st, base, out["Pb"].values)
    out["P"].loc[:, st.STN[st.p_ok == 0].values] = np.nan
    st.to_csv(HERE / "aws_stations.csv", index=False)
    ds = xr.Dataset({"T": (("time", "stn"), out["T"].values.astype("float32")),
                     "P": (("time", "stn"), out["P"].div(t.days_in_month, axis=0).values.astype("float32"))},
                    coords=dict(time=t.to_timestamp(), stn=st.STN.values, lat=("stn", st.lat.values), lon=("stn", st.lon.values),
                                alt=("stn", st.alt.values), name=("stn", st.name.fillna("").astype(str).values),
                                jeju=("stn", st.jeju.values.astype(int))))
    ds.attrs.update(source="KMA API hub climate statistics sts_ta/sts_rn monthly", freq="month")
    Path(outdir).mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(Path(outdir) / "target_aws.nc")
    print("target_aws", dict(ds.sizes), "강수 사용", int(st.p_ok.sum()), "결측 T", round(float(ds["T"].isnull().mean()), 3))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    tm2 = (pd.Timestamp.now(tz="Asia/Seoul") - pd.offsets.MonthBegin(1)).strftime("%Y%m")
    if "--select" in sys.argv or not (HERE / "aws_stations.csv").exists():
        select(tm2)
    build(args[0] if args else "/home/user/kds/aws", tm2)
