"""기준망(약 85지점) 월자료 만들기 — 기상청 API허브 기후통계(sts_ta, sts_rn) 월자료.

기준망 = ASOS 66지점(training/asos66_stations.csv) + 지점번호 300 이하 가운데
         1991~2026년 월별 유효 연수 20년 이상이고, 이전 5 km 미만·고도 변화 50 m 미만인 지점.
선정 결과는 training/base_stations.csv 에 고정해 둔다 (--select 로 다시 선정).

  KMA_AUTHKEY=... python training/build_base_network.py [--select] OUTDIR

출력: OUTDIR/target_base.nc  (time=월 시작일, stn)  T: 월평균기온(°C), P: 월평균 일강수량(mm/일 = 월합계/일수)
주의: sts API는 요청 종료시각(tm2) 시점의 지점정보로 자료를 거르므로 tm2를 반드시 최근 달로 두고 한 번에 받는다.
      (연도별로 tm2를 과거로 두면 그 시점 지점정보가 없는 지점의 옛 자료가 빠진다.)
"""
import sys
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pyesd"))
from kma_util import _get  # noqa: E402

T0 = "199101"
CHUNK = 25
MIN_YEARS, MAX_MOVE_KM, MAX_DALT = 20, 5.0, 50.0


def fetch(var, stns, tm2):
    """sts_{var} 월자료 (TM, STN, LAT, LON, ALTD, 값...)를 지점 묶음별로 받아 합친다."""
    out = []
    for i in range(0, len(stns), CHUNK):
        ids = ",".join(map(str, stns[i:i + CHUNK]))
        txt = _get(f"sts_{var}.php", dict(tm1=T0, tm2=tm2, stn_id=ids, disp=0, help=0))
        rows = [ln.rstrip("=").rstrip(",") for ln in txt.splitlines() if ln[:1].isdigit()]
        if rows:
            out.append(pd.read_csv(StringIO("\n".join(rows)), header=None))
    df = pd.concat(out, ignore_index=True)
    df = df.rename(columns={0: "TM", 1: "STN", 2: "LAT", 3: "LON", 4: "ALT", 5: "VAL"})
    df = df[["TM", "STN", "LAT", "LON", "ALT", "VAL"]]
    df.loc[df.VAL <= -99, "VAL"] = np.nan
    return df


def station_info():
    """지점정보(stn_inf) ASOS·AWS: 지점번호 → 이름, 고도."""
    rec = {}
    for inf, col in (("SFC", 10), ("AWS", 8)):          # 이름 열 위치가 SFC·AWS 응답에서 다르다
        txt = _get("stn_inf.php", dict(inf=inf, stn="", help=0))
        for ln in txt.splitlines():
            p = ln.split()
            if ln.startswith("#") or len(p) <= col or not p[0].isdigit():
                continue
            rec.setdefault(int(p[0]), p[col])
    return rec


def select(tm2):
    """기준망 선정. 이전 이력은 연도별(tm2=그 해 12월) 호출로 좌표가 시기별로 달리 나오는 것을 이용한다."""
    a66 = pd.read_csv(HERE / "asos66_stations.csv")
    cand = list(range(90, 301))
    ta = fetch("ta", cand, tm2)
    ta["year"] = ta.TM // 100; ta["mon"] = ta.TM % 100
    ok = ta[ta.VAL.notna() & ta.year.between(1991, int(tm2[:4]))]
    minyrs = ok.groupby(["STN", "mon"]).year.nunique().unstack().reindex(columns=range(1, 13)).fillna(0).min(axis=1)
    last = ok.groupby("STN").year.max()
    keep = minyrs[(minyrs >= MIN_YEARS) & (last.reindex(minyrs.index) >= int(tm2[:4]) - 1)].index
    # 이전 이력: 1997~현재 연도별 호출에서 좌표·고도 범위
    hist = []
    for y in range(1997, int(tm2[:4]) + 1):
        d = fetch("ta", list(keep), f"{y}12" if y < int(tm2[:4]) else tm2)
        hist.append(d[["STN", "LAT", "LON", "ALT"]])
    h = pd.concat(hist).drop_duplicates()
    g = h.groupby("STN")
    move = np.hypot((g.LAT.max() - g.LAT.min()) * 111.0, (g.LON.max() - g.LON.min()) * 88.0)
    dalt = g.ALT.max() - g.ALT.min()
    names = station_info()
    cur = ta.sort_values("TM").groupby("STN")[["LAT", "LON", "ALT"]].last()
    rows = []
    for s in keep:
        in66 = s in set(a66.STN)
        if not in66 and (move.get(s, 0) >= MAX_MOVE_KM or dalt.get(s, 0) >= MAX_DALT):
            continue
        r = cur.loc[s]
        jeju = int(a66.set_index("STN").jeju.get(s, 0)) if in66 else int(r.LAT < 34.0 and r.LON < 127.2)
        rows.append(dict(STN=s, name=a66.set_index("STN").name.get(s, names.get(s, "")), lat=r.LAT, lon=r.LON,
                         alt=r.ALT, jeju=jeju, asos66=int(in66), min_years=int(minyrs[s]),
                         move_km=round(float(move.get(s, 0)), 2), dalt_m=round(float(dalt.get(s, 0)), 1)))
    st = pd.DataFrame(rows).sort_values("STN")
    st.to_csv(HERE / "base_stations.csv", index=False)
    print("기준망", len(st), "지점 (ASOS66", int(st.asos66.sum()), "+ 추가", int((st.asos66 == 0).sum()), ")")
    return st


def precip_check(a, st, max_km=30.0, min_r=0.9):
    """추가 지점의 월강수량이 30 km 안의 ASOS 66 지점과 상관 0.9 이상인지 본다 (공항 우량계 등 이질 자료 거르기).
    30 km 안에 ASOS가 없으면(도서) 통과. 반환: 지점별 1(사용)/0(제외)"""
    a66 = st.asos66.values == 1
    la, lo = np.deg2rad(st.lat.values), np.deg2rad(st.lon.values)
    d = 6371 * np.arccos(np.clip(np.sin(la[:, None]) * np.sin(la) + np.cos(la[:, None]) * np.cos(la) * np.cos(lo[:, None] - lo), -1, 1))
    ok = np.ones(len(st), int)
    for j in np.where(~a66)[0]:
        c = np.where(a66)[0]; n = c[np.argmin(d[j, c])]
        if d[j, n] > max_km:
            continue
        x, y = a.iloc[:, j].values, a.iloc[:, n].values; k = np.isfinite(x) & np.isfinite(y)
        r = np.corrcoef(x[k], y[k])[0, 1] if k.sum() > 24 else np.nan
        ok[j] = int(np.isfinite(r) and r >= min_r)
        print(f"  강수 점검 {st.name.iloc[j]} ↔ {st.name.iloc[n]} {d[j, n]:.0f} km r={r:.2f} {'사용' if ok[j] else '제외'}")
    return ok


def build(outdir, tm2):
    st = pd.read_csv(HERE / "base_stations.csv")
    stns = st.STN.tolist()
    t = pd.period_range(f"{T0[:4]}-{T0[4:]}", f"{tm2[:4]}-{tm2[4:]}", freq="M")
    data = {}
    for var, nm in (("ta", "T"), ("rn", "P")):
        d = fetch(var, stns, tm2)
        d["per"] = pd.PeriodIndex([f"{x // 100}-{x % 100:02d}" for x in d.TM], freq="M")
        a = d.pivot_table(index="per", columns="STN", values="VAL", aggfunc="first").reindex(index=t, columns=stns)
        if nm == "P":                                    # 월합계(mm) → 월평균 일강수량(mm/일)
            st["p_ok"] = precip_check(a, st)
            a.loc[:, st.STN[st.p_ok == 0].values] = np.nan  # 강수 품질 미달 지점은 기온에만 쓴다
            a = a.div(t.days_in_month, axis=0)
        data[nm] = (("time", "stn"), a.values.astype("float32"))
    st.to_csv(HERE / "base_stations.csv", index=False)
    ds = xr.Dataset(data, coords=dict(time=t.to_timestamp(), stn=stns,
                                      lat=("stn", st.lat.values), lon=("stn", st.lon.values),
                                      alt=("stn", st.alt.values), name=("stn", st.name.fillna("").astype(str).values),
                                      jeju=("stn", st.jeju.values.astype(int)), asos66=("stn", st.asos66.values.astype(int))))
    ds.attrs.update(source="KMA API hub climate statistics sts_ta/sts_rn monthly", freq="month")
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out / "target_base.nc")
    print("target_base", dict(ds.sizes), "결측 T", float(ds["T"].isnull().mean()), "P", float(ds["P"].isnull().mean()))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    tm2 = (pd.Timestamp.now(tz="Asia/Seoul") - pd.offsets.MonthBegin(1)).strftime("%Y%m")   # 직전 달
    if "--select" in sys.argv or not (HERE / "base_stations.csv").exists():
        select(tm2)
    build(args[0] if args else "/home/user/kds/work", tm2)
