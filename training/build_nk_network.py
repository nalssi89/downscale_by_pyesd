"""북한 27지점(GTS) 월자료 만들기 — fetch_nk_synop.py 가 받은 09·21시(KST) 자료에서.

기온: 일평균 = (최저기온[09시] + 최고기온[21시]) / 2 (없으면 (09시+21시)/2 + 지점·달별 차이), 한 달 60% 이상 날이면 월평균.
강수: 09시(전날 21~09시)·21시(09~21시) 12시간 강수. 보고가 있는데(기온 있음) RN 이 -9 이면 무강수(0)로 본다
      (GTS SYNOP 강수군 생략 = 무강수). 보고 70% 이상이면 월값 = 보고 평균 x 2 (mm/일).
1981~2010 평년(API sfc_nko_norm1 tmst=2011)과 같은 기간 평균을 비교해 점검표를 남긴다.

  python training/build_nk_network.py RAWDIR OUTDIR
출력: OUTDIR/target_nk.nc (time, stn)  T °C, P mm/일; OUTDIR/nk_check.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
T0 = "1991-01"


def monthly(raw):
    d = pd.concat([pd.read_csv(f) for f in sorted(Path(raw).glob("nk_*.csv"))], ignore_index=True)
    d = d.drop_duplicates(["TM", "STN"])
    d["t"] = pd.to_datetime(d.TM.astype(str), format="%Y%m%d%H%M")
    d["h"] = d.t.dt.hour; d["date"] = d.t.dt.normalize()
    for c in ("TA", "TA_MIN", "TA_MAX"):
        d.loc[(d[c] <= -50) | (d[c] > 42), c] = np.nan            # 결측(-99)·이상값(최고 52 °C 등) 제거
    rep = d.TA.notna() | (d.RN >= 0)
    d["R"] = np.where(d.RN >= 0, d.RN, np.where(rep, 0.0, np.nan))
    a = d[d.h == 9].set_index(["date", "STN"]); b = d[d.h == 21].set_index(["date", "STN"])
    day = pd.DataFrame({"mm": (a.TA_MIN + b.TA_MAX) / 2, "obs": (a.TA + b.TA) / 2}).reset_index()
    day.loc[day.mm < -60, "mm"] = np.nan
    # (최고+최저)/2 가 없는 날은 (09시+21시)/2 에 지점·달별 평균 차이를 더해 채운다 (두 값이 다 있는 날로 추정)
    day["m"] = day.date.dt.month
    off = (day.mm - day.obs).groupby([day.STN, day.m]).mean().rename("off")
    day = day.join(off, on=["STN", "m"])
    day["T"] = day.mm.fillna(day.obs + day.off)
    day["ym"] = day.date.dt.to_period("M")
    T = day.groupby(["ym", "STN"]).T.agg(["mean", "count"]).reset_index()
    T.loc[T["count"] < 0.6 * T.ym.dt.days_in_month, "mean"] = np.nan
    d["ym"] = d.date.dt.to_period("M")
    # 강수: 보고된 12시간 강수의 평균 x 2 = mm/일 (보고 70% 이상). 09시 강수(전날 21~09시)는 09시 날짜에 귀속
    P = d.groupby(["ym", "STN"]).R.agg(["mean", "count"]).reset_index()
    P["val"] = np.where(P["count"] >= 0.7 * 2 * P.ym.dt.days_in_month, 2 * P["mean"], np.nan)
    Tw = T.pivot(index="ym", columns="STN", values="mean")
    Pw = P.pivot(index="ym", columns="STN", values="val")
    return Tw, Pw


def norm8110():
    sys.path.insert(0, str(HERE.parent / "pyesd"))
    from kma_util import _get
    txt = _get("sfc_nko_norm1.php", dict(norm="M", tmst=2011, stn=0, MM1=1, DD1=1, MM2=12, DD2=31))
    rows = [ln.rstrip("=").split(",") for ln in txt.splitlines() if ln.strip()[:1].isdigit()]
    n = pd.DataFrame([[int(r[1]), int(r[2]), float(r[3]), float(r[6])] for r in rows], columns=["STN", "MM", "TA", "RN"])
    n["STN"] += 47000
    return n


def build(raw, outdir):
    st = pd.read_csv(HERE / "nk_stations.csv")
    Tw, Pw = monthly(raw)
    t = pd.period_range(T0, Tw.index.max(), freq="M")
    Tw = Tw.reindex(index=t, columns=st.STN); Pw = Pw.reindex(index=t, columns=st.STN)
    # 점검: 1991~2010 자료 평균 vs 1981~2010 평년
    try:
        n = norm8110()
        rows = []
        for s in st.STN:
            for mm in range(1, 13):
                sel = (Tw.index.month == mm) & (Tw.index.year <= 2010)
                rows.append(dict(STN=s, MM=mm, T_obs=Tw.loc[sel, s].mean(), P_obs=Pw.loc[sel, s].mean() * pd.Period(f"2001-{mm:02d}").days_in_month))
        chk = pd.DataFrame(rows).merge(n, on=["STN", "MM"], how="left")
        chk.to_csv(Path(outdir) / "nk_check.csv", index=False)
        print("평년 비교(1991~2010 자료 - 1981~2010 평년): T", round((chk.T_obs - chk.TA).median(), 2), "°C, P 비",
              round((chk.P_obs.sum() / chk.RN[chk.P_obs.notna()].sum()), 2))
    except Exception as e:  # noqa: BLE001
        print("평년 비교 생략:", e)
    ds = xr.Dataset({"T": (("time", "stn"), Tw.values.astype("float32")), "P": (("time", "stn"), Pw.values.astype("float32"))},
                    coords=dict(time=t.to_timestamp(), stn=st.STN.values, lat=("stn", st.lat.values), lon=("stn", st.lon.values),
                                alt=("stn", st.alt.values), name=("stn", st.name.values.astype(str)), jeju=("stn", np.zeros(len(st), int))))
    ds.attrs.update(source="KMA API hub nko_sfctm (North Korea GTS SYNOP) 09/21 KST", freq="month")
    Path(outdir).mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(Path(outdir) / "target_nk.nc")
    print("target_nk", dict(ds.sizes), "결측 T", round(float(ds["T"].isnull().mean()), 3), "P", round(float(ds["P"].isnull().mean()), 3))


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "/home/user/kds/nk/raw", sys.argv[2] if len(sys.argv) > 2 else "/home/user/kds/nk")
