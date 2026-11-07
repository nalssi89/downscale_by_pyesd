"""북한 27지점 GTS 지상관측(API허브 nko_sfctm.php) 09·21시(KST) 자료 받기.

nko_sfctm 은 한 시각(tm)씩만 주고 기간 조회(tm1~tm2)는 빈 응답이라, 하루 두 번(09시: 최저기온·전날21~09 강수,
21시: 최고기온·09~21 강수)을 받아 월자료를 만든다. 연도별 파일로 저장해 끊겨도 이어 받는다.

  KMA_AUTHKEY=... python training/fetch_nk_synop.py OUTDIR [Y1 Y2 WORKERS]
"""
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pyesd"))
from kma_util import _get  # noqa: E402

COLS = ["TM", "STN", "TA", "TA_MIN", "TA_MAX", "RN", "WD", "WS", "TD", "HM", "WC", "VV", "CA_TOT", "CA_MID", "CT", "PA", "PS"]


FAILED = []


def one(tm):
    try:
        txt = _get("nko_sfctm.php", dict(tm=tm, stn="", help=0))
    except RuntimeError:                       # 504 등 일시 장애: 기록해 두고 나중에 다시 받는다
        FAILED.append(tm)
        return []
    return [ln.split() for ln in txt.splitlines() if ln[:1].isdigit()]


def year(y, out, workers):
    f = out / f"nk_{y}.csv"
    if f.exists():
        return
    days = pd.date_range(f"{y}-01-01", min(pd.Timestamp(f"{y}-12-31"), pd.Timestamp.now() - pd.Timedelta(days=1)))
    tms = [d.strftime("%Y%m%d") + h for d in days for h in ("0900", "2100")]
    with ThreadPoolExecutor(workers) as ex:
        rows = [r for rr in ex.map(one, tms) for r in rr]
    df = pd.DataFrame([r[:17] for r in rows if len(r) >= 17], columns=COLS)
    df.to_csv(f, index=False)
    print(y, len(tms), "calls", len(df), "rows", "failed", len(FAILED), flush=True)
    if FAILED:
        with open(out / "failed.txt", "a") as fh:
            fh.write("\n".join(FAILED) + "\n")
        FAILED.clear()


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/user/kds/nk/raw"); out.mkdir(parents=True, exist_ok=True)
    y1, y2, w = (int(a) for a in (sys.argv[2:5] if len(sys.argv) > 4 else (1991, 2026, 4)))
    for y in range(y2, y1 - 1, -1):          # 최근부터 (평년 2021 미제공 → 최근 자료 먼저 확인)
        year(y, out, w)
