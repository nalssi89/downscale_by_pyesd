"""get_cds.py 요청 목록을 데이터셋 묶음(ERA5 단일면 / ERA5 기압면 / SEAS5)별로 한 번에 1건씩 순차 처리. 이미 CDS에 올라가 있는(대기/성공) 같은 요청은 새로 내지 않고 이어받는다.
  python seas5_monthly/seq_cds.py OUTDIR --init 10,11 --fc-year 2026
"""
import os, sys, time, json, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import get_cds as g
from ecmwf.datastores import Client

KEYS = ("variable", "year", "month", "area", "pressure_level", "leadtime_month", "product_type", "system", "grid")
def norm(req):
    o = {}
    for k in KEYS:
        if k in req:
            v = req[k]; v = v if isinstance(v, list) else [v]
            o[k] = sorted(str(float(x)) if k in ("area", "grid") else str(x) for x in v)
    return json.dumps(o, sort_keys=True)

out = Path(sys.argv[1]); init = sys.argv[sys.argv.index("--init") + 1]; fy = int(sys.argv[sys.argv.index("--fc-year") + 1])
c = Client(url=g.CDS_URL, key=os.environ["CDSAPI_KEY"])
existing = {}
for j in c.get_jobs(limit=100).json["jobs"]:
    if j["status"] in ("accepted", "running", "successful"):
        try:
            r = c.get_remote(j["jobID"]); existing.setdefault((j["processID"], norm(r.request)), r)
        except Exception: pass
print(f"{len(existing)} reusable jobs on CDS", flush=True)

reqs, merges = [], {}
for part, ds, req, final in g.era5_requests():
    merges.setdefault(final, []).append(part)
    if not (out / final).exists(): reqs.append((part, ds, req))
reqs += g.seas5_requests([int(m) for m in init.split(",")], fy)
# 이미 CDS에 대기 중인 요청을 앞으로
reqs.sort(key=lambda x: 0 if any((x[1], norm(r)) in existing for r in (x[2] if isinstance(x[2], list) else [x[2]])) else 1)
todo = [x for x in reqs if not ((out / x[0]).exists() and (out / x[0]).stat().st_size > 0)]
print(f"{len(todo)} to fetch sequentially", flush=True)

def queue_full(rem):
    try:
        import requests
        t = requests.get(rem.url + "/results", headers={"PRIVATE-TOKEN": os.environ["CDSAPI_KEY"]}, timeout=60).text
        return "queued requests" in t or "available slots" in t
    except Exception:
        return False
def lane_of(ds):
    return "era5_sl" if ds.startswith("reanalysis-era5-single") else "era5_pl" if ds.startswith("reanalysis-era5-pressure") else "seas5"
def run_lane(todo):
    for i, (name, ds, req) in enumerate(todo, 1):
        f = out / name; f.parent.mkdir(parents=True, exist_ok=True); t0 = time.time(); msg = ""
        for r in (req if isinstance(req, list) else [req]):
            try:
                rem = existing.get((ds, norm(r)))
                for attempt in range(8):
                    if rem is None:
                        rem = c.submit(ds, r)
                    print(f"[{i}/{len(todo)}] {name} job {rem.request_id[:8]} {rem.status}", flush=True)
                    while rem.status not in ("successful", "failed", "rejected", "dismissed", "deleted"):
                        time.sleep(30)
                    if rem.status == "rejected" and queue_full(rem):      # 큐 한도 거절 → 10분 뒤 같은 요청 재제출
                        print(f"   queue full, retry in 10 min ({name})", flush=True); time.sleep(600); rem = None; continue
                    break
                if rem.status != "successful":
                    raise RuntimeError(f"status {rem.status}: {rem.json.get('metadata', {})}")
                tmp = f.with_suffix(".part"); rem.download(str(tmp)); tmp.rename(f)
                msg = f"ok {name} {f.stat().st_size // 1024} KB {time.time() - t0:.0f}s"; break
            except Exception as e:
                msg = f"FAIL {name} {time.time() - t0:.0f}s {str(e)[:300]}"
                (out / f"log_{Path(name).name}.txt").write_text(traceback.format_exc())
        print(msg, time.strftime("%H:%M"), flush=True)


from concurrent.futures import ThreadPoolExecutor
lanes = {}
for x in todo: lanes.setdefault(lane_of(x[1]), []).append(x)
# SEAS5: 실황 11월(13일 공개)은 맨 뒤
lanes.get("seas5", []).sort(key=lambda x: ("_fc" in x[0] and "_11_" in x[0], "_fc" in x[0]))
print({k: len(v) for k, v in lanes.items()}, flush=True)
with ThreadPoolExecutor(len(lanes) or 1) as ex: list(ex.map(run_lane, lanes.values()))

for final, parts in merges.items():
    try: print(g.merge_era5(out, final, parts), flush=True)
    except Exception as e: print("merge FAIL", final, e, flush=True)
print("DONE", time.strftime("%H:%M"), flush=True)
