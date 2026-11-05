"""get_cds.py 요청 목록을 데이터셋 묶음(ERA5 단일면 / ERA5 기압면 / SEAS5)별로 한 번에 1건씩 순차 처리. 이미 CDS에 올라가 있는(대기/성공) 같은 요청은 새로 내지 않고 이어받는다.
  python seas5_monthly/seq_cds.py OUTDIR --init 10,11 --fc-year 2026
"""
import os, sys, time, json, traceback, faulthandler, signal
faulthandler.register(signal.SIGUSR1)
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
import seas5_combined as sc
_fin = lambda nm: [f"{nm.replace('combined/', '').replace(k, '')}{x}.nc" for k, xs in (("_slall.nc", ("_sfc", "_sst_nino34", "_sst_io")), ("_plall.nc", ("_pl", "_pl200"))) if k in nm for x in xs]
for x in sc.requests([int(m) for m in init.split(",")], fy):
    fins = _fin(x[0])
    if fins and all((out / f).exists() for f in fins):   # 잘라낸 결과가 이미 다 있으면 요청 안 함
        continue
    reqs.append(x)
# 이미 CDS에 대기 중인 요청을 앞으로
reqs.sort(key=lambda x: 0 if any((x[1], norm(r)) in existing for r in (x[2] if isinstance(x[2], list) else [x[2]])) else 1)
todo = [x for x in reqs if not ((out / x[0]).exists() and (out / x[0]).stat().st_size > 0)]
print(f"{len(todo)} to fetch sequentially", flush=True)

def unzip_if_needed(p):
    """CDS가 zip(여러 stepType nc)으로 주면 하나의 nc로 합쳐 같은 경로에 둔다."""
    import zipfile, tempfile, xarray as xr
    if not zipfile.is_zipfile(p): return
    with tempfile.TemporaryDirectory() as td:
        zipfile.ZipFile(p).extractall(td)
        ds = xr.merge([xr.open_dataset(x).load() for x in sorted(Path(td).glob("*.nc"))], compat="override")
        ds.to_netcdf(str(p) + ".tmp")
    os.replace(str(p) + ".tmp", p)
def queue_full(rem):
    try:
        import requests
        t = requests.get(rem.url + "/results", headers={"PRIVATE-TOKEN": os.environ["CDSAPI_KEY"]}, timeout=60).text
        return "queued requests" in t or "available slots" in t
    except Exception:
        return False
def lane_of(ds):      # ERA5 지상(이미 대기 중인 2건)만 따로, 나머지는 한 줄로: 사용자당 대기열 자리를 1개만 더 씀
    return "era5_sl" if ds.startswith("reanalysis-era5-single") else "main"
def run_lane(todo):
    c = Client(url=g.CDS_URL, key=os.environ["CDSAPI_KEY"])      # 묶음마다 별도 클라이언트
    for i, (name, ds, req) in enumerate(todo, 1):
        f = out / name; f.parent.mkdir(parents=True, exist_ok=True); t0 = time.time(); msg = ""
        if f.exists() and f.stat().st_size > 0:                     # 같은 이름 파일이 이미 있으면 요청하지 않음
            print(f"skip {name} (exists)", flush=True); continue
        for r in (req if isinstance(req, list) else [req]):
            try:
                rem = existing.get((ds, norm(r)))
                for attempt in range(8):
                    if rem is None:
                        rem = c.submit(ds, r)
                    print(f"[{i}/{len(todo)}] {name} job {rem.request_id[:8]} {rem.status}", flush=True)
                    st = rem.status
                    while st not in ("successful", "failed", "rejected", "dismissed", "deleted"):
                        time.sleep(60)
                        try: st = c.get_remote(rem.request_id).status
                        except Exception as e: print("   poll error", name, str(e)[:100], flush=True)
                    print(f"   {name} -> {st} {time.strftime('%H:%M')}", flush=True)
                    rem = c.get_remote(rem.request_id)
                    if rem.status == "rejected" and queue_full(rem):      # 큐 한도 거절 → 10분 뒤 같은 요청 재제출
                        print(f"   queue full, retry in 10 min ({name})", flush=True); time.sleep(600); rem = None; continue
                    break
                if rem.status != "successful":
                    raise RuntimeError(f"status {rem.status}: {rem.json.get('metadata', {})}")
                tmp = f.with_suffix(".part"); rem.download(str(tmp)); unzip_if_needed(tmp); tmp.rename(f)
                if "combined/" in name:
                    print("   split ->", sc.split(out, name), flush=True)
                msg = f"ok {name} {f.stat().st_size // 1024} KB {time.time() - t0:.0f}s"; break
            except Exception as e:
                msg = f"FAIL {name} {time.time() - t0:.0f}s {str(e)[:300]}"
                (out / f"log_{Path(name).name}.txt").write_text(traceback.format_exc())
        print(msg, time.strftime("%H:%M"), flush=True)


from concurrent.futures import ThreadPoolExecutor
lanes = {}
for x in todo: lanes.setdefault(lane_of(x[1]), []).append(x)
# SEAS5: 실황 11월(13일 공개)은 맨 뒤
def _ord(x):
    n = x[0]
    if "era5_pl_2026" in n: return 0                       # 이미 대기 중
    if "_hind_10_" in n: return 1
    if n.startswith("parts/"): return 2
    if "_hind_11_" in n: return 3
    if "_fc" in n and "_10_" in n: return 4
    return 5                                               # 11월 실황(13일 공개)
lanes.get("main", []).sort(key=_ord)
if os.environ.get("DRY"):
    [print(k, x[0]) for k, v in lanes.items() for x in v]; sys.exit()
print({k: len(v) for k, v in lanes.items()}, flush=True)
with ThreadPoolExecutor(len(lanes) or 1) as ex: list(ex.map(run_lane, lanes.values()))

for final, parts in merges.items():
    try: print(g.merge_era5(out, final, parts), flush=True)
    except Exception as e: print("merge FAIL", final, e, flush=True)
print("DONE", time.strftime("%H:%M"), flush=True)
