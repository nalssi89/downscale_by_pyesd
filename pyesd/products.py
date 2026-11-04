"""표출용 확률 산출 — 멤버별 투영 편차 + 교차검증 잔차 σ 정규혼합 (objective_climate_outlook 사이트 field 형식).

  below/near/above  3분위 확률 (관측 1991~2020 편차 분위 q33·q67)
  gt_median         평년 중앙값 초과 확률,  lt_p10 / gt_p90  극한(하위·상위 10 %) 확률
  p10~p90           예측범위 백분위 (기온: 편차 °C, 강수: 평년비 %)
  thr               임계값 확률: 기온 편차 ±1·±2 °C, 강수 평년비 50·150·200 %, 기간 누적 mm (주 10·30·50·100, 월 50·100·200·300)
"""
import numpy as np
from scipy.stats import norm

T_ANOM_THR = (-2, -1, 1, 2)
P_RATIO_THR = (50, 150, 200)
P_ABS_THR = {"week": (10, 30, 50, 100), "month": (50, 100, 200, 300)}
PCT_KEYS = ("below", "near", "above", "gt_median", "lt_p10", "gt_p90")


def cdf(x, mu, s):
    """P(Z < x), Z ~ 멤버별 N(mu_k, s) 혼합. mu (member, npt), s (npt,), x (npt,) 또는 스칼라"""
    return norm.cdf((x - mu) / s).mean(0)


def fields(mu, s, q, ycl, var, kind, ndays, normal=None, k=40, seed=0):
    """mu (member, npt) 편차(강수는 sqrt 공간), s σ (npt), q dict q10·q33·q50·q67·q90 (npt) 관측 편차 분위,
    ycl 기후값 (npt; 강수 sqrt(mm/일)), kind 'week'|'month', ndays 기간 일수, normal 강수 평년 (mm/일).
    반환 dict: anom, spread, value, clim, below/near/above, gt_median, lt_p10, gt_p90, p10~p90, thr{...}, (강수) normal, pct"""
    r = dict(anom=mu.mean(0), spread=mu.std(0), below=cdf(q["q33"], mu, s), above=1 - cdf(q["q67"], mu, s),
             gt_median=1 - cdf(q["q50"], mu, s), lt_p10=cdf(q["q10"], mu, s), gt_p90=1 - cdf(q["q90"], mu, s))
    r["near"] = 1 - r["below"] - r["above"]
    rng = np.random.default_rng(seed)
    z = (mu[:, None, :] + s[None, None, :] * rng.standard_normal((mu.shape[0], k, mu.shape[1]))).reshape(-1, mu.shape[1])
    Q = np.percentile(z, (10, 25, 50, 75, 90), 0)
    thr = {}
    if var == "T":
        r["value"] = ycl + r["anom"]; r["clim"] = ycl
        for i, qq in enumerate((10, 25, 50, 75, 90)):
            r[f"p{qq}"] = Q[i]
        for kq in ("q10", "q33", "q50", "q67", "q90"):
            r["b" + kq[1:]] = ycl + q[kq]                                      # 평년 분포 경계 (°C)
        for t in T_ANOM_THR:
            thr[f"anom_{'gt' if t > 0 else 'lt'}{t:+d}"] = 1 - cdf(t, mu, s) if t > 0 else cdf(t, mu, s)
    else:
        nm = np.maximum(normal, 0.05)
        val = np.clip(ycl[None] + z, 0, None) ** 2                          # mm/일 표본
        vm = np.clip(ycl[None] + mu, 0, None) ** 2
        r["value"] = vm.mean(0); r["spread"] = vm.std(0); r["normal"] = normal; r["clim"] = normal
        r["pct"] = 100 * r["value"] / nm
        for i, qq in enumerate((10, 25, 50, 75, 90)):
            r[f"p{qq}"] = 100 * np.clip(ycl + Q[i], 0, None) ** 2 / nm
        for kq in ("q10", "q33", "q50", "q67", "q90"):
            r["b" + kq[1:]] = np.clip(ycl + q[kq], 0, None) ** 2                 # 평년 분포 경계 (mm/일)
        ratio = val / nm[None]
        for t in P_RATIO_THR:
            thr[f"ratio_{'gt' if t > 100 else 'lt'}{t}"] = (ratio > t / 100).mean(0) if t > 100 else (ratio < t / 100).mean(0)
        for t in P_ABS_THR[kind]:
            thr[f"abs_gt{t}"] = (val * ndays > t).mean(0)
    r["thr"] = thr
    return r


def to_jsonable(r, var):
    """field dict -> JSON 값 (확률은 정수 %, 그 외 반올림)"""
    rnd = lambda x, nd: [None if not np.isfinite(v) else round(float(v), nd) for v in np.asarray(x)]  # noqa: E731
    pct = lambda x: [None if not np.isfinite(v) else int(round(100 * float(v))) for v in np.asarray(x)]  # noqa: E731
    out = {}
    for f, v in r.items():
        if f == "thr":
            out[f] = {kk: pct(vv) for kk, vv in v.items()}
        elif f in PCT_KEYS:
            out[f] = pct(v)
        elif f == "pct" or (var == "P" and f.startswith("p") and f[1:].isdigit()):
            out[f] = rnd(v, 0)
        else:
            out[f] = rnd(v, 2)
    return out
