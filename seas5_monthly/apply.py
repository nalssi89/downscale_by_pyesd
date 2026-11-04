"""학습된 SEAS5 월별 모형을 SEAS5 장(재예보·실황)에 적용하는 공통 함수."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "pyesd"))
import predictors as pr  # noqa: E402

MODELS = HERE / "models"


def load_model(kind, var):
    return dict(np.load(MODELS / f"{kind}_{var}.npz", allow_pickle=False))


def model_W(M):
    return M["W"] if "W" in M else pr.unpack_W(M["W_idx"], M["W_val"], M["W_ngrid"])


def model_eofs(M, m):
    e = pr.EOFs()
    e.mu = {v: M[f"m{m}_mu_{v}"] for v in pr.EOF_VARS}; e.sd = {v: M[f"m{m}_sd_{v}"] for v in pr.EOF_VARS}
    e.eof = {v: M[f"m{m}_eof_{v}"] for v in pr.EOF_VARS}; e.pcsd = {v: M[f"m{m}_pcsd_{v}"] for v in pr.EOF_VARS}
    return e


class Applier:
    """SEAS5 편차 dict(member, init, lead, lat, lon; ERA5 격자로 보간된 것)에 모형 적용."""

    def __init__(self, M):
        self.M = M
        self.Wt = pr.Weights.__new__(pr.Weights); self.Wt.W = model_W(M)
        self.idx = [str(x) for x in M["idx"]]; self.patterns = pr.Patterns.from_npz(M)

    def predict(self, A, lead, month):
        """lead 하나 선택 → 달 month 모형으로 편차 예측. 반환 (member, init, npt)"""
        sub = {k: v.sel(lead=[lead]) for k, v in A.items()}
        X = pr.feature_table(sub, self.Wt, model_eofs(self.M, month), ("member", "init", "lead"), self.idx, self.patterns)
        c = self.M[f"m{month}_coef"]
        mu = np.einsum("npj,jp->nj", X, c[:, :-1]) + c[:, -1][None]
        nm, ni = sub["t2m"].sizes["member"], sub["t2m"].sizes["init"]
        return mu.reshape(nm, ni, -1)

    def quantiles(self, mu, month, q=(10, 25, 50, 75, 90), k=40, seed=0):
        """멤버 편차 (member, npt) + 잔차 σ 정규혼합의 백분위 (len(q), npt) — 멤버별 투영에 k개 잔차 표본을 더해 계산"""
        s = self.M[f"m{month}_sigma"]
        rng = np.random.default_rng(seed)
        z = mu[:, None, :] + s[None, None, :] * rng.standard_normal((mu.shape[0], k, mu.shape[1]))
        return np.percentile(z.reshape(-1, mu.shape[1]), q, 0)

    def probs(self, mu, month):
        """멤버 편차 (member, ..., npt) + CV 잔차 σ 정규혼합 → 3분위 확률 (below, near, above)"""
        M = self.M; s = M[f"m{month}_sigma"]
        pb = norm.cdf((M[f"m{month}_q33"] - mu) / s).mean(0)
        pa = 1 - norm.cdf((M[f"m{month}_q67"] - mu) / s).mean(0)
        return pb, 1 - pb - pa, pa


def seas5_anoms_on_grid(hc, Ff, c9120, c9316):
    """재예보 기후(hc = predictors.hind_clim) 기준 편차 + ERA5 기후 차이 보정 → 변수별 ERA5 1° 격자로 보간.
    모형에 없는 확장 변수는 있어도 무시되고, 모형이 요구하는 변수가 없으면 feature_table 에서 KeyError."""
    return pr.regrid(pr.seas5_anoms(hc, Ff, c9120, c9316), c9120)


def lead_month(init, lead):
    return (pd.Timestamp(init) + pd.DateOffset(months=int(lead) - 1)).month
