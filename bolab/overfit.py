"""How much of the best backtest is luck: deflated Sharpe ratio, PBO and walk-forward selection."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats

EULER = 0.5772156649015329


# ----- deflated Sharpe ratio (Bailey & López de Prado, 2014) ---------------------------------
def expected_max_sharpe(sr_var: float, n_trials: float) -> float:
    """Expected maximum of n_trials Sharpe estimates when every true Sharpe is zero."""
    if n_trials <= 1:
        return 0.0
    z = stats.norm.ppf
    return float(np.sqrt(sr_var) * ((1 - EULER) * z(1 - 1 / n_trials) + EULER * z(1 - 1 / (n_trials * np.e))))


def probabilistic_sharpe(returns: np.ndarray, benchmark_sr: float) -> float:
    """P(true per-period Sharpe > benchmark_sr), allowing for skewness and fat tails."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    sr = r.mean() / r.std(ddof=1)
    skew, kurt = stats.skew(r), stats.kurtosis(r, fisher=False)
    denom = np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr ** 2)
    return float(stats.norm.cdf((sr - benchmark_sr) * np.sqrt(len(r) - 1) / denom))


def deflated_sharpe(returns: np.ndarray, sr_var: float, n_trials: float) -> dict:
    sr0 = expected_max_sharpe(sr_var, n_trials)
    return {"sr0_per_period": sr0, "dsr": probabilistic_sharpe(returns, sr0)}


def effective_trials(M: pd.DataFrame) -> dict:
    """Two estimates of how many independent strategies the family really contains."""
    C = np.corrcoef(M.to_numpy().T)
    n = C.shape[0]
    rho = float((C.sum() - n) / (n * (n - 1)))
    eig = np.linalg.eigvalsh(C)
    return {"n": n, "avg_corr": rho, "n_eff_avg_corr": rho + (1 - rho) * n,
            "n_eff_participation": float(eig.sum() ** 2 / (eig ** 2).sum())}


# ----- probability of backtest overfitting (Bailey, Borwein, López de Prado & Zhu, 2017) ------
def pbo_cscv(M: pd.DataFrame, n_blocks: int = 16) -> dict:
    """Combinatorially symmetric cross-validation.

    Split the sample into n_blocks contiguous blocks; for every way of choosing half of them as
    the in-sample set, pick the strategy with the best in-sample Sharpe and record where it ranks
    out of sample. PBO is the share of splits in which it ranks below the median.
    """
    X = M.to_numpy()
    T = (len(X) // n_blocks) * n_blocks
    X = X[len(X) - T:]
    blocks = X.reshape(n_blocks, T // n_blocks, X.shape[1])
    S1, S2 = blocks.sum(axis=1), (blocks ** 2).sum(axis=1)
    m = T // n_blocks
    combos = np.array(list(itertools.combinations(range(n_blocks), n_blocks // 2)))
    C = np.zeros((len(combos), n_blocks))
    C[np.arange(len(combos))[:, None], combos] = 1.0

    def sharpe(mask):
        n = mask.sum(axis=1, keepdims=True) * m
        s1, s2 = mask @ S1, mask @ S2
        mean = s1 / n
        var = (s2 - n * mean ** 2) / (n - 1)
        return mean / np.sqrt(var)

    sr_in, sr_out = sharpe(C), sharpe(1 - C)
    best = sr_in.argmax(axis=1)
    rows = np.arange(len(combos))
    ranks = (sr_out < sr_out[rows, best][:, None]).sum(axis=1) + 1  # 1 = worst
    omega = ranks / (X.shape[1] + 1)
    logit = np.log(omega / (1 - omega))
    sel_in, sel_out = sr_in[rows, best], sr_out[rows, best]
    slope = np.polyfit(sel_in, sel_out, 1)[0]
    return {"splits": len(combos), "pbo": float((logit <= 0).mean()), "logit": logit,
            "sel_in": sel_in, "sel_out": sel_out, "median_out": np.median(sr_out, axis=1),
            "p_loss": float((sel_out < 0).mean()), "degradation_slope": float(slope)}


# ----- walk-forward selection ----------------------------------------------------------------
def walk_forward(M: pd.DataFrame, train_months: int = 240, test_months: int = 60) -> tuple[pd.Series, list[dict]]:
    """Every test_months, pick the strategy with the best Sharpe over the previous train_months
    and hold it for the next test_months. Returns the out-of-sample series and the picks."""
    out, picks = [], []
    for start in range(train_months, len(M), test_months):
        train = M.iloc[start - train_months:start]
        sr = train.mean() / train.std()
        pick = sr.idxmax()
        test = M[pick].iloc[start:start + test_months]
        out.append(test)
        picks.append({"from": str(test.index[0]), "to": str(test.index[-1]), "pick": pick,
                      "train_sharpe": float(sr.max() * np.sqrt(12)),
                      "test_sharpe": float(test.mean() / test.std() * np.sqrt(12))})
    return pd.concat(out).rename("walk-forward"), picks
