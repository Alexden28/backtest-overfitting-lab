"""Performance statistics and factor regressions with Newey-West standard errors."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def summary(r: pd.Series) -> dict:
    r = r.dropna()
    wealth = (1 + r).cumprod()
    dd = wealth / wealth.cummax() - 1
    sr = r.mean() / r.std()
    return {"months": int(len(r)), "ann_return": float(r.mean() * 12), "ann_vol": float(r.std() * np.sqrt(12)),
            "sharpe": float(sr * np.sqrt(12)), "t_stat": float(sr * np.sqrt(len(r))),
            "max_drawdown": float(dd.min()), "skew": float(stats.skew(r)),
            "kurtosis": float(stats.kurtosis(r, fisher=False)), "worst_month": float(r.min()),
            "worst_month_date": str(r.idxmin())}


def ols_hac(y: np.ndarray, X: np.ndarray, lags: int) -> tuple[np.ndarray, np.ndarray, float]:
    """OLS coefficients, Newey-West (Bartlett kernel) standard errors and R²."""
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    u = y - X @ beta
    Xu = X * u[:, None]
    S = Xu.T @ Xu
    for k in range(1, lags + 1):
        w = 1 - k / (lags + 1)
        G = Xu[k:].T @ Xu[:-k]
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    r2 = 1 - u.var() / y.var()
    return beta, np.sqrt(np.diag(cov)), float(r2)


def factor_alpha(r: pd.Series, factors: pd.DataFrame, cols: list[str], lags: int = 12) -> dict:
    """Annualised alpha and loadings of r on the given factors, HAC (Newey-West) t-statistics."""
    df = pd.concat([r.rename("r"), factors[cols]], axis=1).dropna()
    X = np.column_stack([np.ones(len(df)), df[cols].to_numpy()])
    beta, se, r2 = ols_hac(df["r"].to_numpy(), X, lags)
    t = beta / se
    return {"from": str(df.index[0]), "to": str(df.index[-1]), "alpha_ann": float(beta[0] * 12),
            "alpha_t": float(t[0]), "r2": r2,
            **{f"beta_{c}": float(b) for c, b in zip(cols, beta[1:])},
            **{f"t_{c}": float(x) for c, x in zip(cols, t[1:])}}
