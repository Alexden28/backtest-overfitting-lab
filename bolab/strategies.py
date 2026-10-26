"""A family of industry momentum strategies, every one computed without look-ahead.

Weights are formed at the end of month t from returns up to t and earn the returns of month t+1.

Cross-sectional (XS, Moskowitz & Grinblatt 1999): rank industries by their return over the
lookback window, buy the top fraction q and sell the bottom fraction q.
Time-series (TS, Moskowitz, Ooi & Pedersen 2012): hold each industry long or short according to
the sign of its own past return, in excess of the T-bill.
Holding H > 1 uses overlapping portfolios (Jegadeesh & Titman 1993): the position is the average
of the last H monthly formations. Costs are charged on the change in weights.
"""
from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

LOOKBACKS = (1, 3, 6, 9, 12, 18, 24, 36)
SKIPS = (0, 1)
HOLDS = (1, 3, 6, 12)
QUANTILES = (0.1, 0.2, 0.3)
WEIGHTINGS = ("equal", "inv_vol")
VOL_WINDOW = 12


@dataclass(frozen=True)
class Config:
    family: str  # "XS" | "TS"
    lookback: int
    skip: int
    hold: int
    weighting: str
    q: float | None = None  # XS only

    @property
    def name(self) -> str:
        q = f"-q{int(self.q * 100)}" if self.q else ""
        w = "ew" if self.weighting == "equal" else "iv"
        return f"{self.family}-L{self.lookback}-S{self.skip}-H{self.hold}{q}-{w}"


def grid() -> list[Config]:
    xs = [Config("XS", L, S, H, w, q)
          for L, S, H, q, w in itertools.product(LOOKBACKS, SKIPS, HOLDS, QUANTILES, WEIGHTINGS)]
    ts = [Config("TS", L, S, H, w) for L, S, H, w in itertools.product(LOOKBACKS, SKIPS, HOLDS, WEIGHTINGS)]
    return xs + ts


def _past_return(R: np.ndarray, lookback: int, skip: int) -> np.ndarray:
    """Compounded return over months t-skip-lookback+1 .. t-skip; NaN unless all are observed."""
    logr = np.log1p(np.nan_to_num(R, nan=0.0))
    obs = np.isfinite(R).astype(float)
    c = np.vstack([np.zeros((1, R.shape[1])), np.cumsum(logr, axis=0)])
    n = np.vstack([np.zeros((1, R.shape[1])), np.cumsum(obs, axis=0)])
    T = R.shape[0]
    out = np.full(R.shape, np.nan)
    for t in range(T):
        hi, lo = t - skip + 1, t - skip - lookback + 1  # rows of c are offset by one
        if lo < 0:
            continue
        full = (n[hi] - n[lo]) == lookback
        out[t] = np.where(full, np.expm1(c[hi] - c[lo]), np.nan)
    return out


def _trailing_vol(R: np.ndarray) -> np.ndarray:
    return pd.DataFrame(R).rolling(VOL_WINDOW, min_periods=VOL_WINDOW).std().to_numpy()


def signal(R: np.ndarray, rf: np.ndarray, cfg: Config) -> np.ndarray:
    """Past return used to rank (XS) or to sign (TS, in excess of the T-bill) at the end of month t."""
    return _past_return(R if cfg.family == "XS" else R - rf[:, None], cfg.lookback, cfg.skip)


def formation_weights(R: np.ndarray, sig: np.ndarray, cfg: Config, vol: np.ndarray) -> np.ndarray:
    """Weights formed at the end of each month, before overlapping."""
    valid = np.isfinite(sig) & np.isfinite(R)
    if cfg.weighting == "inv_vol":
        valid &= np.isfinite(vol) & (vol > 0)
        scale = np.where(valid, 1.0 / np.where(vol > 0, vol, np.nan), 0.0)
    else:
        scale = valid.astype(float)
    W = np.zeros(R.shape)
    for t in range(R.shape[0]):
        idx = np.nonzero(valid[t])[0]
        if cfg.family == "XS":
            if len(idx) < 10:
                continue
            k = max(1, int(np.floor(cfg.q * len(idx))))
            order = idx[np.argsort(sig[t, idx], kind="stable")]
            for leg, sign in ((order[-k:], 1.0), (order[:k], -1.0)):
                s = scale[t, leg]
                W[t, leg] = sign * s / s.sum()
        else:
            if len(idx) == 0:
                continue
            s = scale[t, idx]
            W[t, idx] = np.sign(sig[t, idx]) * s / s.sum()
    return W


def strategy_returns(R: pd.DataFrame, rf: pd.Series, cfg: Config, cost_bps: float = 10.0,
                     vol: np.ndarray | None = None, sig: np.ndarray | None = None) -> pd.Series:
    """Monthly net return of one configuration (XS long-short, TS in excess of the T-bill)."""
    Rv = R.to_numpy()
    rfv = rf.reindex(R.index).to_numpy()
    vol = _trailing_vol(Rv) if vol is None else vol
    sig = signal(Rv, rfv, cfg) if sig is None else sig
    F = formation_weights(Rv, sig, cfg, vol)
    W = pd.DataFrame(F).rolling(cfg.hold, min_periods=cfg.hold).mean().to_numpy()  # overlapping
    formed = pd.Series(np.abs(F).sum(axis=1) > 0).astype(float)
    started = formed.rolling(cfg.hold, min_periods=cfg.hold).min().fillna(0).to_numpy() == 1
    excess = Rv - rfv[:, None] if cfg.family == "TS" else Rv
    nxt = np.nan_to_num(np.roll(excess, -1, axis=0), nan=0.0)
    gross = np.einsum("tn,tn->t", np.nan_to_num(W), nxt)
    turnover = np.abs(np.diff(np.nan_to_num(W), axis=0, prepend=0.0)).sum(axis=1)
    net = gross - cost_bps / 1e4 * turnover
    # the return of weights formed at t is earned in month t+1; the shift also drops the last
    # formation, whose next month is not in the data
    out = pd.Series(np.where(started, net, np.nan), index=R.index).shift(1)
    return out.rename(cfg.name)


def all_returns(R: pd.DataFrame, rf: pd.Series, cost_bps: float = 10.0,
                placebo_seed: int | None = None) -> tuple[pd.DataFrame, list[Config]]:
    """Every configuration's net returns. With placebo_seed, each signal is replaced by random
    noise where it exists: the same search, holdings, turnover and costs, but no information."""
    cfgs = grid()
    Rv, rfv = R.to_numpy(), rf.reindex(R.index).to_numpy()
    vol = _trailing_vol(Rv)
    rng = np.random.default_rng(placebo_seed)
    sigs = {}
    for c in cfgs:  # 32 distinct signals for 512 configurations
        key = (c.family, c.lookback, c.skip)
        if key not in sigs:
            s = signal(Rv, rfv, c)
            if placebo_seed is not None:
                s = np.where(np.isfinite(s), rng.standard_normal(s.shape), np.nan)
            sigs[key] = s
    M = pd.concat([strategy_returns(R, rf, c, cost_bps, vol, sigs[(c.family, c.lookback, c.skip)])
                   for c in cfgs], axis=1)
    return M, cfgs


def config_table(cfgs: list[Config]) -> pd.DataFrame:
    return pd.DataFrame([{**asdict(c), "name": c.name} for c in cfgs]).set_index("name")
