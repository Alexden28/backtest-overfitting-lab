"""Offline tests on synthetic data: no look-ahead, and the overfitting statistics behave as theory says."""
import numpy as np
import pandas as pd
import pytest

from bolab.metrics import factor_alpha
from bolab.overfit import deflated_sharpe, expected_max_sharpe, pbo_cscv, probabilistic_sharpe, walk_forward
from bolab.strategies import Config, strategy_returns


def panel(T=400, N=30, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.period_range("1950-01", periods=T, freq="M")
    R = pd.DataFrame(rng.normal(0.008, 0.05, (T, N)), index=idx, columns=[f"i{k}" for k in range(N)])
    rf = pd.Series(0.002, index=idx)
    return R, rf


@pytest.mark.parametrize("cfg", [Config("XS", 12, 1, 1, "equal", 0.3), Config("XS", 3, 0, 6, "inv_vol", 0.2),
                                 Config("TS", 6, 0, 3, "equal"), Config("TS", 12, 1, 12, "inv_vol")])
def test_no_look_ahead(cfg):
    """Changing returns from month k on must not change the strategy's returns before k."""
    R, rf = panel()
    k = 250
    base = strategy_returns(R, rf, cfg)
    shocked = R.copy()
    shocked.iloc[k:] = shocked.iloc[k:] * -3 + 0.1
    after = strategy_returns(shocked, rf, cfg)
    pd.testing.assert_series_equal(base.iloc[:k], after.iloc[:k])
    assert not np.allclose(base.iloc[k + 1:].dropna(), after.iloc[k + 1:].dropna())


def test_cross_sectional_momentum_is_long_short():
    R, rf = panel()
    planted = R.copy()
    planted.iloc[:, 0] += 0.02  # one industry that always wins
    r = strategy_returns(planted, rf, Config("XS", 12, 1, 1, "equal", 0.1), cost_bps=0)
    # 30 industries, q = 10% -> 3 per leg: the winner sits in the long leg at weight 1/3
    assert r.mean() == pytest.approx(0.02 / 3, abs=0.002)


def test_pbo_is_about_half_for_pure_noise():
    rng = np.random.default_rng(1)
    M = pd.DataFrame(rng.normal(0, 0.04, (960, 100)))
    assert 0.35 < pbo_cscv(M)["pbo"] < 0.65


def test_pbo_is_low_when_one_strategy_has_real_skill():
    rng = np.random.default_rng(2)
    X = rng.normal(0, 0.04, (960, 100))
    X[:, 0] += 0.02
    assert pbo_cscv(pd.DataFrame(X))["pbo"] < 0.05


def test_deflated_sharpe():
    rng = np.random.default_rng(3)
    r = rng.normal(0.01, 0.04, 600)  # true monthly Sharpe 0.25
    assert expected_max_sharpe(0.01, 1) == 0
    assert expected_max_sharpe(0.01, 1000) > expected_max_sharpe(0.01, 10) > 0
    one = deflated_sharpe(r, 0.01, 1)["dsr"]
    assert one == pytest.approx(probabilistic_sharpe(r, 0.0))
    assert deflated_sharpe(r, 0.01, 1000)["dsr"] < one


def test_walk_forward_only_uses_the_past():
    rng = np.random.default_rng(4)
    M = pd.DataFrame(rng.normal(0, 0.04, (360, 5)), columns=list("abcde"))
    M.iloc[:240, 2] += 0.03  # best in the first training window only
    wf, picks = walk_forward(M, train_months=240, test_months=60)
    assert picks[0]["pick"] == "c" and len(wf) == 120


def test_factor_alpha_recovers_known_alpha():
    rng = np.random.default_rng(5)
    idx = pd.period_range("1960-01", periods=600, freq="M")
    f = pd.DataFrame({"MKT": rng.normal(0.006, 0.045, 600)}, index=idx)
    r = pd.Series(0.004 + 0.5 * f["MKT"] + rng.normal(0, 0.01, 600), index=idx)
    a = factor_alpha(r, f, ["MKT"])
    assert a["alpha_ann"] == pytest.approx(0.048, abs=0.01) and a["beta_MKT"] == pytest.approx(0.5, abs=0.05)
