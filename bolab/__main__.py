"""python -m bolab  — run every experiment and write reports/ (results.json, results.md, figures)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .data import ROOT, factors, industries
from .metrics import factor_alpha, summary
from .overfit import deflated_sharpe, effective_trials, pbo_cscv, walk_forward
from .plots import plot_all
from .strategies import Config, all_returns, config_table, strategy_returns

OUT = ROOT / "reports"
TEXTBOOK = Config("XS", 12, 1, 1, "equal", 0.3)  # 12-1 momentum, monthly rebalance, top/bottom 30%
FF3_UMD = ["Mkt-RF", "SMB", "HML", "UMD"]
FF5_UMD = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "UMD"]
ERAS = [("1930-07", "1962-12"), ("1963-01", "1999-12"), ("2000-01", "2026-12")]


def ann_sharpe(X: pd.DataFrame) -> pd.Series:
    return X.mean() / X.std() * np.sqrt(12)


def main() -> None:
    R, f = industries(), factors()
    M, cfgs = all_returns(R, f["RF"])
    start = M.apply(lambda s: s.first_valid_index()).max()
    M = M.loc[start:]
    P = all_returns(R, f["RF"], placebo_seed=1)[0].loc[start:]
    P0 = all_returns(R, f["RF"], cost_bps=0, placebo_seed=1)[0].loc[start:]
    print(f"{M.shape[1]} strategies, {len(M)} months ({M.index[0]} to {M.index[-1]})")

    sr_m = M.mean() / M.std()
    best = sr_m.idxmax()
    trials = effective_trials(M)
    dsr = {f"n={n:.0f}": deflated_sharpe(M[best].to_numpy(), float(sr_m.var()), n)
           for n in (trials["n"], trials["n_eff_avg_corr"], trials["n_eff_participation"])}
    for v in dsr.values():
        v["sr0_annual"] = v.pop("sr0_per_period") * np.sqrt(12)

    cscv, cscv_p, cscv_p0 = pbo_cscv(M), pbo_cscv(P), pbo_cscv(P0)
    wf, picks = walk_forward(M)
    textbook = M[TEXTBOOK.name]
    umd = f["UMD"].loc[start:]

    series = {"best in hindsight (" + best + ")": M[best], "walk-forward selection": wf,
              "textbook 12-1 momentum": textbook, "UMD factor (stocks)": umd}
    perf = {k: summary(s) for k, s in series.items()}
    alphas = {k: {"FF3+UMD": factor_alpha(s, f, FF3_UMD), "FF5+UMD": factor_alpha(s, f, FF5_UMD)}
              for k, s in series.items() if not k.startswith("UMD")}
    costs = {c.name: {str(bps): float(ann_sharpe(strategy_returns(R, f["RF"], c, bps).loc[start:].to_frame()).iloc[0])
                      for bps in (0, 10, 25, 50)}
             for c in (TEXTBOOK, next(c for c in cfgs if c.name == best))}
    eras = {name: {f"{a[:4]}-{b[:4]}": float(ann_sharpe(s.loc[a:b].to_frame()).iloc[0]) for a, b in ERAS}
            for name, s in (("textbook", textbook), ("best", M[best]), ("walk-forward", wf))}

    def pbo_view(p):
        return {"pbo": p["pbo"], "splits": p["splits"], "p_loss": p["p_loss"],
                "degradation_slope": p["degradation_slope"],
                "selected_in_sharpe_median": float(np.median(p["sel_in"]) * np.sqrt(12)),
                "selected_out_sharpe_median": float(np.median(p["sel_out"]) * np.sqrt(12)),
                "all_out_sharpe_median": float(np.median(p["median_out"]) * np.sqrt(12))}

    sr_real, sr_plac = ann_sharpe(M), ann_sharpe(P)
    result = {
        "sample": [str(M.index[0]), str(M.index[-1]), len(M)], "n_strategies": M.shape[1],
        "best": best, "trials": trials, "deflated_sharpe": dsr,
        "sharpe_distribution": {"real": sr_real.describe().to_dict(), "placebo": sr_plac.describe().to_dict()},
        "pbo": {"real": pbo_view(cscv), "placebo_net": pbo_view(cscv_p), "placebo_no_costs": pbo_view(cscv_p0)},
        "performance": perf, "alphas": alphas, "cost_sensitivity": costs, "eras": eras, "walk_forward_picks": picks,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    config_table(cfgs).assign(sharpe=sr_real).sort_values("sharpe", ascending=False).to_csv(OUT / "strategies.csv")
    plot_all(M, P, sr_real, sr_plac, cscv, series, cfgs, dsr, best, OUT)
    print(json.dumps({k: result[k] for k in ("best", "deflated_sharpe", "pbo")}, indent=1, default=str))


if __name__ == "__main__":
    main()
