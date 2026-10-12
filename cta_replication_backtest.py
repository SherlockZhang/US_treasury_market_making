"""
Rolling-regression CTA replication backtest (DBMF-style, reconstructed).

Idea: regress a CTA target's daily excess returns on liquid factor returns over a
trailing window, hold the fitted betas for the next `step` days (weekly rebalance),
and measure how well the out-of-sample replica tracks the target.

Compares estimators (OLS / ridge / bounded LS) x window lengths (30..120 days).

Usage:
    pip install yfinance numpy pandas scipy matplotlib
    python cta_replication_backtest.py                  # target = KMLM (rules-based CTA index ETF)
    python cta_replication_backtest.py --target DBMF    # sanity check vs. actual DBMF (circular but informative)
    python cta_replication_backtest.py --target my_sg_cta.csv   # CSV with columns: date, level (e.g. SG CTA Index)
    python cta_replication_backtest.py --synthetic      # offline smoke test, no downloads

Caveats (read these):
  * Factors are unlevered ETFs (excess of BIL) as free proxies for futures. Futures-based
    tickers (ES=F, ZN=F, 6E=F ...) work too but Yahoo's continuous series are NOT roll-adjusted,
    which injects fake returns on roll days. Swap in Norgate/Barchart/Databento data if you have it.
  * KMLM/CTA are CTA-like ETFs, not the SG CTA Index DBMF targets. The SG index is on SocGen's
    site (manual CSV download) -- use --target path.csv for the faithful version.
  * DBi's real specification is proprietary. This is a generic return-based clone.
"""
import argparse
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

# ~10 factor proxies (ETF total-return series)
FACTORS = ["SPY", "EFA", "EEM",          # equities: US, DM ex-US, EM
           "SHY", "IEF", "TLT",          # Treasuries: ~2y, ~7-10y, 20y+
           "FXE", "FXY",                 # EUR, JPY
           "GLD", "DBC"]                 # gold, broad commodities
RF = "BIL"                               # T-bill proxy for collateral yield
WINDOWS = [30, 60, 90, 120]
METHODS = {
    "ols":     dict(method="ols"),
    "ridge":   dict(method="ridge", lam=0.5),
    "bounded": dict(method="bounded", bound=3.0),
}
STEP = 5  # weekly rebalance


# ----------------------------------------------------------------- data
def load_prices(tickers, start="2019-01-01"):
    import yfinance as yf
    df = yf.download(tickers, start=start, auto_adjust=True, progress=False)["Close"]
    return df.dropna(how="all")


def load_data(target, start="2019-01-01"):
    if target.lower().endswith(".csv"):
        t = pd.read_csv(target, parse_dates=[0], index_col=0).iloc[:, 0].rename("TARGET")
        px = load_prices(FACTORS + [RF, "DBMF"], start)
        px = px.join(t, how="inner")
    else:
        tick = list(dict.fromkeys(FACTORS + [RF, "DBMF", target]))
        px = load_prices(tick, start).rename(columns={target: "TARGET"}) if target != "DBMF" else \
             load_prices(tick, start).assign(TARGET=lambda d: d["DBMF"])
    ret = px.pct_change().dropna(subset=["TARGET"])
    ret = ret.dropna(subset=FACTORS + [RF])
    return ret


def synthetic(n=1800, seed=0):
    """Smoke-test data: target has slowly drifting betas on the factors + noise."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-05-08", periods=n)
    vol = np.array([.012, .011, .014, .0008, .003, .008, .005, .006, .009, .008])
    X = rng.normal(0, 1, (n, len(FACTORS))) * vol
    beta = np.cumsum(rng.normal(0, .02, (n, len(FACTORS))), axis=0) * 0.2 + 1.0
    beta = np.clip(beta * np.sign(rng.normal(size=len(FACTORS))), -3, 3)
    y = (X * beta).sum(1) + rng.normal(0, .003, n)
    df = pd.DataFrame(X, index=idx, columns=FACTORS)
    df[RF] = 0.0001
    df["TARGET"] = y + 0.0001
    df["DBMF"] = df["TARGET"] + rng.normal(0, .002, n)
    return df


# ----------------------------------------------------------------- estimators
def fit(X, y, method="ols", lam=0.5, bound=3.0):
    Xc, yc = X - X.mean(0), y - y.mean()
    n, k = Xc.shape
    if method == "ols":
        return np.linalg.lstsq(Xc, yc, rcond=None)[0]
    if method == "ridge":                       # ridge on standardized factors
        s = Xc.std(0) + 1e-12
        Z = Xc / s
        b = np.linalg.solve(Z.T @ Z / n + lam * np.eye(k), Z.T @ yc / n)
        return b / s
    if method == "bounded":                     # box-constrained least squares
        return lsq_linear(Xc, yc, bounds=(-bound, bound)).x
    raise ValueError(method)


def run_replica(df, window, cfg, step=STEP):
    X = df[FACTORS].sub(df[RF], axis=0).values         # excess factor returns
    y = (df["TARGET"] - df[RF]).values                 # excess target returns
    rf = df[RF].values
    preds, betas = [], []
    for t in range(window, len(df) - 1, step):
        b = fit(X[t - window:t], y[t - window:t], **cfg)   # uses data strictly before t
        nxt = slice(t, min(t + step, len(df)))
        preds.append(pd.Series(X[nxt] @ b + rf[nxt], index=df.index[nxt]))
        betas.append(pd.Series(b, index=FACTORS, name=df.index[t]))
    return pd.concat(preds), pd.DataFrame(betas)


# ----------------------------------------------------------------- metrics
def ann_ret(r):
    return (1 + r).prod() ** (252 / len(r)) - 1


def evaluate(rep, df):
    tgt = df["TARGET"].reindex(rep.index)
    ex_r, ex_t = rep - df[RF].reindex(rep.index), tgt - df[RF].reindex(rep.index)
    # lag diagnostic: corr(replica_t, target_{t-k}); peak at k>0 => replica lags target
    lags = {k: ex_r.corr(ex_t.shift(k)) for k in range(-3, 11)}
    best_lag = max(lags, key=lags.get)
    wk = lambda s: (1 + s).resample("W").prod() - 1
    return dict(
        corr_d=ex_r.corr(ex_t),
        corr_w=wk(rep).corr(wk(tgt)),
        TE=(rep - tgt).std() * np.sqrt(252),
        ret_rep=ann_ret(rep),
        ret_tgt=ann_ret(tgt),
        vol_ratio=rep.std() / tgt.std(),
        best_lag_d=best_lag,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="KMLM")
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--plot", default="replication_results.png")
    a = ap.parse_args()

    df = synthetic() if a.synthetic else load_data(a.target, a.start)
    if df.empty:
        raise SystemExit(
            "No usable observations were loaded. Yahoo Finance may be "
            "rate-limiting the request; retry later or run with --synthetic. "
            "For a CSV target, check that its dates overlap the factor data."
        )
    print(f"Sample: {df.index[0].date()} -> {df.index[-1].date()}  ({len(df)} days), target={a.target}")

    rows, store = [], {}
    for name, cfg in METHODS.items():
        for w in WINDOWS:
            rep, betas = run_replica(df, w, cfg)
            m = evaluate(rep, df)
            rows.append(dict(method=name, window=w, **m))
            store[(name, w)] = (rep, betas)
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 160, "display.float_format", "{:.3f}".format)
    print("\n", res.to_string(index=False))
    print("\nReading it: corr/TE = tracking quality; best_lag_d > 0 = replica lags target by that many days;")
    print("vol_ratio > 1 = replica is over-levered vs target (the vol-targeting asymmetry).")

    # If actual DBMF is present, compare the replica of the same target to it directly
    if "DBMF" in df and a.target.upper() != "DBMF":
        rep, _ = store[("ridge", 60)]
        d = df["DBMF"].reindex(rep.index)
        print(f"\nRidge-60 replica vs actual DBMF: daily corr = {rep.corr(d):.3f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
        for name in METHODS:
            s = res[res.method == name]
            ax[0].plot(s.window, s.TE, marker="o", label=name)
            ax[1].plot(s.window, s.best_lag_d, marker="o", label=name)
        ax[0].set(title="Tracking error vs window", xlabel="window (days)", ylabel="ann. TE")
        ax[1].set(title="Effective lag vs window", xlabel="window (days)", ylabel="days")
        ax[0].legend()
        rep, _ = store[("ridge", 60)]
        (1 + df["TARGET"].reindex(rep.index)).cumprod().plot(ax=ax[2], label="target")
        (1 + rep).cumprod().plot(ax=ax[2], label="ridge-60 replica")
        ax[2].set(title="Cumulative return"); ax[2].legend()
        plt.tight_layout(); plt.savefig(a.plot, dpi=130)
        print(f"Saved {a.plot}")
    except ImportError:
        print("matplotlib not installed; skipping plot")


if __name__ == "__main__":
    main()
