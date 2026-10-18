"""
MARKET RISK TERMINAL
====================
Bloomberg-style Streamlit dashboard that combines:

  * a live market-state monitor (indices, VIX regime, sectors, FX, commodities)
  * your market risk engine (Historical, Parametric, Monte Carlo,
    GARCH-Normal and GARCH-Student-t VaR / Expected Shortfall)
  * rolling backtests (Kupiec, Christoffersen, conditional coverage, Basel traffic light)
  * profit / loss projection (fan chart from simulated portfolio paths)
  * portfolio optimisation and rebalancing suggestions
  * stress tests and rule-based recommendations

Run:   streamlit run app.py
Needs: pip install streamlit yfinance plotly pandas numpy scipy arch
(`arch` is optional; without it the two GARCH models are skipped.)

The engine functions below are the same methods as in your engine scripts
(loss convention: positive number = loss, VaR = loss quantile), rewritten as
reusable functions that take the user's live portfolio instead of CSV files.
"""

import hashlib
import html
import json
import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from scipy.optimize import minimize
from scipy.special import xlogy
from scipy.stats import chi2, kurtosis, norm, skew
from scipy.stats import t as student_t

try:
    from arch import arch_model

    HAS_ARCH = True
except Exception:  # arch not installed
    HAS_ARCH = False


# =========================================================
# PAGE CONFIG AND TERMINAL THEME
# =========================================================

st.set_page_config(
    page_title="Market Risk Terminal",
    page_icon="📟",
    layout="wide",
    initial_sidebar_state="expanded",
)

AMBER = "#ff9f1a"
GREEN = "#00d26a"
RED = "#ff3b3b"
CYAN = "#35c2ff"
GREY = "#8a8a8a"

st.markdown(
    f"""
    <style>
    html, body, [class*="css"], .stApp {{
        font-family: "IBM Plex Mono", Consolas, "Courier New", monospace;
    }}
    .stApp, [data-testid="stAppViewContainer"] {{ background: #000000; color: #e0e0e0; }}
    [data-testid="stHeader"] {{ background: #000000; }}
    [data-testid="stSidebar"] {{ background: #0a0a0a; border-right: 1px solid #2a2a2a; }}
    .block-container {{ padding-top: 1rem; padding-bottom: 2rem; max-width: 100%; }}
    h1, h2, h3, h4 {{ color: {AMBER}; letter-spacing: 0.04em; }}

    div[data-testid="stMetric"] {{
        background: #0b0b0b; border: 1px solid #2a2a2a; border-left: 3px solid {AMBER};
        padding: 10px 12px; border-radius: 2px;
    }}
    div[data-testid="stMetricLabel"] p {{
        color: {GREY}; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em;
    }}
    div[data-testid="stMetricValue"] {{ color: #ffffff; font-size: 1.35rem; }}

    .stTabs [data-baseweb="tab-list"] {{ gap: 2px; border-bottom: 1px solid #2a2a2a; }}
    .stTabs [data-baseweb="tab"] {{
        background: #0b0b0b; color: {GREY}; padding: 8px 16px; border-radius: 0;
        font-size: 0.8rem; letter-spacing: 0.08em;
    }}
    .stTabs [aria-selected="true"] {{ background: {AMBER}; color: #000000 !important; font-weight: 700; }}

    .term-bar {{
        display: flex; justify-content: space-between; align-items: center;
        background: {AMBER}; color: #000; padding: 6px 14px; font-weight: 700;
        letter-spacing: 0.1em; font-size: 0.85rem;
    }}
    .tape {{
        overflow: hidden; white-space: nowrap; background: #050505;
        border-bottom: 1px solid #2a2a2a; border-top: 1px solid #2a2a2a; padding: 6px 0;
    }}
    .tape-inner {{ display: inline-block; animation: scroll 70s linear infinite; }}
    .tape:hover .tape-inner {{ animation-play-state: paused; }}
    @keyframes scroll {{ from {{ transform: translateX(0); }} to {{ transform: translateX(-50%); }} }}
    .tk {{ margin: 0 22px; font-size: 0.82rem; }}
    .tk b {{ color: {AMBER}; }}
    .up {{ color: {GREEN}; }}
    .dn {{ color: {RED}; }}

    .sec {{
        color: {AMBER}; border-bottom: 1px solid #2a2a2a; margin: 18px 0 8px 0;
        padding-bottom: 3px; font-size: 0.8rem; letter-spacing: 0.12em; font-weight: 700;
    }}
    .sig {{ border-left: 4px solid {GREY}; background: #0b0b0b; padding: 8px 12px; margin: 6px 0; font-size: 0.85rem; }}
    .sig.high {{ border-left-color: {RED}; }}
    .sig.med  {{ border-left-color: {AMBER}; }}
    .sig.low  {{ border-left-color: {GREEN}; }}
    .sig.info {{ border-left-color: {CYAN}; }}
    .sig .tag {{ font-weight: 700; margin-right: 8px; }}
    .qt {{ width: 100%; border-collapse: collapse; font-size: 0.82rem; }}
    .qt th {{ color: {GREY}; text-align: right; padding: 4px 8px; border-bottom: 1px solid #2a2a2a; font-weight: 400; }}
    .qt td {{ text-align: right; padding: 4px 8px; border-bottom: 1px solid #141414; }}
    .qt th:first-child, .qt td:first-child {{ text-align: left; color: {AMBER}; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# FORMATTING HELPERS
# =========================================================

def money(value, currency):
    if value is None or not np.isfinite(value):
        return "N/A"
    symbol = "$" if currency == "USD" else "₹"
    sign = "-" if value < 0 else ""
    return f"{sign}{symbol}{abs(value):,.2f}"


def pct(value, digits=2):
    if value is None or not np.isfinite(value):
        return "N/A"
    return f"{value:.{digits}%}"


def section(title):
    st.markdown(f'<div class="sec">▌{html.escape(title)}</div>', unsafe_allow_html=True)


def style_fig(fig, height=360, title=None, hover="x unified"):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#000000",
        plot_bgcolor="#050505",
        font=dict(family="Consolas, monospace", color="#d0d0d0", size=11),
        margin=dict(l=10, r=10, t=40 if title else 15, b=10),
        height=height,
        title=dict(text=title, font=dict(color=AMBER, size=13)) if title else None,
        legend=dict(orientation="h", y=1.1, x=0),
        hovermode=hover,
    )
    fig.update_xaxes(gridcolor="#1a1a1a", zeroline=False)
    fig.update_yaxes(gridcolor="#1a1a1a", zeroline=False)
    return fig


# =========================================================
# MARKET DATA
# =========================================================

MARKET_SYMBOLS = {
    "S&P 500": "^GSPC",
    "NASDAQ": "^IXIC",
    "DOW": "^DJI",
    "VIX": "^VIX",
    "NIFTY 50": "^NSEI",
    "SENSEX": "^BSESN",
    "INDIA VIX": "^INDIAVIX",
    "FTSE 100": "^FTSE",
    "NIKKEI 225": "^N225",
    "GOLD": "GC=F",
    "CRUDE OIL": "CL=F",
    "US 10Y YIELD": "^TNX",
    "DOLLAR INDEX": "DX-Y.NYB",
    "USD/INR": "INR=X",
    "BITCOIN": "BTC-USD",
}

US_SECTORS = {
    "Technology": "XLK", "Financials": "XLF", "Energy": "XLE", "Health Care": "XLV",
    "Discretionary": "XLY", "Staples": "XLP", "Industrials": "XLI", "Utilities": "XLU",
    "Materials": "XLB", "Real Estate": "XLRE", "Comm. Services": "XLC",
}

IN_SECTORS = {
    "Bank": "^NSEBANK", "IT": "^CNXIT", "Auto": "^CNXAUTO", "Pharma": "^CNXPHARMA",
    "FMCG": "^CNXFMCG", "Energy": "^CNXENERGY", "Metal": "^CNXMETAL", "Realty": "^CNXREALTY",
}

BENCHMARK = {"USD": "^GSPC", "INR": "^NSEI"}
VOL_INDEX = {"USD": "^VIX", "INR": "^INDIAVIX"}


@st.cache_data(ttl=900, show_spinner=False)
def download_close(tickers, start=None, end=None, period=None):
    """Download adjusted closing prices. Returns a DataFrame (columns = tickers)."""

    tickers = list(tickers)
    kwargs = dict(tickers=tickers, auto_adjust=True, progress=False, threads=True)

    if period:
        kwargs["period"] = period
    else:
        # Yahoo's end date is exclusive, so add one day.
        kwargs["start"] = start
        kwargs["end"] = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    data = yf.download(**kwargs)

    if data is None or data.empty:
        raise ValueError(
            "No market data was returned. Check the ticker symbols, "
            "date range and internet connection."
        )

    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(0):
            prices = data["Close"].copy()
        elif "Close" in data.columns.get_level_values(-1):
            prices = data.xs("Close", axis=1, level=-1).copy()
        else:
            raise ValueError("Closing prices were not found in the downloaded data.")
    else:
        if "Close" not in data.columns:
            raise ValueError("Closing prices were not found in the downloaded data.")
        prices = data[["Close"]].copy()
        prices.columns = [tickers[0]]

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(tickers[0])

    prices.columns = [str(c).upper() for c in prices.columns]
    prices.index = pd.to_datetime(prices.index)
    if getattr(prices.index, "tz", None) is not None:
        prices.index = prices.index.tz_localize(None)

    return prices.sort_index().dropna(axis=1, how="all")


@st.cache_data(ttl=900, show_spinner=False)
def download_volume(tickers, start=None, end=None):
    """Download daily share volume where the data provider supplies it."""
    tickers = list(tickers)
    kwargs = dict(tickers=tickers, auto_adjust=False, progress=False, threads=True)
    if start is not None and end is not None:
        kwargs["start"] = start
        kwargs["end"] = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        kwargs["period"] = "3mo"
    data = yf.download(**kwargs)
    if data is None or data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        if "Volume" in data.columns.get_level_values(0):
            volume = data["Volume"].copy()
        elif "Volume" in data.columns.get_level_values(-1):
            volume = data.xs("Volume", axis=1, level=-1).copy()
        else:
            return pd.DataFrame()
    else:
        if "Volume" not in data.columns:
            return pd.DataFrame()
        volume = data[["Volume"]].copy()
        volume.columns = [tickers[0]]
    if isinstance(volume, pd.Series):
        volume = volume.to_frame(tickers[0])
    volume.columns = [str(c).upper() for c in volume.columns]
    volume.index = pd.to_datetime(volume.index)
    if getattr(volume.index, "tz", None) is not None:
        volume.index = volume.index.tz_localize(None)
    return volume.sort_index()


def build_quotes(close):
    """Last price and 1D / 5D / 1M / YTD changes for the market watch."""

    rows = []
    for name, ticker in MARKET_SYMBOLS.items():
        if ticker.upper() not in close.columns:
            continue

        s = close[ticker.upper()].dropna()
        if len(s) < 3:
            continue

        def change(n):
            return s.iloc[-1] / s.iloc[-1 - n] - 1 if len(s) > n else np.nan

        prior_year = s[s.index.year < s.index[-1].year]
        base = prior_year.iloc[-1] if len(prior_year) else s.iloc[0]

        rows.append({
            "Name": name,
            "Ticker": ticker.upper(),
            "Last": float(s.iloc[-1]),
            "1D": change(1),
            "5D": change(5),
            "1M": change(21),
            "YTD": float(s.iloc[-1] / base - 1),
        })

    return pd.DataFrame(rows)


def market_regime(close, currency):
    """Classify the current market state from volatility index and benchmark trend."""

    bench = BENCHMARK[currency]
    vix_t = VOL_INDEX[currency]
    out = {"label": "N/A", "color": GREY, "vol_label": "N/A", "trend_label": "N/A",
           "vix": np.nan, "bench_ret_1m": np.nan, "realized_vol": np.nan, "notes": []}

    score = 0  # positive = risk-on, negative = risk-off

    vix_series = None
    for candidate in (vix_t, "^VIX"):
        if candidate in close.columns and close[candidate].dropna().size:
            vix_series = close[candidate].dropna()
            break

    if vix_series is not None:
        vix = float(vix_series.iloc[-1])
        out["vix"] = vix
        if vix < 15:
            out["vol_label"], score = "CALM", score + 1
        elif vix < 20:
            out["vol_label"] = "NORMAL"
        elif vix < 30:
            out["vol_label"], score = "ELEVATED", score - 1
        else:
            out["vol_label"], score = "STRESS", score - 2

    if bench in close.columns:
        s = close[bench].dropna()
        if len(s) >= 50:
            ma50 = s.rolling(50).mean().iloc[-1]
            ma200 = s.rolling(200).mean().iloc[-1] if len(s) >= 200 else np.nan
            above50 = s.iloc[-1] > ma50
            above200 = (s.iloc[-1] > ma200) if np.isfinite(ma200) else None

            score += 1 if above50 else -1
            if above200 is not None:
                score += 1 if above200 else -1

            if above50 and above200 is not False:
                out["trend_label"] = "UPTREND"
            elif (not above50) and above200 is not True:
                out["trend_label"] = "DOWNTREND"
            else:
                out["trend_label"] = "MIXED"

            out["bench_ret_1m"] = float(s.iloc[-1] / s.iloc[-22] - 1) if len(s) > 22 else np.nan
            out["realized_vol"] = float(s.pct_change().tail(20).std() * np.sqrt(252))

    if score >= 2:
        out["label"], out["color"] = "RISK-ON", GREEN
    elif score <= -2:
        out["label"], out["color"] = "RISK-OFF", RED
    elif out["vol_label"] == "N/A" and out["trend_label"] == "N/A":
        pass
    else:
        out["label"], out["color"] = "NEUTRAL", AMBER

    return out


def tape_html(quotes):
    items = []
    for _, row in quotes.iterrows():
        cls = "up" if row["1D"] >= 0 else "dn"
        arrow = "▲" if row["1D"] >= 0 else "▼"
        items.append(
            f'<span class="tk"><b>{html.escape(row["Name"])}</b> {row["Last"]:,.2f} '
            f'<span class="{cls}">{arrow} {row["1D"]:+.2%}</span></span>'
        )
    joined = "".join(items)
    return f'<div class="tape"><div class="tape-inner">{joined}{joined}</div></div>'


def quotes_table_html(quotes):
    def cell(v):
        if not np.isfinite(v):
            return "<td>–</td>"
        cls = "up" if v >= 0 else "dn"
        return f'<td class="{cls}">{v:+.2%}</td>'

    rows = ""
    for _, r in quotes.iterrows():
        rows += (
            f"<tr><td>{html.escape(r['Name'])}</td><td>{r['Last']:,.2f}</td>"
            f"{cell(r['1D'])}{cell(r['5D'])}{cell(r['1M'])}{cell(r['YTD'])}</tr>"
        )
    return (
        '<table class="qt"><tr><th>MARKET</th><th>LAST</th><th>1D</th><th>5D</th>'
        f"<th>1M</th><th>YTD</th></tr>{rows}</table>"
    )


# =========================================================
# RISK ENGINE  (same methods as the engine scripts)
# =========================================================

def hist_var_es(returns, value, conf, horizon=1):
    """Historical simulation VaR / ES. Horizon uses square-root-of-time scaling."""

    losses = -returns.to_numpy(dtype=float) * np.sqrt(horizon) * value
    var = float(np.quantile(losses, conf))
    es = float(losses[losses >= var].mean())
    return var, es


def param_var_es(returns, value, conf, horizon=1):
    """Variance-covariance (normal) VaR / ES."""

    mu = float(returns.mean())
    sigma = float(returns.std(ddof=1))
    z = norm.ppf(conf)
    var = (z * sigma * np.sqrt(horizon) - mu * horizon) * value
    es = (sigma * np.sqrt(horizon) * norm.pdf(z) / (1 - conf) - mu * horizon) * value
    return float(var), float(es)


def mc_var_es(asset_returns, weights, value, conf, horizon, sims, seed=42):
    """Multivariate Monte Carlo VaR / ES (correlated normal asset returns)."""

    mu = asset_returns.mean().to_numpy()
    cov = asset_returns.cov().to_numpy()

    rng = np.random.default_rng(seed)
    simulated = rng.multivariate_normal(
        mean=mu * horizon,
        cov=cov * horizon,
        size=int(sims),
        check_valid="ignore",
    )

    losses = -(simulated @ weights.to_numpy()) * value
    var = float(np.quantile(losses, conf))
    es = float(losses[losses >= var].mean())
    return var, es


@st.cache_data(show_spinner=False)
def garch_var_es(returns, value, conf, horizon, dist):
    """
    Fit GARCH(1,1) (normal or Student-t) on the latest 500 returns and forecast
    horizon-day VaR / ES. Returns a dict, or None when the fit is unavailable.
    """

    if not HAS_ARCH or len(returns) < 150:
        return None

    sample = returns.iloc[-500:] * 100  # percentage units for numerical stability

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = arch_model(
                sample, mean="Constant", vol="GARCH", p=1, q=1, dist=dist, rescale=False
            ).fit(disp="off", show_warning=False)

        if fitted.convergence_flag != 0:
            return None

        forecast = fitted.forecast(horizon=horizon, reindex=False)
        variance_path = forecast.variance.iloc[-1].to_numpy(dtype=float)

        total_variance = float(variance_path.sum())  # h-day variance (% squared)
        if not np.isfinite(total_variance) or total_variance <= 0:
            return None

        sigma_h = np.sqrt(total_variance) / 100
        mu_h = float(fitted.params["mu"]) / 100 * horizon
        alpha = 1 - conf
        nu = None

        if dist == "normal":
            z = norm.ppf(conf)
            var_ret = z * sigma_h - mu_h
            es_ret = sigma_h * norm.pdf(z) / alpha - mu_h
        else:
            nu = float(fitted.params["nu"])
            if nu <= 2:
                return None
            scale = np.sqrt((nu - 2) / nu)  # standardise to unit variance
            q = student_t.ppf(alpha, df=nu)
            var_ret = -q * scale * sigma_h - mu_h
            es_unit = student_t.pdf(q, df=nu) / alpha * (nu + q ** 2) / (nu - 1)
            es_ret = es_unit * scale * sigma_h - mu_h

        cond_vol = pd.Series(
            np.asarray(fitted.conditional_volatility) / 100,
            index=sample.index,
        )

        return {
            "var": float(var_ret * value),
            "es": float(es_ret * value),
            "vol_daily": float(np.sqrt(variance_path[0]) / 100),
            "persistence": float(fitted.params["alpha[1]"] + fitted.params["beta[1]"]),
            "nu": nu,
            "cond_vol": cond_vol,
        }

    except Exception:
        return None


def run_engine(ctx, conf, horizon, sims):
    """Run every engine model on the user's portfolio."""

    r = ctx["port_ret"]
    value = ctx["value"]

    rows = []

    v, e = hist_var_es(r, value, conf, horizon)
    rows.append(("Historical Simulation", v, e))

    v, e = param_var_es(r, value, conf, horizon)
    rows.append(("Parametric (Normal)", v, e))

    v, e = mc_var_es(ctx["asset_returns"], ctx["weights"], value, conf, horizon, sims)
    rows.append(("Monte Carlo (Multivariate)", v, e))

    garch_n = garch_var_es(r, value, conf, horizon, "normal")
    garch_t = garch_var_es(r, value, conf, horizon, "t")

    if garch_n:
        rows.append(("GARCH(1,1) Normal", garch_n["var"], garch_n["es"]))
    if garch_t:
        rows.append(("GARCH(1,1) Student-t", garch_t["var"], garch_t["es"]))

    table = pd.DataFrame(rows, columns=["Model", "VaR", "ES"])
    table["VaR_pct"] = table["VaR"] / value
    table["ES_pct"] = table["ES"] / value

    return table, garch_n, garch_t


# ---------- backtesting ----------

def kupiec_test(exceptions, conf):
    x = np.asarray(exceptions, dtype=int)
    n, failures = len(x), int(x.sum())
    expected = 1 - conf
    observed = failures / n

    ll_null = xlogy(failures, expected) + xlogy(n - failures, 1 - expected)
    ll_alt = xlogy(failures, observed) + xlogy(n - failures, 1 - observed)

    lr = max(0.0, -2 * (ll_null - ll_alt))
    return lr, float(chi2.sf(lr, df=1))


def christoffersen_test(exceptions):
    x = np.asarray(exceptions, dtype=int)
    prev, curr = x[:-1], x[1:]

    n00 = int(((prev == 0) & (curr == 0)).sum())
    n01 = int(((prev == 0) & (curr == 1)).sum())
    n10 = int(((prev == 1) & (curr == 0)).sum())
    n11 = int(((prev == 1) & (curr == 1)).sum())

    total = n00 + n01 + n10 + n11
    ones = n01 + n11
    pooled = ones / total if total else 0.0

    n0, n1 = n00 + n01, n10 + n11
    p01 = n01 / n0 if n0 else 0.0
    p11 = n11 / n1 if n1 else 0.0

    ll_null = xlogy(ones, pooled) + xlogy(total - ones, 1 - pooled)
    ll_alt = (
        xlogy(n01, p01) + xlogy(n00, 1 - p01)
        + xlogy(n11, p11) + xlogy(n10, 1 - p11)
    )

    lr = max(0.0, -2 * (ll_null - ll_alt))
    return lr, float(chi2.sf(lr, df=1))


def choose_lookback(n_obs):
    if n_obs >= 600:
        return 500
    if n_obs >= 300:
        return 250
    if n_obs >= 150:
        return 100
    return None


def rolling_backtests(port_ret, conf, lookback):
    """Out-of-sample rolling 1-day VaR backtests (fractions of portfolio value)."""

    r = port_ret
    loss = -r
    z = norm.ppf(conf)

    hist = loss.rolling(lookback).quantile(conf).shift(1)
    mu = r.rolling(lookback).mean().shift(1)
    sd = r.rolling(lookback).std(ddof=1).shift(1)
    parametric = z * sd - mu
    ewma_var = (r ** 2).ewm(alpha=0.06, adjust=False).mean().shift(1)  # RiskMetrics λ = 0.94
    ewma = z * np.sqrt(ewma_var)

    index = hist.dropna().index

    frames = {}
    for name, var in {
        "Historical Simulation": hist,
        "Parametric (Normal)": parametric,
        "EWMA (RiskMetrics)": ewma,
    }.items():
        df = pd.DataFrame({"VaR": var.reindex(index), "Loss": loss.reindex(index)}).dropna()
        df["Exception"] = (df["Loss"] > df["VaR"]).astype(int)
        frames[name] = df

    return frames


def backtest_summary(frames, conf):
    rows = []
    for name, df in frames.items():
        exc = df["Exception"].to_numpy()
        n = len(exc)
        count = int(exc.sum())

        lr_uc, p_uc = kupiec_test(exc, conf)
        lr_ind, p_ind = christoffersen_test(exc)
        p_cc = float(chi2.sf(lr_uc + lr_ind, df=2))

        last250 = int(exc[-250:].sum())
        zone = "GREEN" if last250 <= 4 else "YELLOW" if last250 <= 9 else "RED"

        rows.append({
            "Model": name,
            "Obs": n,
            "Exceptions": count,
            "Rate": count / n,
            "Expected": 1 - conf,
            "Kupiec p": p_uc,
            "Independence p": p_ind,
            "Cond. Coverage p": p_cc,
            "Last 250d exc.": last250,
            "Basel zone": zone,
        })
    return pd.DataFrame(rows)


# ---------- optimisation ----------

def optimize_weights(mu, cov, rf, max_weight, objective):
    n = len(mu)
    max_weight = max(max_weight, 1 / n + 1e-9)
    w0 = np.full(n, 1 / n)

    if objective == "min_vol":
        def fun(w):
            return float(w @ cov @ w)
    elif objective == "max_sharpe":
        def fun(w):
            return -float((w @ mu - rf) / np.sqrt(w @ cov @ w + 1e-16))
    else:  # risk parity
        def fun(w):
            vol = np.sqrt(w @ cov @ w + 1e-16)
            contrib = w * (cov @ w) / vol
            return float(np.sum((contrib - vol / n) ** 2) * 1e4)

    result = minimize(
        fun,
        w0,
        method="SLSQP",
        bounds=[(0.0, max_weight)] * n,
        constraints=({"type": "eq", "fun": lambda w: w.sum() - 1},),
        options={"maxiter": 500, "ftol": 1e-12},
    )

    w = np.clip(result.x if result.success else w0, 0, None)
    return w / w.sum()


def weight_stats(w, mu, cov, rf, asset_returns, conf):
    ret = float(w @ mu)
    vol = float(np.sqrt(w @ cov @ w))
    port = asset_returns.to_numpy() @ w
    var1 = float(np.quantile(-port, conf))
    return {
        "Ann. return": ret,
        "Ann. volatility": vol,
        "Sharpe": (ret - rf) / vol if vol > 0 else np.nan,
        f"1D VaR": var1,
    }


# =========================================================
# PORTFOLIO CONTEXT
# =========================================================

def validate_holdings(holdings_input, currency):
    """Return (clean DataFrame or None, error message, notes)."""

    notes = []

    df = holdings_input.copy()
    df["Ticker"] = df["Ticker"].fillna("").astype(str).str.strip().str.upper()
    df = df[df["Ticker"] != ""].copy()

    if df.empty:
        return None, "Add at least one investment to continue.", notes

    df["Shares"] = pd.to_numeric(df["Shares"], errors="coerce")
    df["Purchase Price"] = pd.to_numeric(df["Purchase Price"], errors="coerce")

    # Rows that are still being typed (no valid share count yet) are skipped
    # instead of blocking the whole dashboard.
    incomplete = df["Shares"].isna() | (df["Shares"] <= 0)
    if incomplete.any():
        notes.append(
            "Skipped until you enter a share count above 0: "
            + ", ".join(df.loc[incomplete, "Ticker"])
        )
        df = df[~incomplete].copy()

    if df.empty:
        return None, "Enter a share count for at least one holding.", notes

    if df["Ticker"].duplicated().any():
        return None, (
            "Each ticker should appear only once. Combine duplicates and enter "
            "the average purchase price."
        ), notes

    indian = [t.endswith(".NS") or t.endswith(".BO") for t in df["Ticker"]]

    if any(indian) and not all(indian):
        return None, "Mixed Indian and US-listed portfolios are not supported. Use one market.", notes

    if currency == "INR" and not all(indian):
        return None, "INR mode needs Indian-listed tickers such as RELIANCE.NS or TCS.NS.", notes

    if currency == "USD" and any(indian):
        return None, "Indian-listed securities are quoted in INR. Select INR mode.", notes

    return df.reset_index(drop=True), "", notes


def calculate_holdings(holdings, prices):
    latest = prices.iloc[-1]
    previous = prices.iloc[-2] if len(prices) >= 2 else latest

    rows = []
    for _, h in holdings.iterrows():
        ticker, shares, buy = h["Ticker"], float(h["Shares"]), float(h["Purchase Price"])
        cur, prev = float(latest[ticker]), float(previous[ticker])

        invested = shares * buy
        value = shares * cur

        rows.append({
            "Ticker": ticker,
            "Shares": shares,
            "Purchase Price": buy,
            "Current Price": cur,
            "Invested Amount": invested,
            "Current Value": value,
            "Total Profit/Loss": value - invested,
            "Daily Profit/Loss": shares * (cur - prev),
            "Return on Cost": (value - invested) / invested if invested > 0 else np.nan,
            "Daily Price Change": cur / prev - 1 if prev > 0 else np.nan,
        })

    return pd.DataFrame(rows)


def max_drawdown(values):
    peak = values.cummax()
    drawdown = values / peak - 1
    return float(drawdown.min()), drawdown


def build_context(valid_holdings, currency, start_date, end_date):
    notes = []
    requested = valid_holdings["Ticker"].tolist()

    raw = download_close(tuple(requested), start=str(start_date), end=str(end_date))

    # Unknown / delisted symbols are excluded instead of failing the whole portfolio.
    skipped = [t for t in requested if t not in raw.columns]
    if skipped:
        notes.append(
            "No market data found (check the symbol), so these were excluded: " + ", ".join(skipped)
        )

    tickers = [t for t in requested if t in raw.columns]
    if not tickers:
        raise ValueError("Market data could not be retrieved for: " + ", ".join(requested))

    holdings = valid_holdings[valid_holdings["Ticker"].isin(tickers)].reset_index(drop=True)

    raw = raw[tickers]
    prices = raw.dropna(how="any")

    if len(prices) < 60:
        raise ValueError(
            "Fewer than 60 complete price observations. Choose a longer date range "
            "or check the ticker symbols."
        )

    if len(prices) < 0.9 * len(raw.dropna(how="all")):
        shortest = raw.apply(lambda col: col.first_valid_index()).idxmax()
        notes.append(
            f"History is limited to {len(prices)} trading days because {shortest} "
            "has a shorter price history than the other holdings."
        )

    # Missing purchase price -> use the latest price (profit/loss starts at zero).
    latest = prices.iloc[-1]
    fill = holdings["Purchase Price"].isna() | (holdings["Purchase Price"] <= 0)
    if fill.any():
        holdings.loc[fill, "Purchase Price"] = holdings.loc[fill, "Ticker"].map(latest)
        notes.append(
            "No purchase price entered for " + ", ".join(holdings.loc[fill, "Ticker"])
            + ": the latest price was used, so their profit/loss starts at 0."
        )

    summary = calculate_holdings(holdings, prices)
    total_value = float(summary["Current Value"].sum())
    invested = float(summary["Invested Amount"].sum())

    weights = pd.Series(
        (summary["Current Value"] / total_value).to_numpy(),
        index=tickers,
        name="weight",
    )

    asset_returns = prices.pct_change(fill_method=None).dropna()
    port_ret = (asset_returns @ weights).rename("portfolio_return")

    # Approximate liquidity using 20-session average daily dollar volume.
    # This is a proxy, not an executable quote or a guarantee of market depth.
    avg_dollar_volume = pd.Series(dtype=float)
    try:
        volume = download_volume(tuple(tickers), start=str(start_date), end=str(end_date))
        if not volume.empty:
            common = [t for t in tickers if t in volume.columns and t in prices.columns]
            if common:
                avg_dollar_volume = volume[common].reindex(prices.index).mul(prices[common]).tail(20).mean()
    except Exception:
        avg_dollar_volume = pd.Series(dtype=float)

    quantities = holdings.set_index("Ticker")["Shares"]
    historical_values = prices.mul(quantities, axis=1).sum(axis=1)
    mdd, drawdown = max_drawdown(historical_values)

    # Beta versus the local benchmark.
    bench_ret, betas, port_beta = None, None, np.nan
    try:
        bench_prices = download_close(
            (BENCHMARK[currency],), start=str(start_date), end=str(end_date)
        )
        bench_ret = bench_prices.iloc[:, 0].pct_change(fill_method=None).dropna()
        aligned = asset_returns.join(bench_ret.rename("BENCH"), how="inner")

        if len(aligned) >= 60:
            var_b = aligned["BENCH"].var()
            betas = pd.Series(
                {t: aligned[t].cov(aligned["BENCH"]) / var_b for t in tickers}
            )
            port_beta = float((betas * weights).sum())
    except Exception:
        pass

    return {
        "currency": currency,
        "tickers": tickers,
        "notes": notes,
        "holdings": holdings,
        "prices": prices,
        "summary": summary,
        "value": total_value,
        "invested": invested,
        "pnl": total_value - invested,
        "daily_pnl": float(summary["Daily Profit/Loss"].sum()),
        "weights": weights,
        "asset_returns": asset_returns,
        "avg_dollar_volume": avg_dollar_volume,
        "port_ret": port_ret,
        "hist_values": historical_values,
        "max_dd": mdd,
        "drawdown": drawdown,
        "bench_ret": bench_ret,
        "betas": betas,
        "port_beta": port_beta,
        "ann_vol": float(port_ret.std(ddof=1) * np.sqrt(252)),
        "ann_ret": float(port_ret.mean() * 252),
    }


def risk_contributions(ctx):
    """Share of portfolio variance contributed by each asset."""

    w = ctx["weights"].to_numpy()
    cov = ctx["asset_returns"].cov().to_numpy()
    port_var = float(w @ cov @ w)
    if port_var <= 0:
        return pd.Series(w, index=ctx["tickers"])
    return pd.Series(w * (cov @ w) / port_var, index=ctx["tickers"])


def risk_score(ctx):
    """0-100 composite risk score (volatility, tail loss, concentration, drawdown)."""

    r = ctx["port_ret"]
    var1 = max(
        hist_var_es(r, 1.0, 0.99)[0],
        param_var_es(r, 1.0, 0.99)[0],
    )
    hhi = float((ctx["weights"] ** 2).sum())

    parts = {
        "Volatility": float(np.clip(ctx["ann_vol"] / 0.40, 0, 1) * 35),
        "Tail loss (1D 99% VaR)": float(np.clip(var1 / 0.05, 0, 1) * 25),
        "Concentration": float(np.clip(hhi, 0, 1) * 20),
        "Drawdown": float(np.clip(-ctx["max_dd"] / 0.5, 0, 1) * 20),
    }
    total = sum(parts.values())

    label = "LOW" if total < 25 else "MODERATE" if total < 50 else "ELEVATED" if total < 75 else "HIGH"
    return total, label, parts, var1


# =========================================================
# PROJECTION AND STRESS
# =========================================================

def simulate_paths(port_ret, value, horizon, paths, method, use_drift, seed=7):
    rng = np.random.default_rng(seed)
    r = port_ret.to_numpy(dtype=float)
    mu = float(r.mean()) if use_drift else 0.0
    sd = float(r.std(ddof=1))

    if method == "Bootstrap (historical)":
        sample = r if use_drift else r - r.mean()
        sims = rng.choice(sample, size=(paths, horizon), replace=True)
    elif method == "Student-t (fat tails)":
        nu = float(np.clip(student_t.fit(r)[0], 2.5, 30))
        sims = mu + sd * rng.standard_t(nu, size=(paths, horizon)) * np.sqrt((nu - 2) / nu)
    else:
        sims = rng.normal(mu, sd, size=(paths, horizon))

    sims = np.clip(sims, -0.99, None)
    growth = np.cumprod(1 + sims, axis=1)
    return np.concatenate([np.full((paths, 1), value), value * growth], axis=1)


# =========================================================
# RECOMMENDATIONS
# =========================================================

def build_signals(ctx, engine, backtest, regime, optimizer, score, projection):
    sigs = []

    def add(level, title, text):
        sigs.append((level, title, text))

    cur = ctx["currency"]
    weights = ctx["weights"]
    n = len(weights)
    value = ctx["value"]

    # Concentration
    top, top_w = weights.idxmax(), float(weights.max())
    if n == 1:
        add("high", "SINGLE-ASSET PORTFOLIO",
            "All of your capital is in one security. Spreading across several uncorrelated assets is the most effective risk reducer.")
    elif top_w >= 0.40:
        add("high", "HIGH CONCENTRATION",
            f"{top} is {top_w:.1%} of the portfolio. A large drop in this one holding would dominate your result.")
    elif top_w >= 0.25:
        add("med", "REVIEW LARGEST POSITION",
            f"{top} is {top_w:.1%} of the portfolio. Check that this exposure matches your loss tolerance.")
    else:
        add("low", "POSITION SIZING",
            "No single holding is 25% or more of the portfolio.")

    if n < 5:
        add("med", "FEW HOLDINGS",
            f"Only {n} holding(s). Diversification benefits usually improve noticeably with more, less-correlated assets.")

    # Risk contribution
    rc = risk_contributions(ctx)
    if n >= 2:
        worst = rc.idxmax()
        if rc.max() >= 0.5 and rc.max() > 1.3 * weights[worst]:
            add("med", "RISK CONCENTRATION",
                f"{worst} is {weights[worst]:.1%} of capital but contributes {rc.max():.1%} of portfolio variance.")

    # VaR level
    var_cons = engine["VaR_pct"].max()
    if var_cons >= 0.04:
        add("high", "HIGH TAIL RISK",
            f"Worst-case model VaR is {var_cons:.2%} of value ({money(engine['VaR'].max(), cur)}). A loss of that size is expected once every ~100 days.")
    elif var_cons >= 0.025:
        add("med", "MODERATE TAIL RISK",
            f"Worst-case model VaR is {var_cons:.2%} of value ({money(engine['VaR'].max(), cur)}).")

    # Model disagreement / fat tails
    spread = engine["VaR"].max() / max(engine["VaR"].min(), 1e-9)
    kurt = float(kurtosis(ctx["port_ret"]))
    if kurt > 3 or spread > 1.3:
        add("med", "FAT TAILS DETECTED",
            f"Excess kurtosis is {kurt:.1f} and VaR estimates differ by {spread:.2f}x across models. "
            "Normal-distribution VaR likely understates extreme losses; prefer the Student-t GARCH, Historical or Expected Shortfall figures.")

    # Backtest
    if backtest is not None:
        par = backtest[backtest["Model"] == "Parametric (Normal)"]
        if len(par) and par["Kupiec p"].iloc[0] < 0.05:
            add("high", "NORMAL VaR FAILS BACKTEST",
                "Kupiec test rejects the normal-VaR exception rate at 5%. Treat parametric numbers as optimistic.")
        if (backtest["Independence p"] < 0.05).any():
            names = ", ".join(backtest.loc[backtest["Independence p"] < 0.05, "Model"])
            add("med", "EXCEPTION CLUSTERING",
                f"Christoffersen test finds clustered VaR breaches for: {names}. Losses tend to arrive in streaks.")
        red = backtest[backtest["Basel zone"] == "RED"]
        if len(red):
            add("high", "BASEL RED ZONE",
                "At least one model recorded 10+ exceptions in the last 250 days.")

    # Market regime
    if regime["label"] == "RISK-OFF":
        add("high", "MARKET RISK-OFF",
            f"Volatility index is {regime['vix']:.1f} ({regime['vol_label']}) and the benchmark trend is {regime['trend_label']}. "
            "Consider smaller position sizes or hedges until conditions stabilise.")
    elif regime["label"] == "RISK-ON" and regime["vol_label"] == "CALM":
        add("info", "CALM MARKET",
            "Volatility is low. Historical VaR is calibrated to quiet conditions, so it may understate risk if volatility jumps.")
    elif regime["label"] == "NEUTRAL":
        add("info", "NEUTRAL MARKET", "No strong market-wide signal either way.")

    # Beta
    pb = ctx["port_beta"]
    if np.isfinite(pb):
        if pb > 1.2:
            add("med", "HIGH MARKET SENSITIVITY",
                f"Portfolio beta is {pb:.2f}: expect roughly {pb:.1f}x the benchmark's move in either direction.")
        elif pb < 0.6:
            add("info", "LOW MARKET SENSITIVITY",
                f"Portfolio beta is {pb:.2f}: it moves much less than the benchmark.")

    # Correlation
    if n >= 2:
        corr = ctx["asset_returns"].corr().to_numpy()
        avg_corr = float(corr[np.triu_indices(n, 1)].mean())
        if avg_corr >= 0.7:
            add("med", "HIGH CORRELATION",
                f"Average pairwise correlation is {avg_corr:.2f}. Holdings tend to fall together, so diversification is weaker than the number of positions suggests.")

    # Drawdown
    cur_dd = float(ctx["drawdown"].iloc[-1])
    if cur_dd <= -0.15:
        add("med", "IN DRAWDOWN",
            f"Portfolio value is {cur_dd:.1%} below its previous peak (max drawdown in sample {ctx['max_dd']:.1%}).")

    # Optimizer
    if optimizer is not None:
        cur_vol = optimizer["current"]["Ann. volatility"]
        mv_vol = optimizer["min_vol"]["Ann. volatility"]
        if cur_vol > 0 and mv_vol <= 0.85 * cur_vol:
            add("med", "VOLATILITY CAN BE REDUCED",
                f"A minimum-volatility mix cuts annual volatility from {cur_vol:.1%} to {mv_vol:.1%}. See the OPTIMIZER tab for the trades.")

    # Projection
    if projection is not None:
        if projection["prob_loss"] > 0.45:
            add("info", "PROJECTION",
                f"Simulated probability of a loss over {projection['horizon']} trading days is {projection['prob_loss']:.0%}.")

    # Losers
    losers = ctx["summary"][ctx["summary"]["Total Profit/Loss"] < 0]
    if not losers.empty:
        w = losers.sort_values("Total Profit/Loss").iloc[0]
        add("info", "LARGEST UNREALIZED LOSS",
            f"{w['Ticker']} is {money(w['Total Profit/Loss'], cur)} ({pct(w['Return on Cost'])}) versus cost. "
            "Decide on current outlook, not on the purchase price.")

    order = {"high": 0, "med": 1, "info": 2, "low": 3}
    sigs.sort(key=lambda s: order[s[0]])
    return sigs


# =========================================================
# CHANGE TRACKING AND AI ANALYST
# =========================================================

def portfolio_snapshot(ctx, engine, score, conf, horizon):
    """Compact record of the portfolio and its headline risk metrics."""

    n = len(ctx["tickers"])
    rc = risk_contributions(ctx)

    avg_corr = np.nan
    if n >= 2:
        corr = ctx["asset_returns"].corr().to_numpy()
        avg_corr = float(corr[np.triu_indices(n, 1)].mean())

    worst = engine.loc[engine["VaR"].idxmax()]

    return {
        "currency": ctx["currency"],
        "conf": conf,
        "horizon": horizon,
        "tickers": list(ctx["tickers"]),
        "shares": dict(zip(ctx["holdings"]["Ticker"], ctx["holdings"]["Shares"].astype(float))),
        "weights": {k: float(v) for k, v in ctx["weights"].items()},
        "metrics": {
            "VaR": float(worst["VaR"]),
            "VaR_pct": float(worst["VaR_pct"]),
            "ES": float(engine["ES"].max()),
            "ES_pct": float(engine["ES_pct"].max()),
            "ann_vol": float(ctx["ann_vol"]),
            "beta": float(ctx["port_beta"]) if np.isfinite(ctx["port_beta"]) else np.nan,
            "score": float(score),
            "max_weight": float(ctx["weights"].max()),
            "top_risk_share": float(rc.max()),
            "avg_corr": avg_corr,
            "max_dd": float(ctx["max_dd"]),
            "value": float(ctx["value"]),
        },
    }


def diff_snapshots(base, current):
    added = [t for t in current["shares"] if t not in base["shares"]]
    removed = [t for t in base["shares"] if t not in current["shares"]]
    resized = [
        {"ticker": t, "shares_before": base["shares"][t], "shares_after": current["shares"][t]}
        for t in current["shares"]
        if t in base["shares"] and abs(current["shares"][t] - base["shares"][t]) > 1e-9
    ]
    return {
        "added": added,
        "removed": removed,
        "resized": resized,
        "changed": bool(added or removed or resized),
    }


def ticker_impact(ctx, ticker, conf, horizon):
    """What one holding does to portfolio risk: stand-alone stats plus VaR with vs without it."""

    if ticker not in ctx["tickers"]:
        return None

    w = ctx["weights"]
    ar = ctx["asset_returns"]
    rc = risk_contributions(ctx)

    def conservative_var(r):
        return max(hist_var_es(r, 1.0, conf, horizon)[0], param_var_es(r, 1.0, conf, horizon)[0])

    out = {
        "Ticker": ticker,
        "Weight": float(w[ticker]),
        "Ann. vol": float(ar[ticker].std(ddof=1) * np.sqrt(252)),
        "Beta": float(ctx["betas"][ticker]) if ctx["betas"] is not None else np.nan,
        "Risk share": float(rc[ticker]),
        "Corr. with rest": np.nan,
        "VaR % with": conservative_var(ctx["port_ret"]),
        "VaR % without": np.nan,
        "VaR % change": np.nan,
    }

    if len(ctx["tickers"]) >= 2 and w[ticker] < 0.999:
        rest = w.drop(ticker)
        rest = rest / rest.sum()
        rest_ret = ar[rest.index] @ rest
        out["Corr. with rest"] = float(np.corrcoef(ar[ticker], rest_ret)[0, 1])
        out["VaR % without"] = conservative_var(rest_ret)
        out["VaR % change"] = out["VaR % with"] - out["VaR % without"]

    return out


def _clean(obj, digits=4):
    """Round numbers and drop NaN so the payload is compact valid JSON."""

    if isinstance(obj, dict):
        return {str(k): _clean(v, digits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v, digits) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return round(float(obj), digits) if np.isfinite(obj) else None
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def build_ai_payload(ctx, engine, backtest_df, regime, optimizer, score, score_label,
                     base, current, changes, impacts, signals, conf, horizon):
    rc = risk_contributions(ctx)
    ann_vol = ctx["asset_returns"].std(ddof=1) * np.sqrt(252)

    holdings = []
    for _, row in ctx["summary"].iterrows():
        t = row["Ticker"]
        holdings.append({
            "ticker": t,
            "weight": ctx["weights"][t],
            "return_on_cost": row["Return on Cost"],
            "beta": ctx["betas"][t] if ctx["betas"] is not None else None,
            "annual_volatility": ann_vol[t],
            "share_of_portfolio_variance": rc[t],
        })

    payload = {
        "note_on_units": "All ratios are decimal fractions (0.02 = 2%). Losses are positive numbers.",
        "currency": ctx["currency"],
        "var_confidence": conf,
        "var_horizon_days": horizon,
        "market_state": {
            "regime": regime["label"],
            "volatility_index": regime["vix"],
            "volatility_label": regime["vol_label"],
            "benchmark_trend": regime["trend_label"],
        },
        "portfolio": {
            "market_value": ctx["value"],
            "unrealized_pnl": ctx["pnl"],
            "number_of_holdings": len(ctx["tickers"]),
            "annual_volatility": ctx["ann_vol"],
            "beta_vs_benchmark": ctx["port_beta"],
            "max_drawdown_in_sample": ctx["max_dd"],
            "risk_score_0_to_100": score,
            "risk_score_label": score_label,
        },
        "holdings": holdings,
        "var_es_by_model": [
            {"model": r.Model, "VaR_amount": r.VaR, "ES_amount": r.ES, "VaR_pct": r.VaR_pct, "ES_pct": r.ES_pct}
            for r in engine.itertuples()
        ],
        "rule_based_signals": [{"level": lv, "title": ti, "detail": tx} for lv, ti, tx in signals[:8]],
    }

    if backtest_df is not None:
        payload["backtest"] = [
            {"model": r["Model"], "exception_rate": r["Rate"], "expected_rate": r["Expected"],
             "kupiec_p": r["Kupiec p"], "independence_p": r["Independence p"], "basel_zone": r["Basel zone"]}
            for _, r in backtest_df.iterrows()
        ]

    if optimizer is not None:
        payload["optimizer_targets"] = {
            key: {
                "weights": dict(zip(ctx["tickers"], optimizer[key + "_w"])),
                "annual_volatility": optimizer[key]["Ann. volatility"],
            }
            for key in ("min_vol", "max_sharpe", "risk_parity")
        }

    if changes["changed"]:
        payload["changes_since_baseline"] = {
            "added": changes["added"],
            "removed": changes["removed"],
            "resized": changes["resized"],
            "metrics_before": base["metrics"],
            "metrics_after": current["metrics"],
            "per_holding_impact": impacts,
        }
    else:
        payload["changes_since_baseline"] = None

    return _clean(payload)


AI_SYSTEM_PROMPT = (
    "You are a senior market-risk analyst inside a portfolio risk dashboard. "
    "Use ONLY the numbers in the JSON you are given. Do not invent prices, news, earnings, ratings or "
    "forecasts about any company. Write for a retail investor in plain language. "
    "This is educational analysis, not investment advice.\n\n"
    "Reply in markdown, at most about 350 words, with exactly these sections:\n"
    "**Verdict** - one or two sentences on the overall risk level.\n"
    "**What changed** - only if changes_since_baseline is not null: explain how the added/removed/resized "
    "holdings moved VaR, Expected Shortfall, volatility, beta, concentration and correlation, quoting the "
    "before and after numbers in currency and percent, and say whether the change is good or bad for risk.\n"
    "**Holding check** - for each new or changed holding (otherwise the largest two holdings): weight, "
    "volatility, beta, correlation with the rest of the portfolio, contribution to risk.\n"
    "**Recommended actions** - 2 to 4 concrete, quantitative actions (for example target weights or a "
    "position to trim), consistent with optimizer_targets when present.\n"
    "**Caveats** - model limits that matter here (fat tails, failed backtests, short history, regime).\n"
    "VaR is the loss threshold exceeded only (1 - confidence) of the time; Expected Shortfall is the average "
    "loss beyond it. If the data conflicts (for example a failed backtest), say so."
)


AI_EXPLAIN_PROMPT = (
    "You are a quantitative-finance analyst explaining a dashboard to a retail investor. "
    "Use only the supplied JSON facts; do not invent prices, news, forecasts, or guarantees. "
    "Explain what the metrics mean, the main risk drivers, what the user should investigate next, "
    "and any model limitations. Keep the answer concise and plain-English. Treat recommendations as "
    "educational decision support, not personalized financial advice. If data is missing, say so."
)

def ai_explanation_widget(section_name, facts, widget_key):
    """Opt-in AI explanation for an individual dashboard section."""
    with st.expander(f"🤖 AI explanation — {section_name}"):
        st.caption("Uses server-side credentials and only the metrics supplied for this section.")
        cache_key = f"ai_section_{widget_key}"
        if not ai_key:
            st.info(
                "AI is not configured. Add LLM credentials to `.streamlit/secrets.toml` or server environment variables; "
                "no API key is requested or stored in this dashboard."
            )
            return
        if st.button(f"Explain {section_name}", key=f"explain_btn_{widget_key}"):
            try:
                with st.spinner(f"Explaining {section_name.lower()}..."):
                    cleaned = _clean(facts)
                    answer = call_llm(
                        ai_provider, ai_key, ai_model, ai_base_url, AI_EXPLAIN_PROMPT,
                        f"Explain this {section_name} dashboard using only these facts:\n\n"
                        + json.dumps(cleaned, allow_nan=False, indent=2),
                    )
                    st.session_state[cache_key] = answer
            except Exception as exc:
                st.error(f"AI explanation failed: {exc}")
        if cache_key in st.session_state:
            st.markdown(st.session_state[cache_key])


def call_llm(provider, api_key, model, base_url, system_prompt, user_prompt):
    """Call Anthropic or any OpenAI-compatible chat endpoint (Hugging Face, OpenAI, Groq, ...)."""

    if provider.startswith("Anthropic"):
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": 1200,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_prompt}],
            },
            timeout=90,
        )
        if response.status_code >= 400:
            raise RuntimeError(f"{response.status_code}: {response.text[:300]}")
        blocks = response.json().get("content", [])
        return "".join(b.get("text", "") for b in blocks if b.get("type") == "text").strip()

    response = requests.post(
        base_url.rstrip("/") + "/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "max_tokens": 1200,
            "temperature": 0.3,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        },
        timeout=90,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"{response.status_code}: {response.text[:300]}")
    return response.json()["choices"][0]["message"]["content"].strip()


def get_secret(*names):
    for name in names:
        try:
            value = st.secrets.get(name)
            if value:
                return str(value)
        except Exception:
            pass
        value = os.environ.get(name, "")
        if value:
            return value
    return ""


# Backend-only AI configuration (never expose keys in Streamlit widgets):
# .streamlit/secrets.toml example:
# LLM_PROVIDER = "OpenAI-compatible"
# LLM_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
# LLM_BASE_URL = "https://router.huggingface.co/v1"
# LLM_API_KEY = "your-server-side-token"
# Or set LLM_PROVIDER, LLM_MODEL, LLM_BASE_URL and LLM_API_KEY as server environment variables.
# For Anthropic: set LLM_PROVIDER = "Anthropic" and ANTHROPIC_API_KEY.

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown(f"<h3>⚙ CONTROLS</h3>", unsafe_allow_html=True)

currency = st.sidebar.selectbox(
    "Portfolio currency / market",
    ["USD", "INR"],
    help="USD: US-listed (AAPL, MSFT, SPY). INR: Indian-listed (RELIANCE.NS, TCS.NS). No FX conversion.",
)

today = pd.Timestamp.today().normalize()

start_date = st.sidebar.date_input(
    "History start",
    value=(today - pd.DateOffset(years=5)).date(),
    max_value=today.date(),
)

end_date = st.sidebar.date_input(
    "History end",
    value=today.date(),
    min_value=start_date,
    max_value=today.date(),
)

confidence = st.sidebar.select_slider(
    "VaR confidence", options=[0.90, 0.95, 0.975, 0.99], value=0.99,
    format_func=lambda v: f"{v:.1%}",
)

horizon = st.sidebar.select_slider(
    "VaR horizon (trading days)", options=[1, 2, 5, 10, 21], value=1
)

simulations = st.sidebar.select_slider(
    "Monte Carlo simulations", options=[10_000, 50_000, 100_000, 250_000], value=100_000,
    format_func=lambda v: f"{v:,}",
)

rf_rate = st.sidebar.number_input(
    "Risk-free rate (annual, %)", min_value=0.0, max_value=25.0,
    value=4.0 if currency == "USD" else 6.5, step=0.25,
    help="Used for Sharpe ratios and the max-Sharpe optimizer. Enter the current rate yourself.",
) / 100

max_weight_cap = st.sidebar.slider(
    "Max weight per asset (optimizer)", 0.10, 1.0, 0.40, 0.05,
    format="%.2f",
)

risk_tolerance = st.sidebar.selectbox(
    "Investor risk tolerance",
    ["Conservative", "Moderate", "Aggressive"],
    index=1,
    help="Used to frame portfolio-allocation comparisons; it does not predict future returns.",
)
transaction_cost_bps = st.sidebar.number_input(
    "Estimated transaction cost (bps per traded value)", min_value=0.0, max_value=200.0, value=5.0, step=1.0,
    help="User-supplied estimate; 1 basis point = 0.01%.",
)
slippage_bps = st.sidebar.number_input(
    "Estimated slippage (bps per traded value)", min_value=0.0, max_value=500.0, value=5.0, step=1.0,
    help="Simplified execution-cost assumption, not a live market-impact model.",
)
liquidity_participation_pct = st.sidebar.slider(
    "Assumed max participation in daily volume (%)", 1, 20, 5, 1,
    help="Used to estimate how many sessions a position might take to liquidate based on historical volume.",
)

if st.sidebar.button("↻ Refresh market data"):
    st.cache_data.clear()
    st.rerun()

# AI credentials are read only on the server, never requested in the browser UI.
ai_provider = get_secret("LLM_PROVIDER") or "OpenAI-compatible"
ai_model = get_secret("LLM_MODEL") or "meta-llama/Llama-3.3-70B-Instruct"
ai_base_url = get_secret("LLM_BASE_URL") or "https://router.huggingface.co/v1"
if ai_provider.lower().startswith("anthropic"):
    ai_key = get_secret("ANTHROPIC_API_KEY")
else:
    ai_key = get_secret("LLM_API_KEY", "HF_TOKEN", "OPENAI_API_KEY")

if not HAS_ARCH:
    st.sidebar.warning("`arch` is not installed, so GARCH models are skipped. Run: pip install arch")

if start_date >= end_date:
    st.sidebar.error("Start date must be earlier than end date.")
    st.stop()


# =========================================================
# HEADER + MARKET DATA
# =========================================================

market_error = None
quotes = pd.DataFrame()
market_close = pd.DataFrame()
regime = {"label": "N/A", "color": GREY, "vol_label": "N/A", "trend_label": "N/A",
          "vix": np.nan, "bench_ret_1m": np.nan, "realized_vol": np.nan, "notes": []}

try:
    market_close = download_close(tuple(v.upper() for v in MARKET_SYMBOLS.values()), period="1y")
    quotes = build_quotes(market_close)
    regime = market_regime(market_close, currency)
except Exception as exc:
    market_error = str(exc)

st.markdown(
    f'<div class="term-bar"><span>📟 MARKET RISK TERMINAL</span>'
    f'<span>{currency} MODE &nbsp;|&nbsp; {datetime.now().strftime("%d %b %Y %H:%M")} LOCAL</span></div>',
    unsafe_allow_html=True,
)

if not quotes.empty:
    st.markdown(tape_html(quotes), unsafe_allow_html=True)

tab_market, tab_port, tab_risk, tab_proj, tab_opt, tab_stress, tab_bt, tab_sig, tab_ai = st.tabs(
    ["MARKET", "PORTFOLIO", "RISK ENGINE", "P&L PROJECTION", "OPTIMIZER",
     "STRESS TEST", "BACKTEST", "SIGNALS", "AI ANALYST"]
)


# =========================================================
# TAB: MARKET
# =========================================================

with tab_market:
    if market_error:
        st.error(f"Market data unavailable: {market_error}")
    elif quotes.empty:
        st.info("No market quotes were returned.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Market state", regime["label"])
        c2.metric(
            "Volatility index",
            f"{regime['vix']:.2f}" if np.isfinite(regime["vix"]) else "N/A",
            delta=regime["vol_label"], delta_color="off",
        )
        c3.metric(
            f"Benchmark trend ({'S&P 500' if currency == 'USD' else 'NIFTY 50'})",
            regime["trend_label"],
            delta=pct(regime["bench_ret_1m"]) + " 1M" if np.isfinite(regime["bench_ret_1m"]) else None,
        )
        up = int((quotes["1D"] > 0).sum())
        c4.metric("Breadth (watchlist up today)", f"{up}/{len(quotes)}")

        left, right = st.columns([1, 1])

        with left:
            section("MARKET WATCH")
            st.markdown(quotes_table_html(quotes), unsafe_allow_html=True)

        with right:
            section("PERFORMANCE — REBASED TO 100 (1Y)")
            default_names = [n for n in (["S&P 500", "NASDAQ", "NIFTY 50", "GOLD"]) if n in quotes["Name"].values]
            chosen = st.multiselect(
                "Instruments", quotes["Name"].tolist(), default=default_names, label_visibility="collapsed"
            )
            if chosen:
                fig = go.Figure()
                for name in chosen:
                    tk = MARKET_SYMBOLS[name].upper()
                    s = market_close[tk].dropna()
                    fig.add_trace(go.Scatter(x=s.index, y=s / s.iloc[0] * 100, name=name, mode="lines"))
                style_fig(fig, height=330)
                st.plotly_chart(fig)

        v_left, v_right = st.columns(2)

        with v_left:
            section("VOLATILITY INDEX")
            vt = VOL_INDEX[currency] if VOL_INDEX[currency] in market_close.columns else "^VIX"
            if vt in market_close.columns:
                vs = market_close[vt].dropna()
                fig = go.Figure(go.Scatter(x=vs.index, y=vs, mode="lines", line=dict(color=AMBER), name=vt))
                for level, color in ((20, AMBER), (30, RED)):
                    fig.add_hline(y=level, line_dash="dot", line_color=color, opacity=0.6)
                style_fig(fig, height=280)
                st.plotly_chart(fig)

        with v_right:
            section("SECTOR PERFORMANCE")
            sectors = US_SECTORS if currency == "USD" else IN_SECTORS
            try:
                sec_close = download_close(tuple(v.upper() for v in sectors.values()), period="3mo")
                sec_rows = []
                for name, tk in sectors.items():
                    tk = tk.upper()
                    if tk in sec_close.columns:
                        s = sec_close[tk].dropna()
                        if len(s) > 22:
                            sec_rows.append({"Sector": name, "1D": s.iloc[-1] / s.iloc[-2] - 1,
                                             "1M": s.iloc[-1] / s.iloc[-22] - 1})
                if sec_rows:
                    sdf = pd.DataFrame(sec_rows).sort_values("1M")
                    fig = go.Figure()
                    fig.add_trace(go.Bar(y=sdf["Sector"], x=sdf["1M"] * 100, orientation="h", name="1M %",
                                         marker_color=[GREEN if v >= 0 else RED for v in sdf["1M"]]))
                    fig.add_trace(go.Bar(y=sdf["Sector"], x=sdf["1D"] * 100, orientation="h", name="1D %",
                                         marker_color=CYAN))
                    fig.update_layout(barmode="group")
                    style_fig(fig, height=280, hover="closest")
                    st.plotly_chart(fig)
                else:
                    st.info("Sector data not available.")
            except Exception:
                st.info("Sector data not available.")


    ai_explanation_widget("Market monitor", {"market_state": regime, "quotes": quotes.head(15).to_dict("records") if not quotes.empty else []}, "market")


# =========================================================
# TAB: PORTFOLIO (input + overview)
# =========================================================

ctx = None
ctx_error = None

with tab_port:
    section("BUILD YOUR PORTFOLIO")
    st.caption(
        "Enter ticker, shares and your purchase price per share. Add or remove rows directly. "
        "US examples: AAPL, MSFT, SPY. India examples: RELIANCE.NS, TCS.NS, INFY.NS."
    )

    # Keep the widget key stable; never write the returned frame back into session_state.
    if "initial_holdings_data" not in st.session_state:
        st.session_state["initial_holdings_data"] = pd.DataFrame([
            {"Ticker": "AAPL", "Shares": 10.0, "Purchase Price": 200.0},
            {"Ticker": "MSFT", "Shares": 5.0, "Purchase Price": 400.0},
            {"Ticker": "SPY", "Shares": 4.0, "Purchase Price": 500.0},
        ])

    holdings_input = st.data_editor(
        st.session_state["initial_holdings_data"],
        num_rows="dynamic",
        hide_index=True,
        key="holdings_editor",
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker Symbol"),
            "Shares": st.column_config.NumberColumn("Number of Shares", min_value=0.000001, step=1.0, format="%.4f"),
            "Purchase Price": st.column_config.NumberColumn("Purchase Price Per Share", min_value=0.0, step=1.0, format="%.2f"),
        },
    )

    valid_holdings, validation_message, validation_notes = validate_holdings(holdings_input, currency)

    for note in validation_notes:
        st.info(note)

    if valid_holdings is None:
        ctx_error = validation_message
        st.warning(validation_message)
    else:
        try:
            with st.spinner("Downloading market data and running the risk engine..."):
                ctx = build_context(valid_holdings, currency, start_date, end_date)
            for note in ctx["notes"]:
                st.warning(note)
        except Exception as exc:
            ctx_error = str(exc)
            st.error(f"Portfolio analysis failed: {exc}")

    if ctx is not None:
        st.caption(
            f"{len(ctx['tickers'])} investment(s) · {currency} · "
            f"{len(ctx['port_ret'])} daily return observations · "
            f"latest data {ctx['prices'].index[-1].strftime('%d %b %Y')} (may be delayed)"
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total invested", money(ctx["invested"], currency))
        c2.metric("Market value", money(ctx["value"], currency))
        c3.metric("Unrealized P&L", money(ctx["pnl"], currency),
                  delta=pct(ctx["pnl"] / ctx["invested"]) if ctx["invested"] > 0 else None)
        c4.metric("Today's P&L", money(ctx["daily_pnl"], currency),
                  delta=pct(ctx["daily_pnl"] / (ctx["value"] - ctx["daily_pnl"]))
                  if ctx["value"] - ctx["daily_pnl"] > 0 else None)

        section("HOLDINGS")
        disp = ctx["summary"].copy()
        disp["Allocation"] = ctx["weights"].to_numpy()
        for col in ["Purchase Price", "Current Price", "Invested Amount", "Current Value",
                    "Total Profit/Loss", "Daily Profit/Loss"]:
            disp[col] = disp[col].map(lambda x: money(x, currency))
        for col in ["Return on Cost", "Daily Price Change", "Allocation"]:
            disp[col] = disp[col].map(pct)
        if ctx["betas"] is not None:
            disp["Beta"] = ctx["betas"].reindex(disp["Ticker"]).map(lambda b: f"{b:.2f}").to_numpy()
        st.dataframe(disp, hide_index=True)

        # Historical liquidity proxy from 20-session average dollar volume.
        section("LIQUIDITY ESTIMATE (HISTORICAL VOLUME PROXY)")
        liq_rows = []
        for _, holding in ctx["summary"].iterrows():
            ticker = holding["Ticker"]
            adv = float(ctx["avg_dollar_volume"].get(ticker, np.nan)) if ticker in ctx["avg_dollar_volume"].index else np.nan
            position_value = float(holding["Current Value"])
            if np.isfinite(adv) and adv > 0:
                adv_share = position_value / adv
                est_days = position_value / (adv * (liquidity_participation_pct / 100.0))
            else:
                adv_share, est_days = np.nan, np.nan
            liq_rows.append({
                "Ticker": ticker, "Position value": money(position_value, currency),
                "20D avg daily traded value": money(adv, currency) if np.isfinite(adv) else "N/A",
                "Position / ADV": pct(adv_share) if np.isfinite(adv_share) else "N/A",
                f"Estimated sessions at {liquidity_participation_pct}% ADV": f"{est_days:.1f}" if np.isfinite(est_days) else "N/A",
            })
        st.dataframe(pd.DataFrame(liq_rows), hide_index=True)
        st.caption("Liquidity is approximated from historical volume and price. It excludes order-book depth, market impact, halts and intraday variation; estimates are not execution guarantees.")

        st.download_button(
            "Download holdings CSV",
            data=ctx["summary"].to_csv(index=False).encode("utf-8"),
            file_name="portfolio_holdings.csv",
            mime="text/csv",
        )

        l, r = st.columns(2)
        with l:
            section("PORTFOLIO VALUE (CURRENT QUANTITIES, HISTORICAL PRICES)")
            fig = go.Figure(go.Scatter(x=ctx["hist_values"].index, y=ctx["hist_values"], mode="lines",
                                       line=dict(color=AMBER), name="Value"))
            style_fig(fig, height=300)
            st.plotly_chart(fig)
        with r:
            section("DRAWDOWN FROM PEAK")
            fig = go.Figure(go.Scatter(x=ctx["drawdown"].index, y=ctx["drawdown"] * 100, fill="tozeroy",
                                       line=dict(color=RED), name="Drawdown %"))
            style_fig(fig, height=300)
            st.plotly_chart(fig)

        l, r = st.columns(2)
        with l:
            section("ALLOCATION")
            fig = px.pie(ctx["summary"], names="Ticker", values="Current Value", hole=0.5,
                         color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_traces(textinfo="label+percent")
            style_fig(fig, height=300, hover="closest")
            st.plotly_chart(fig)
        with r:
            section("PROFIT / LOSS BY HOLDING")
            pnl = ctx["summary"].sort_values("Total Profit/Loss")
            fig = go.Figure(go.Bar(x=pnl["Ticker"], y=pnl["Total Profit/Loss"],
                                   text=[money(v, currency) for v in pnl["Total Profit/Loss"]],
                                   textposition="outside",
                                   marker_color=[GREEN if v >= 0 else RED for v in pnl["Total Profit/Loss"]]))
            fig.update_xaxes(type="category")
            style_fig(fig, height=300, hover="closest")
            st.plotly_chart(fig)

        st.caption(
            "Historical values are reconstructed with your current share counts and do not reflect "
            "trades you made in the past. Profit/loss excludes fees, taxes and dividends."
        )


    if ctx is not None:
        ai_explanation_widget("Portfolio overview", {"currency": currency, "value": ctx["value"], "invested": ctx["invested"], "unrealized_pnl": ctx["pnl"], "daily_pnl": ctx["daily_pnl"], "weights": ctx["weights"].to_dict(), "annual_volatility": ctx["ann_vol"], "max_drawdown": ctx["max_dd"], "liquidity_proxy": ctx["avg_dollar_volume"].to_dict()}, "portfolio")


# =========================================================
# ENGINE RUN (shared by later tabs)
# =========================================================

engine = garch_n = garch_t = None
backtest_frames = backtest_df = None
score = score_label = score_parts = var1_conservative = None
optimizer = None
projection = None

if ctx is not None:
    try:
        engine, garch_n, garch_t = run_engine(ctx, confidence, horizon, simulations)
        score, score_label, score_parts, var1_conservative = risk_score(ctx)

        lookback = choose_lookback(len(ctx["port_ret"]))
        if lookback:
            backtest_frames = rolling_backtests(ctx["port_ret"], confidence, lookback)
            backtest_df = backtest_summary(backtest_frames, confidence)

        # Optimizer (needs 2+ assets)
        if len(ctx["tickers"]) >= 2:
            mu_ann = ctx["asset_returns"].mean().to_numpy() * 252
            cov_ann = ctx["asset_returns"].cov().to_numpy() * 252
            w_cur = ctx["weights"].to_numpy()
            optimizer = {
                "mu": mu_ann, "cov": cov_ann,
                "current": weight_stats(w_cur, mu_ann, cov_ann, rf_rate, ctx["asset_returns"], confidence),
                "w_current": w_cur,
            }
            for key in ("min_vol", "max_sharpe", "risk_parity"):
                w_opt = optimize_weights(mu_ann, cov_ann, rf_rate, max_weight_cap, key)
                optimizer[key + "_w"] = w_opt
                optimizer[key] = weight_stats(w_opt, mu_ann, cov_ann, rf_rate, ctx["asset_returns"], confidence)
    except Exception as exc:
        st.error(f"Risk engine error: {exc}")
        engine = None


# ---------- baseline / change tracking ----------

current_snapshot = None
baseline = None
changes = {"added": [], "removed": [], "resized": [], "changed": False}
base_metrics = {}

if ctx is not None and engine is not None:
    current_snapshot = portfolio_snapshot(ctx, engine, score, confidence, horizon)
    baseline = st.session_state.get("baseline")

    # A baseline is only comparable when market, confidence and horizon match.
    if (
        baseline is None
        or baseline["currency"] != currency
        or baseline["conf"] != confidence
        or baseline["horizon"] != horizon
    ):
        st.session_state["baseline"] = current_snapshot
        baseline = current_snapshot

    changes = diff_snapshots(baseline, current_snapshot)
    base_metrics = baseline["metrics"] if changes["changed"] else {}


def mdelta(key, current, kind, fallback=None):
    """Metric delta text: change versus the baseline portfolio when holdings changed."""

    base_value = base_metrics.get(key)
    if base_value is not None and np.isfinite(base_value) and np.isfinite(current):
        d = current - base_value
        if kind == "money":
            text = ("+" if d >= 0 else "") + money(d, currency)
        elif kind == "pp":
            text = f"{d * 100:+.2f} pp"
        else:
            text = f"{d:+.2f}"
        return text + " vs baseline", "inverse"
    return fallback, "off"


def need_portfolio():
    st.info(ctx_error or "Enter a valid portfolio in the PORTFOLIO tab.")


# =========================================================
# TAB: RISK ENGINE
# =========================================================

with tab_risk:
    if ctx is None or engine is None:
        need_portfolio()
    else:
        cons_row = engine.loc[engine["VaR"].idxmax()]
        h_txt = f"{horizon}-day"

        c1, c2, c3, c4, c5 = st.columns(5)

        d_score, k_score = mdelta("score", score, "num", score_label)
        d_var, k_var = mdelta("VaR_pct", cons_row["VaR_pct"], "pp", pct(cons_row["VaR_pct"]))
        d_es, k_es = mdelta("ES_pct", engine["ES_pct"].max(), "pp", pct(engine["ES_pct"].max()))
        d_vol, k_vol = mdelta("ann_vol", ctx["ann_vol"], "pp")
        d_beta, k_beta = mdelta("beta", ctx["port_beta"], "num")

        c1.metric("Risk score", f"{score:.0f}/100", delta=d_score, delta_color=k_score)
        c2.metric(f"{h_txt} VaR {confidence:.1%} (worst model)", money(cons_row["VaR"], currency),
                  delta=d_var, delta_color=k_var)
        c3.metric(f"{h_txt} Expected Shortfall", money(engine["ES"].max(), currency),
                  delta=d_es, delta_color=k_es)
        c4.metric("Annual volatility", pct(ctx["ann_vol"]), delta=d_vol, delta_color=k_vol)
        c5.metric("Portfolio beta", f"{ctx['port_beta']:.2f}" if np.isfinite(ctx["port_beta"]) else "N/A",
                  delta=d_beta, delta_color=k_beta)

        if changes["changed"]:
            st.caption("Deltas compare against your baseline portfolio. See the AI ANALYST tab for the full impact analysis.")

        l, r = st.columns([1, 2])

        with l:
            section("COMPOSITE RISK GAUGE")
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                number=dict(font=dict(color="#ffffff")),
                gauge=dict(
                    axis=dict(range=[0, 100], tickcolor=GREY),
                    bar=dict(color=AMBER),
                    bgcolor="#0b0b0b",
                    steps=[
                        dict(range=[0, 25], color="#0f3d22"),
                        dict(range=[25, 50], color="#3d3a0f"),
                        dict(range=[50, 75], color="#4d2a0a"),
                        dict(range=[75, 100], color="#4d0f0f"),
                    ],
                ),
            ))
            style_fig(gauge, height=250, hover="closest")
            st.plotly_chart(gauge)
            comp = pd.DataFrame({"Component": list(score_parts), "Points": list(score_parts.values())})
            st.dataframe(comp.assign(Points=comp["Points"].map(lambda x: f"{x:.1f}")), hide_index=True)

        with r:
            section(f"VaR / ES BY MODEL — {h_txt.upper()}, {confidence:.1%} CONFIDENCE")
            show = engine.copy()
            show["VaR"] = show["VaR"].map(lambda x: money(x, currency))
            show["ES"] = show["ES"].map(lambda x: money(x, currency))
            show["VaR_pct"] = show["VaR_pct"].map(pct)
            show["ES_pct"] = show["ES_pct"].map(pct)
            show = show.rename(columns={"VaR_pct": "VaR %", "ES_pct": "ES %"})
            st.dataframe(show, hide_index=True)

            fig = go.Figure()
            fig.add_trace(go.Bar(x=engine["Model"], y=engine["VaR"], name="VaR", marker_color=AMBER))
            fig.add_trace(go.Bar(x=engine["Model"], y=engine["ES"], name="Expected Shortfall", marker_color=RED))
            fig.update_layout(barmode="group")
            style_fig(fig, height=270, hover="closest")
            st.plotly_chart(fig)

        st.caption(
            "VaR: the loss threshold exceeded only (1 − confidence) of the time. ES: the average loss when "
            "that threshold is exceeded. Horizons above 1 day use square-root-of-time scaling for Historical "
            "and Parametric, scaled covariance for Monte Carlo, and the multi-step variance forecast for GARCH."
        )

        l, r = st.columns(2)

        with l:
            section("DAILY RETURN DISTRIBUTION VS NORMAL")
            rets = ctx["port_ret"]
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=rets * 100, nbinsx=60, histnorm="probability density",
                                       marker_color=CYAN, opacity=0.7, name="Historical"))
            xs = np.linspace(rets.min() * 100, rets.max() * 100, 300)
            fig.add_trace(go.Scatter(x=xs, y=norm.pdf(xs, rets.mean() * 100, rets.std() * 100),
                                     line=dict(color=AMBER), name="Normal fit"))
            v1, _ = hist_var_es(rets, 1.0, confidence, 1)
            fig.add_vline(x=-v1 * 100, line_dash="dash", line_color=RED,
                          annotation_text=f"1D hist VaR {v1:.2%}")
            style_fig(fig, height=320, hover="closest")
            fig.update_xaxes(title="Daily return (%)")
            st.plotly_chart(fig)
            c1, c2 = st.columns(2)
            c1.metric("Skewness", f"{skew(rets):.2f}")
            c2.metric("Excess kurtosis", f"{kurtosis(rets):.2f}")

        with r:
            section("GARCH CONDITIONAL VOLATILITY (ANNUALISED)")
            if garch_t or garch_n:
                g = garch_t or garch_n
                fig = go.Figure(go.Scatter(x=g["cond_vol"].index, y=g["cond_vol"] * np.sqrt(252) * 100,
                                           line=dict(color=AMBER), name="Cond. vol %"))
                fig.add_hline(y=ctx["ann_vol"] * 100, line_dash="dot", line_color=GREY,
                              annotation_text="Sample vol")
                style_fig(fig, height=320)
                st.plotly_chart(fig)
                c1, c2, c3 = st.columns(3)
                c1.metric("Next-day vol forecast (ann.)", pct(g["vol_daily"] * np.sqrt(252)))
                c2.metric("Persistence (α+β)", f"{g['persistence']:.3f}")
                c3.metric("Student-t d.o.f.", f"{garch_t['nu']:.1f}" if garch_t else "N/A")
            else:
                st.info("GARCH models unavailable (install `arch`, or the fit did not converge).")

        l, r = st.columns(2)

        with l:
            section("RISK CONTRIBUTION VS CAPITAL WEIGHT")
            rc = risk_contributions(ctx)
            fig = go.Figure()
            fig.add_trace(go.Bar(x=ctx["tickers"], y=ctx["weights"] * 100, name="Capital weight %", marker_color=CYAN))
            fig.add_trace(go.Bar(x=ctx["tickers"], y=rc.to_numpy() * 100, name="Variance contribution %", marker_color=AMBER))
            fig.update_layout(barmode="group")
            style_fig(fig, height=300, hover="closest")
            st.plotly_chart(fig)

        with r:
            section("CORRELATION MATRIX")
            if len(ctx["tickers"]) >= 2:
                corr = ctx["asset_returns"].corr()
                fig = px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdYlGn_r",
                                aspect="auto")
                style_fig(fig, height=300, hover="closest")
                st.plotly_chart(fig)
            else:
                st.info("Add a second holding to see correlations.")


    if ctx is not None and engine is not None:
        ai_explanation_widget("Risk engine", {"risk_score": score, "risk_score_label": score_label, "var_es_models": engine.to_dict("records"), "confidence": confidence, "horizon_days": horizon, "annual_volatility": ctx["ann_vol"], "portfolio_beta": ctx["port_beta"]}, "risk")


# =========================================================
# TAB: P&L PROJECTION
# =========================================================

with tab_proj:
    if ctx is None:
        need_portfolio()
    else:
        section("PROFIT / LOSS PROJECTION (SIMULATED PORTFOLIO PATHS)")

        p1, p2, p3 = st.columns(3)
        horizon_label = p1.selectbox(
            "Projection horizon",
            ["1 week (5d)", "2 weeks (10d)", "1 month (21d)", "3 months (63d)", "6 months (126d)", "1 year (252d)"],
            index=2,
        )
        proj_h = int(horizon_label.split("(")[1].split("d")[0])
        method = p2.selectbox("Return model", ["Bootstrap (historical)", "Student-t (fat tails)", "Normal"])
        use_drift = p3.selectbox(
            "Expected drift", ["Zero drift (conservative)", "Historical average return"]
        ).startswith("Historical")

        n_paths = int(min(simulations, 20_000))
        paths = simulate_paths(ctx["port_ret"], ctx["value"], proj_h, n_paths, method, use_drift)
        terminal = paths[:, -1]
        pnl_dist = terminal - ctx["value"]

        loss_dist = -pnl_dist
        var_h = float(np.quantile(loss_dist, confidence))
        es_h = float(loss_dist[loss_dist >= var_h].mean())

        projection = {
            "horizon": proj_h,
            "prob_loss": float((pnl_dist < 0).mean()),
        }

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Expected P&L", money(float(pnl_dist.mean()), currency))
        c2.metric("Median P&L", money(float(np.median(pnl_dist)), currency))
        c3.metric("Probability of loss", f"{projection['prob_loss']:.1%}")
        c4.metric(f"Loss at {confidence:.0%} (VaR)", money(var_h, currency))
        c5.metric("Expected Shortfall", money(es_h, currency))

        days = np.arange(proj_h + 1)
        q = np.percentile(paths, [5, 25, 50, 75, 95], axis=0)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=days, y=q[4], line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=days, y=q[0], fill="tonexty", fillcolor="rgba(255,159,26,0.12)",
                                 line=dict(width=0), name="5–95%"))
        fig.add_trace(go.Scatter(x=days, y=q[3], line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=days, y=q[1], fill="tonexty", fillcolor="rgba(255,159,26,0.28)",
                                 line=dict(width=0), name="25–75%"))
        fig.add_trace(go.Scatter(x=days, y=q[2], line=dict(color=AMBER, width=2), name="Median"))
        fig.add_hline(y=ctx["value"], line_dash="dot", line_color=GREY, annotation_text="Current value")
        style_fig(fig, height=380, title=f"Portfolio value fan chart — {n_paths:,} paths, {method}")
        fig.update_xaxes(title="Trading days ahead")
        st.plotly_chart(fig)

        l, r = st.columns(2)

        with l:
            section("TERMINAL P&L DISTRIBUTION")
            fig = go.Figure(go.Histogram(x=pnl_dist, nbinsx=60, marker_color=CYAN, opacity=0.8))
            fig.add_vline(x=0, line_dash="dash", line_color=GREY)
            fig.add_vline(x=-var_h, line_dash="dash", line_color=RED, annotation_text="VaR")
            style_fig(fig, height=300, hover="closest")
            fig.update_xaxes(title=f"P&L over {proj_h} days ({currency})")
            st.plotly_chart(fig)

        with r:
            section("OUTCOME PROBABILITIES")
            thresholds = [-0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20]
            rows = []
            for th in thresholds:
                if th < 0:
                    p = float((pnl_dist <= th * ctx["value"]).mean())
                    label = f"Loss of {abs(th):.0%} or worse"
                elif th == 0:
                    p = float((pnl_dist > 0).mean())
                    label = "Any gain"
                else:
                    p = float((pnl_dist >= th * ctx["value"]).mean())
                    label = f"Gain of {th:.0%} or more"
                rows.append({
                    "Outcome": label,
                    "Probability": f"{p:.1%}",
                    f"P&L threshold ({currency})": money(th * ctx["value"], currency),
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True)

        st.caption(
            "This is a statistical simulation based on past returns, not a forecast. It assumes the current "
            "weights are maintained, ignores fees and taxes, and cannot anticipate events unlike those in the "
            "sample. Zero drift is the conservative default because historical average returns are noisy."
        )


    if ctx is not None and projection is not None:
        ai_explanation_widget("P&L projection", {"projection": projection, "portfolio_value": ctx["value"], "risk_tolerance": risk_tolerance}, "projection")


# =========================================================
# TAB: OPTIMIZER
# =========================================================

with tab_opt:
    if ctx is None:
        need_portfolio()
    elif optimizer is None:
        st.info("Add at least two holdings to run the portfolio optimizer.")
    else:
        section("PORTFOLIO SUGGESTIONS")

        names = {
            "Current": "current",
            "Minimum Volatility": "min_vol",
            "Maximum Sharpe": "max_sharpe",
            "Risk Parity": "risk_parity",
        }

        rows = []
        for label, key in names.items():
            s = optimizer[key]
            rows.append({
                "Strategy": label,
                "Ann. return": pct(s["Ann. return"]),
                "Ann. volatility": pct(s["Ann. volatility"]),
                "Sharpe": f"{s['Sharpe']:.2f}",
                "1D VaR (hist.)": pct(s["1D VaR"]),
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True)

        target_label = st.selectbox("Target allocation", list(names)[1:], index=0)
        target_key = names[target_label]
        w_target = optimizer[target_key + "_w"]
        w_current = optimizer["w_current"]

        l, r = st.columns(2)

        with l:
            section("CURRENT VS TARGET WEIGHTS")
            fig = go.Figure()
            fig.add_trace(go.Bar(x=ctx["tickers"], y=w_current * 100, name="Current", marker_color=CYAN))
            fig.add_trace(go.Bar(x=ctx["tickers"], y=w_target * 100, name=target_label, marker_color=AMBER))
            fig.update_layout(barmode="group")
            style_fig(fig, height=320, hover="closest")
            fig.update_yaxes(title="Weight (%)")
            st.plotly_chart(fig)

        with r:
            section("RISK / RETURN MAP (RANDOM PORTFOLIOS)")
            rng = np.random.default_rng(11)
            n_assets = len(ctx["tickers"])
            rand_w = rng.dirichlet(np.ones(n_assets), size=3000)
            rand_w = rand_w[(rand_w <= max(max_weight_cap, 1 / n_assets + 1e-9)).all(axis=1)]
            if len(rand_w) == 0:
                rand_w = rng.dirichlet(np.ones(n_assets), size=3000)
            r_ret = rand_w @ optimizer["mu"]
            r_vol = np.sqrt(np.einsum("ij,jk,ik->i", rand_w, optimizer["cov"], rand_w))

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=r_vol * 100, y=r_ret * 100, mode="markers", name="Random",
                marker=dict(size=4, color=(r_ret - rf_rate) / r_vol, colorscale="Viridis",
                            showscale=False, opacity=0.6),
            ))
            for label, key in names.items():
                s = optimizer[key]
                fig.add_trace(go.Scatter(
                    x=[s["Ann. volatility"] * 100], y=[s["Ann. return"] * 100], mode="markers+text",
                    name=label, text=[label], textposition="top center",
                    marker=dict(size=12, symbol="diamond",
                                color=AMBER if key != "current" else RED),
                ))
            style_fig(fig, height=320, hover="closest")
            fig.update_xaxes(title="Annual volatility (%)")
            fig.update_yaxes(title="Annual return (%)")
            st.plotly_chart(fig)

        section(f"REBALANCING TRADES → {target_label.upper()}")

        price = ctx["summary"]["Current Price"].to_numpy()
        delta_value = (w_target - w_current) * ctx["value"]
        delta_shares = delta_value / price

        trades = pd.DataFrame({
            "Ticker": ctx["tickers"],
            "Current weight": w_current,
            "Target weight": w_target,
            "Change": w_target - w_current,
            "Trade value": delta_value,
            "Trade shares": delta_shares,
        })
        total_cost_bps = transaction_cost_bps + slippage_bps
        trades["Estimated execution cost"] = trades["Trade value"].abs() * total_cost_bps / 10000.0
        trades["Action"] = np.where(
            trades["Change"] > 0.005, "BUY", np.where(trades["Change"] < -0.005, "SELL", "HOLD")
        )

        trades_disp = trades.copy()
        for col in ["Current weight", "Target weight", "Change"]:
            trades_disp[col] = trades_disp[col].map(lambda x: f"{x:+.1%}" if col == "Change" else f"{x:.1%}")
        trades_disp["Trade value"] = trades["Trade value"].map(lambda x: money(x, currency))
        trades_disp["Trade shares"] = trades["Trade shares"].map(lambda x: f"{x:+,.2f}")
        trades_disp["Estimated execution cost"] = trades["Estimated execution cost"].map(lambda x: money(x, currency))
        st.dataframe(trades_disp, hide_index=True)
        gross_turnover = float(trades["Trade value"].abs().sum())
        estimated_cost = float(trades["Estimated execution cost"].sum())
        net_after_cost_value = ctx["value"] - estimated_cost
        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("Gross rebalance turnover", money(gross_turnover, currency))
        cc2.metric("Estimated transaction cost + slippage", money(estimated_cost, currency))
        cc3.metric("Cost as % of portfolio", pct(estimated_cost / ctx["value"] if ctx["value"] else np.nan))

        cur_s, tgt_s = optimizer["current"], optimizer[target_key]
        c1, c2, c3 = st.columns(3)
        c1.metric("Volatility", pct(tgt_s["Ann. volatility"]),
                  delta=f"{(tgt_s['Ann. volatility'] - cur_s['Ann. volatility']) * 100:+.2f} pp", delta_color="inverse")
        c2.metric("1D VaR (hist.)", pct(tgt_s["1D VaR"]),
                  delta=f"{(tgt_s['1D VaR'] - cur_s['1D VaR']) * 100:+.2f} pp", delta_color="inverse")
        c3.metric("Sharpe", f"{tgt_s['Sharpe']:.2f}", delta=f"{tgt_s['Sharpe'] - cur_s['Sharpe']:+.2f}")

        st.caption(
            "Optimization uses historical means and covariances over your selected history with a "
            f"{max_weight_cap:.0%} cap per asset and no short selling. Historical returns are poor predictors "
            "of future returns, so minimum-volatility and risk-parity targets are generally more reliable than "
            "max-Sharpe. Ignores transaction costs, taxes and liquidity. Not investment advice."
        )


    if ctx is not None and optimizer is not None:
        ai_explanation_widget("Portfolio optimizer", {"risk_tolerance": risk_tolerance, "optimizer_results": {k: optimizer[k] for k in ["current", "min_vol", "max_sharpe", "risk_parity"]}, "target_weight_scenarios": {k: dict(zip(ctx["tickers"], optimizer[k + "_w"])) for k in ["min_vol", "max_sharpe", "risk_parity"]}, "transaction_cost_bps": transaction_cost_bps, "slippage_bps": slippage_bps}, "optimizer")


# =========================================================
# TAB: STRESS TEST
# =========================================================

with tab_stress:
    if ctx is None or engine is None:
        need_portfolio()
    else:
        bench_name = "S&P 500" if currency == "USD" else "NIFTY 50"

        section(f"MARKET-SHOCK SCENARIOS (BETA-IMPLIED, VS {bench_name})")
        if ctx["betas"] is not None:
            shocks = [-0.05, -0.10, -0.20, -0.30, 0.10]
            rows = []
            for s in shocks:
                pnl = float((ctx["weights"] * ctx["betas"]).sum() * s * ctx["value"])
                rows.append({
                    "Scenario": f"{bench_name} {s:+.0%}",
                    "Portfolio return": pct(pnl / ctx["value"]),
                    "P&L": money(pnl, currency),
                    "Value after": money(ctx["value"] + pnl, currency),
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True)
            st.caption("Uses each holding's historical beta. Real crises usually push correlations and betas higher than history shows.")
        else:
            st.info("Benchmark data not available, so beta-implied scenarios cannot be computed.")

        l, r = st.columns(2)

        with l:
            section("REPLAY OF WORST HISTORICAL DAYS")
            worst = ctx["port_ret"].nsmallest(8)
            wdf = pd.DataFrame({
                "Date": [d.strftime("%d %b %Y") for d in worst.index],
                "Portfolio return": [pct(v) for v in worst.values],
                f"P&L on today's value": [money(v * ctx["value"], currency) for v in worst.values],
            })
            st.dataframe(wdf, hide_index=True)

        with r:
            section("VOLATILITY SHOCK ON VaR")
            mult = st.select_slider("Volatility multiplier", options=[1.0, 1.25, 1.5, 2.0, 2.5, 3.0], value=2.0)
            base_var, base_es = param_var_es(ctx["port_ret"], ctx["value"], confidence, horizon)
            sigma = ctx["port_ret"].std(ddof=1)
            z = norm.ppf(confidence)
            stressed_var = (z * sigma * mult * np.sqrt(horizon) - ctx["port_ret"].mean() * horizon) * ctx["value"]
            c1, c2 = st.columns(2)
            c1.metric("Base normal VaR", money(base_var, currency))
            c2.metric(f"VaR at {mult:.2g}× volatility", money(float(stressed_var), currency))

        section("CUSTOM SCENARIO — SET A PRICE CHANGE PER HOLDING")

        shock_frame = ctx["summary"][["Ticker", "Current Value"]].copy()
        shock_frame["Hypothetical Change (%)"] = -10.0

        edited = st.data_editor(
            shock_frame,
            hide_index=True,
            key="shocks_" + "_".join(ctx["tickers"]),
            disabled=["Ticker", "Current Value"],
            column_config={
                "Current Value": st.column_config.NumberColumn(f"Current Value ({currency})", format="%.2f"),
                "Hypothetical Change (%)": st.column_config.NumberColumn(
                    "Hypothetical Price Change (%)", min_value=-100.0, max_value=100.0, step=1.0, format="%.2f"),
            },
        )

        shock_vals = pd.to_numeric(edited["Hypothetical Change (%)"], errors="coerce")

        if shock_vals.isna().any():
            st.error("Enter valid hypothetical percentage changes.")
        else:
            pnl_each = ctx["summary"]["Current Value"].to_numpy() * shock_vals.to_numpy() / 100
            total = float(pnl_each.sum())
            c1, c2, c3 = st.columns(3)
            c1.metric("Scenario P&L", money(total, currency))
            c2.metric("Value after scenario", money(ctx["value"] + total, currency))
            c3.metric("Scenario return", pct(total / ctx["value"]))
            st.caption(
                "Hypothetical arithmetic only; assumes each holding changes by the entered percentage and "
                "ignores liquidity, costs and changing correlations."
            )


    if ctx is not None and engine is not None:
        ai_explanation_widget("Stress testing", {"portfolio_value": ctx["value"], "market_regime": regime, "historical_worst_returns": ctx["port_ret"].nsmallest(8).to_dict(), "historical_beta": ctx["betas"].to_dict() if ctx["betas"] is not None else None}, "stress")


# =========================================================
# TAB: BACKTEST
# =========================================================

with tab_bt:
    if ctx is None:
        need_portfolio()
    elif backtest_df is None:
        st.info("At least ~150 daily observations are needed for rolling backtests. Extend the history start date.")
    else:
        section(f"ROLLING 1-DAY VaR BACKTEST — {confidence:.1%} CONFIDENCE, {choose_lookback(len(ctx['port_ret']))}-DAY WINDOW")

        show = backtest_df.copy()
        show["Rate"] = show["Rate"].map(lambda x: f"{x:.2%}")
        show["Expected"] = show["Expected"].map(lambda x: f"{x:.2%}")
        for col in ["Kupiec p", "Independence p", "Cond. Coverage p"]:
            show[col] = show[col].map(lambda x: f"{x:.4f}")
        st.dataframe(show, hide_index=True)

        st.caption(
            "Kupiec: are exceptions as frequent as expected? Christoffersen: are exceptions independent "
            "(not clustered)? Conditional coverage combines both. p < 0.05 rejects the model. Basel zones "
            "(for 99% VaR): green ≤ 4, yellow 5–9, red ≥ 10 exceptions in the last 250 days. Failing to "
            "reject does not prove a model is correct."
        )

        model_name = st.selectbox("Model to plot", list(backtest_frames))
        df = backtest_frames[model_name]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df["Loss"] * ctx["value"], line=dict(color=GREY, width=1), name="Actual loss"))
        fig.add_trace(go.Scatter(x=df.index, y=df["VaR"] * ctx["value"], line=dict(color=AMBER, width=1.5), name="VaR"))
        exc = df[df["Exception"] == 1]
        fig.add_trace(go.Scatter(x=exc.index, y=exc["Loss"] * ctx["value"], mode="markers",
                                 marker=dict(color=RED, symbol="x", size=8), name="Exceptions"))
        style_fig(fig, height=380, title=f"{model_name}: loss vs VaR (in {currency}, on today's portfolio value)")
        st.plotly_chart(fig)

        roll = pd.DataFrame({
            name: f["Exception"].rolling(60).sum() for name, f in backtest_frames.items()
        })
        fig = go.Figure()
        for name in roll.columns:
            fig.add_trace(go.Scatter(x=roll.index, y=roll[name], name=name, mode="lines"))
        fig.add_hline(y=60 * (1 - confidence), line_dash="dash", line_color=GREY, annotation_text="Expected")
        style_fig(fig, height=280, title="Rolling 60-day exception count")
        st.plotly_chart(fig)

        st.download_button(
            "Download backtest summary CSV",
            data=backtest_df.to_csv(index=False).encode("utf-8"),
            file_name="var_backtest_summary.csv",
            mime="text/csv",
        )


    if backtest_df is not None:
        ai_explanation_widget("VaR backtesting", {"confidence": confidence, "summary": backtest_df.to_dict("records"), "lookback_days": choose_lookback(len(ctx["port_ret"])) if ctx is not None else None}, "backtest")


# =========================================================
# TAB: SIGNALS
# =========================================================

with tab_sig:
    if ctx is None or engine is None:
        need_portfolio()
    else:
        signals = build_signals(ctx, engine, backtest_df, regime, optimizer, score, projection)

        section("RISK SIGNALS AND RECOMMENDATIONS")
        tags = {"high": ("HIGH", RED), "med": ("WATCH", AMBER), "info": ("INFO", CYAN), "low": ("OK", GREEN)}

        for level, title, text in signals:
            tag, color = tags[level]
            st.markdown(
                f'<div class="sig {level}"><span class="tag" style="color:{color}">[{tag}]</span>'
                f"<b>{html.escape(title)}</b><br>{html.escape(text)}</div>",
                unsafe_allow_html=True,
            )

        section("HOLDING-LEVEL DECISION SUPPORT")
        rc_now = risk_contributions(ctx)
        if optimizer is not None:
            target_name = {"Conservative": "min_vol", "Moderate": "risk_parity", "Aggressive": "max_sharpe"}[risk_tolerance]
            target_weights = pd.Series(optimizer[target_name + "_w"], index=ctx["tickers"])
        else:
            target_weights = ctx["weights"]
        action_rows = []
        for _, holding in ctx["summary"].iterrows():
            ticker = holding["Ticker"]
            current_w = float(ctx["weights"][ticker])
            target_w = float(target_weights[ticker])
            gap = target_w - current_w
            risk_share = float(rc_now[ticker])
            if gap <= -0.05:
                action = "REVIEW / REDUCE TOWARD TARGET"
            elif gap >= 0.05:
                action = "REVIEW / INCREASE TOWARD TARGET"
            else:
                action = "MAINTAIN / MONITOR"
            if risk_tolerance == "Conservative" and (risk_share >= 0.35 or current_w >= 0.30):
                action = "REVIEW CONCENTRATION"
            action_rows.append({
                "Ticker": ticker, "Current weight": pct(current_w), "Target weight (scenario)": pct(target_w),
                "Weight gap": f"{gap * 100:+.1f} pp", "Risk contribution": pct(risk_share),
                "Historical return on cost": pct(float(holding["Return on Cost"])), "Suggested review": action,
            })
        st.dataframe(pd.DataFrame(action_rows), hide_index=True)
        target_description = {"Conservative": "minimum volatility", "Moderate": "risk parity", "Aggressive": "maximum Sharpe"}[risk_tolerance]
        st.caption(
            f"Target scenario is selected from your {risk_tolerance.lower()} risk preference ({target_description}). "
            "These are model-based review prompts, not automatic trade instructions. Optimizer weights are sensitive to the selected history."
        )

        section("RISK SNAPSHOT")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Market state", regime["label"])
        c2.metric("Portfolio risk score", f"{score:.0f} ({score_label})")
        c3.metric(f"{horizon}-day {confidence:.0%} VaR (worst model)", money(engine["VaR"].max(), currency))
        c4.metric("Max drawdown (sample)", pct(ctx["max_dd"]))

        st.caption(
            "Signals are rule-based observations computed from historical data and statistical models. "
            "They are educational and are not investment advice or predictions of future performance."
        )


    if ctx is not None and engine is not None:
        ai_explanation_widget("Signals and decision support", {"risk_tolerance": risk_tolerance, "risk_score": score, "risk_score_label": score_label, "signals": build_signals(ctx, engine, backtest_df, regime, optimizer, score, projection), "holdings": ctx["summary"].to_dict("records"), "weights": ctx["weights"].to_dict()}, "signals")


# =========================================================
# TAB: AI ANALYST
# =========================================================

with tab_ai:
    if ctx is None or engine is None:
        need_portfolio()
    else:
        section("CHANGE IMPACT VS BASELINE PORTFOLIO")

        bcol1, bcol2 = st.columns([4, 1])
        with bcol1:
            if changes["changed"]:
                parts = []
                if changes["added"]:
                    parts.append("added " + ", ".join(changes["added"]))
                if changes["removed"]:
                    parts.append("removed " + ", ".join(changes["removed"]))
                if changes["resized"]:
                    parts.append("resized " + ", ".join(r["ticker"] for r in changes["resized"]))
                st.markdown(f"**Changes since baseline:** {'; '.join(parts)}")
            else:
                st.caption(
                    "No changes versus the baseline yet. Add, remove or resize a holding in the PORTFOLIO tab "
                    "to see how VaR, ES and other metrics move. The baseline resets if you change currency, "
                    "VaR confidence or VaR horizon."
                )
        with bcol2:
            if st.button("Set current as baseline"):
                st.session_state["baseline"] = current_snapshot
                st.session_state.pop("ai_cache", None)
                st.rerun()

        impacts = []
        if changes["changed"]:
            bm, cm = baseline["metrics"], current_snapshot["metrics"]
            spec = [
                ("Portfolio value", "value", "money"),
                (f"VaR {confidence:.0%} {horizon}d (worst model)", "VaR", "money"),
                ("VaR % of value", "VaR_pct", "pct"),
                ("Expected Shortfall", "ES", "money"),
                ("ES % of value", "ES_pct", "pct"),
                ("Annual volatility", "ann_vol", "pct"),
                ("Beta", "beta", "num"),
                ("Risk score", "score", "num"),
                ("Largest position", "max_weight", "pct"),
                ("Top risk contributor share", "top_risk_share", "pct"),
                ("Average correlation", "avg_corr", "num"),
                ("Max drawdown", "max_dd", "pct"),
            ]

            def fmt(v, kind):
                if v is None or not np.isfinite(v):
                    return "N/A"
                return money(v, currency) if kind == "money" else pct(v) if kind == "pct" else f"{v:.2f}"

            rows = []
            for label, key, kind in spec:
                b, c = bm.get(key, np.nan), cm.get(key, np.nan)
                if np.isfinite(b) and np.isfinite(c):
                    d = c - b
                    change = (("+" if d >= 0 else "") + money(d, currency)) if kind == "money" \
                        else f"{d * 100:+.2f} pp" if kind == "pct" else f"{d:+.2f}"
                else:
                    change = "N/A"
                rows.append({"Metric": label, "Before": fmt(b, kind), "After": fmt(c, kind), "Change": change})

            st.dataframe(pd.DataFrame(rows), hide_index=True)

            touched = changes["added"] + [r["ticker"] for r in changes["resized"]]
            for t in touched:
                info = ticker_impact(ctx, t, confidence, horizon)
                if info:
                    impacts.append(info)

            if impacts:
                section("WHAT EACH NEW OR CHANGED HOLDING DOES TO RISK")
                imp = pd.DataFrame(impacts)
                imp_show = pd.DataFrame({
                    "Ticker": imp["Ticker"],
                    "Weight": imp["Weight"].map(pct),
                    "Stand-alone vol": imp["Ann. vol"].map(pct),
                    "Beta": imp["Beta"].map(lambda x: f"{x:.2f}" if np.isfinite(x) else "N/A"),
                    "Share of portfolio variance": imp["Risk share"].map(pct),
                    "Correlation with rest": imp["Corr. with rest"].map(lambda x: f"{x:.2f}" if np.isfinite(x) else "N/A"),
                    "VaR % without it": imp["VaR % without"].map(pct),
                    "VaR % with it": imp["VaR % with"].map(pct),
                    "VaR change": imp["VaR % change"].map(lambda x: f"{x * 100:+.2f} pp" if np.isfinite(x) else "N/A"),
                })
                st.dataframe(imp_show, hide_index=True)
                st.caption(
                    "'Without it' removes the holding and rescales the remaining weights to 100%, using the "
                    "more conservative of historical and normal VaR. A holding with low correlation and "
                    "modest weight can add return potential while barely moving VaR; a high-volatility, "
                    "highly correlated one raises it."
                )

        section("AI RECOMMENDATION")

        signals_for_ai = build_signals(ctx, engine, backtest_df, regime, optimizer, score, projection)
        payload = build_ai_payload(
            ctx, engine, backtest_df, regime, optimizer, score, score_label,
            baseline, current_snapshot, changes, impacts, signals_for_ai, confidence, horizon,
        )
        payload_json = json.dumps(payload, sort_keys=True)
        cache_key = hashlib.md5((ai_provider + ai_model + payload_json).encode()).hexdigest()
        cache = st.session_state.setdefault("ai_cache", {})

        a1, a2 = st.columns([1, 2])
        run_now = a1.button("Generate AI recommendation", type="primary")
        auto = a2.checkbox("Auto-analyze when holdings change", value=False,
                           help="Calls the AI model automatically each time the portfolio changes. Uses your API quota.")

        should_run = run_now or (auto and changes["changed"] and cache_key not in cache)

        if should_run:
            if not ai_key:
                st.warning("AI is not configured. Set LLM credentials in `.streamlit/secrets.toml` or server environment variables.")
            else:
                try:
                    with st.spinner("Analyst is reviewing the portfolio..."):
                        cache[cache_key] = call_llm(
                            ai_provider, ai_key, ai_model, ai_base_url,
                            AI_SYSTEM_PROMPT,
                            "Analyse this portfolio and respond in the required format.\n\n```json\n"
                            + payload_json + "\n```",
                        )
                except Exception as exc:
                    st.error(f"AI request failed: {exc}")

        if cache_key in cache:
            with st.container(border=True):
                st.markdown(cache[cache_key])
            st.caption(
                f"Generated by {ai_model}. AI output can be wrong; it is derived only from the metrics shown "
                "in this dashboard. Educational analysis, not investment advice."
            )
        elif not should_run:
            st.caption(
                "Click the button to get a plain-language recommendation from an LLM. It sees your holdings, "
                "VaR/ES for every model, backtest results, market state and, if you changed the portfolio, "
                "the before/after metrics."
            )

        with st.expander("Data sent to the model"):
            st.code(json.dumps(payload, indent=2), language="json")


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")
st.caption(
    "Market Risk Terminal · educational quantitative-finance project · Yahoo Finance data may be delayed · "
    "risk estimates use current holdings/weights and historical returns · liquidity and execution costs are approximations · "
    "no automatic currency conversion or broker execution · not investment advice."
)