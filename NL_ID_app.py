# -*- coding: utf-8 -*-
"""
SHI – STOCK CHECK / Intraday-Mode (letzte 10 Handelstage, nur RTH)
-----------------------------------------------------------------------------
Kernänderungen ggü. deiner Vorlage:
1) Neuer Datenmodus "Intraday" mit Lookback in Handelstagen und RTH-Filter (Regular Trading Hours).
2) Loader `get_intraday_past_n_days()` lädt Intraday-Balken (1m/2m/5m/15m), schneidet
   Nicht-Handelszeiten weg (inkl. Mittagspausen bei Asienbörsen) und liefert nur die
   letzten N Handelstage.
3) Backtest läuft auf Bars ("Next Bar" via Open des Folge-Balkens). Haltedauer/Cooldown in Bars.
4) Annualisierung wird dynamisch aus Bars/Tag×252 abgeleitet, damit Sharpe/Sortino/CAGR korrekt sind.

Hinweis: Yahoo limitiert 1m auf ~7 Tage. Für 10 Tage nimm 2m/5m/15m.
"""

# ─────────────────────────────────────────────────────────────
# Imports & Global Config
# ─────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from math import sqrt
from datetime import datetime, time, timedelta
from typing import Tuple, List, Dict, Optional
from zoneinfo import ZoneInfo

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="SHI – STOCK CHECK • Intraday", layout="wide")
LOCAL_TZ = ZoneInfo("Europe/Zurich")
pd.options.display.float_format = "{:,.4f}".format

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def to_csv_eu(df: pd.DataFrame, float_format: Optional[str] = None) -> bytes:
    return df.to_csv(index=False, sep=";", decimal=",", date_format="%d.%m.%Y",
                     float_format=float_format).encode("utf-8-sig")


def _normalize_tickers(items: List[str]) -> List[str]:
    cleaned = []
    for x in items or []:
        if not isinstance(x, str):
            continue
        s = x.strip().upper()
        if s:
            cleaned.append(s)
    return list(dict.fromkeys(cleaned))


@st.cache_data(show_spinner=False, ttl=24*60*60)
def get_ticker_name(ticker: str) -> str:
    try:
        tk = yf.Ticker(ticker)
        info = {}
        try:
            info = tk.get_info()
        except Exception:
            info = getattr(tk, "info", {}) or {}
        for k in ("shortName", "longName", "displayName", "companyName", "name"):
            if k in info and info[k]:
                return str(info[k])
    except Exception:
        pass
    return ticker


# ─────────────────────────────────────────────────────────────
# Intraday: Börsenkalender (heuristisch, suffixtauglich)
# ─────────────────────────────────────────────────────────────

from functools import lru_cache

_EX_SUFFIX_MAP: Dict[str, Dict] = {
    # Europa
    ".DE": {"tz": "Europe/Berlin",  "sessions": [("09:00","17:30")]},  # Xetra
    ".F":  {"tz": "Europe/Berlin",  "sessions": [("09:00","17:30")]},
    ".SW": {"tz": "Europe/Zurich",  "sessions": [("09:00","17:30")]},  # SIX
    ".PA": {"tz": "Europe/Paris",   "sessions": [("09:00","17:30")]},
    ".AS": {"tz": "Europe/Amsterdam","sessions": [("09:00","17:30")]},
    ".MI": {"tz": "Europe/Rome",    "sessions": [("09:00","17:30")]},
    ".L":  {"tz": "Europe/London",  "sessions": [("08:00","16:30")]},
    ".IR": {"tz": "Europe/Dublin",  "sessions": [("08:00","16:30")]},

    # Nordamerika
    ".TO": {"tz": "America/Toronto","sessions": [("09:30","16:00")]},
    ".V":  {"tz": "America/Toronto","sessions": [("09:30","16:00")]},
    # Asien
    ".HK": {"tz": "Asia/Hong_Kong", "sessions": [("09:30","12:00"),("13:00","16:00")]},
    ".T":  {"tz": "Asia/Tokyo",     "sessions": [("09:00","11:30"),("12:30","15:00")]},
    ".SS": {"tz": "Asia/Shanghai",  "sessions": [("09:30","11:30"),("13:00","15:00")]},
    ".SZ": {"tz": "Asia/Shanghai",  "sessions": [("09:30","11:30"),("13:00","15:00")]},
}

_DEFAULT_US = {"tz": "America/New_York", "sessions": [("09:30","16:00")]}


def _ticker_suffix(tk: str) -> Optional[str]:
    parts = tk.split('.')
    if len(parts) >= 2:
        return f".{parts[-1]}"
    return None


@lru_cache(maxsize=2048)
def infer_exchange_profile(ticker: str) -> Dict:
    """Best effort: aus Suffix oder Yahoo-Info die Börsenzeiten ableiten."""
    sx = _ticker_suffix(ticker)
    if sx and sx in _EX_SUFFIX_MAP:
        return _EX_SUFFIX_MAP[sx]

    # Fallback über Yahoo-Info
    tz = None
    try:
        tk = yf.Ticker(ticker)
        fi = getattr(tk, "fast_info", None)
        if fi and getattr(fi, "timezone", None):
            tz = fi.timezone
        else:
            info = {}
            try:
                info = tk.get_info()
            except Exception:
                info = getattr(tk, "info", {}) or {}
            tz = info.get("exchangeTimezoneName")
    except Exception:
        pass

    if tz:
        # generische Sitzungszeiten je Region
        if tz.startswith("America/"):
            return {"tz": tz, "sessions": [("09:30","16:00")]}
        if tz.startswith("Europe/"):
            return {"tz": tz, "sessions": [("09:00","17:30")]}
        if tz.startswith("Asia/Hong_Kong"):
            return {"tz": tz, "sessions": [("09:30","12:00"),("13:00","16:00")]}
        if tz.startswith("Asia/Shanghai"):
            return {"tz": tz, "sessions": [("09:30","11:30"),("13:00","15:00")]}
        if tz.startswith("Asia/Tokyo"):
            return {"tz": tz, "sessions": [("09:00","11:30"),("12:30","15:00")]}

    return _DEFAULT_US


def _to_time(hhmm: str) -> time:
    hh, mm = map(int, hhmm.split(':'))
    return time(hh, mm)


def apply_rth_filter(df: pd.DataFrame, tz_str: str, sessions: List[Tuple[str,str]]) -> pd.DataFrame:
    if df.empty:
        return df
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    local = df.index.tz_convert(ZoneInfo(tz_str))
    df = df.copy()
    df.index = local

    # Zeitfenster-Maske über alle Session-Intervalle
    mask = np.zeros(len(df), dtype=bool)
    times = df.index.time
    for o, c in sessions:
        t1, t2 = _to_time(o), _to_time(c)
        mask |= (times >= t1) & (times <= t2)

    df = df.loc[mask]
    # zurück in lokale App-Zeitzone
    df.index = df.index.tz_convert(LOCAL_TZ)
    return df


@st.cache_data(show_spinner=False, ttl=180)
def get_intraday_past_n_days(
    ticker: str,
    days: int = 10,
    interval: str = "5m",
    regular_only: bool = True,
) -> pd.DataFrame:
    """Lädt Intraday-Balken für die letzten N Handelstage und schneidet Nicht-RTH ab."""
    # Puffer, da `period` kalendarische Tage nutzt
    period_days = max(days + 5, 7)
    tk = yf.Ticker(ticker)

    intr = tk.history(
        period=f"{period_days}d", interval=interval,
        auto_adjust=True, actions=False, prepost=not regular_only
    )
    if intr.empty:
        return intr

    intr = intr.sort_index()
    # RTH-Filter nach Börse
    prof = infer_exchange_profile(ticker)
    intr = apply_rth_filter(intr, prof["tz"], prof["sessions"]) if regular_only else intr

    # Letzte N Handelstage schneiden
    if not intr.empty:
        uniq = pd.Index(intr.index.normalize().unique())
        keep = set(uniq[-days:])
        intr = intr.loc[intr.index.normalize().isin(keep)]

    # Aufräumen
    intr = intr[~intr.index.duplicated(keep='last')]
    return intr


# ─────────────────────────────────────────────────────────────
# Features, Training & Backtest  (Bars statt Tage)
# ─────────────────────────────────────────────────────────────

def slope(arr: np.ndarray) -> float:
    x = np.arange(len(arr))
    return np.polyfit(x, arr, 1)[0] if len(arr) >= 2 else 0.0


def make_features(df: pd.DataFrame, lookback_bars: int, horizon_bars: int) -> pd.DataFrame:
    feat = df.copy()
    feat["Range"]     = feat["High"].rolling(lookback_bars).max() - feat["Low"].rolling(lookback_bars).min()
    feat["SlopeHigh"] = feat["High"].rolling(lookback_bars).apply(slope, raw=True)
    feat["SlopeLow"]  = feat["Low"].rolling(lookback_bars).apply(slope, raw=True)
    feat = feat.iloc[max(lookback_bars-1,0):].copy()
    feat["FutureRet"] = feat["Close"].shift(-horizon_bars) / feat["Close"] - 1
    return feat


def backtest_next_bar(
    df: pd.DataFrame,
    entry_thr: float,
    exit_thr: float,
    commission: float,
    slippage_bps: int,
    init_cap: float,
    pos_frac: float,
    min_hold_bars: int = 0,
    cooldown_bars: int = 0,
) -> Tuple[pd.DataFrame, List[dict]]:
    df = df.copy()
    n = len(df)
    if n < 2:
        raise ValueError("Zu wenige Datenpunkte für Backtest.")

    cash_gross = init_cap
    cash_net   = init_cap
    shares     = 0.0
    in_pos     = False

    cost_basis_gross = 0.0
    cost_basis_net   = 0.0

    last_entry_idx: Optional[int] = None
    last_exit_idx:  Optional[int] = None

    equity_gross, equity_net, trades = [], [], []
    cum_pl_net = 0.0

    for i in range(n):
        if i > 0:
            open_today = float(df["Open"].iloc[i])
            slip_buy  = open_today * (1 + slippage_bps / 10000.0)
            slip_sell = open_today * (1 - slippage_bps / 10000.0)
            prob_prev = float(df["SignalProb"].iloc[i-1])
            date_exec = df.index[i]

            cool_ok = True
            if (not in_pos) and cooldown_bars > 0 and last_exit_idx is not None:
                bars_since_exit = i - last_exit_idx
                cool_ok = bars_since_exit >= int(cooldown_bars)

            # ENTRY
            can_enter = (not in_pos) and (prob_prev > entry_thr) and cool_ok
            if can_enter:
                invest_net   = cash_net * pos_frac
                fee_entry    = invest_net * commission
                target_shares = max((invest_net - fee_entry) / slip_buy, 0.0)

                if target_shares > 0 and (target_shares * slip_buy + fee_entry) <= cash_net + 1e-9:
                    shares = target_shares
                    cost_basis_gross = shares * slip_buy
                    cost_basis_net   = shares * slip_buy + fee_entry
                    cash_gross -= cost_basis_gross
                    cash_net   -= cost_basis_net
                    in_pos = True
                    last_entry_idx = i
                    trades.append({
                        "Date": date_exec, "Typ": "Entry", "Price": round(slip_buy, 6),
                        "Shares": round(shares, 4), "Gross P&L": 0.0,
                        "Fees": round(fee_entry, 2), "Net P&L": 0.0,
                        "kum P&L": round(cum_pl_net, 2), "Prob": round(prob_prev, 4),
                        "HoldBars": np.nan
                    })

            # EXIT
            elif in_pos and prob_prev < exit_thr:
                held_bars = (i - last_entry_idx) if last_entry_idx is not None else 0
                if int(min_hold_bars) > 0 and held_bars < int(min_hold_bars):
                    pass
                else:
                    gross_value = shares * slip_sell
                    fee_exit    = gross_value * commission
                    pnl_gross   = gross_value - cost_basis_gross
                    pnl_net     = (gross_value - fee_exit) - cost_basis_net

                    cash_gross += gross_value
                    cash_net   += (gross_value - fee_exit)

                    in_pos = False
                    shares = 0.0
                    cost_basis_gross = 0.0
                    cost_basis_net   = 0.0

                    cum_pl_net += pnl_net
                    trades.append({
                        "Date": date_exec, "Typ": "Exit", "Price": round(slip_sell, 6),
                        "Shares": 0.0, "Gross P&L": round(pnl_gross, 2),
                        "Fees": round(fee_exit, 2), "Net P&L": round(pnl_net, 2),
                        "kum P&L": round(cum_pl_net, 2), "Prob": round(prob_prev, 4),
                        "HoldBars": int(held_bars)
                    })
                    last_exit_idx = i
                    last_entry_idx = None

        close_today = float(df["Close"].iloc[i])
        equity_gross.append(cash_gross + (shares * close_today if in_pos else 0.0))
        equity_net.append(cash_net + (shares * close_today if in_pos else 0.0))

    df_bt = df.copy()
    df_bt["Equity_Gross"] = equity_gross
    df_bt["Equity_Net"]   = equity_net
    return df_bt, trades


def _cagr_from_path(values: pd.Series, bars_per_year: float) -> float:
    if len(values) < 2 or not np.isfinite(bars_per_year) or bars_per_year <= 0:
        return np.nan
    years = len(values) / bars_per_year
    return (values.iloc[-1] / values.iloc[0]) ** (1/years) - 1 if years > 0 else np.nan


def _sortino(rets: pd.Series, ann_factor: float) -> float:
    if rets.empty or not np.isfinite(ann_factor) or ann_factor <= 0:
        return np.nan
    mean = rets.mean() * ann_factor
    downside = rets[rets < 0]
    dd = downside.std() * np.sqrt(ann_factor) if len(downside) else np.nan
    return mean / dd if dd and np.isfinite(dd) and dd > 0 else np.nan


def infer_bars_per_year(idxf: pd.DatetimeIndex) -> float:
    if len(idxf) < 10:
        return 252.0
    # Bars/Tag via Median ermitteln
    df = pd.DataFrame(index=idxf)
    by_day = df.groupby(idxf.normalize()).size()
    bars_per_day = float(by_day.median()) if not by_day.empty else np.nan
    return (bars_per_day * 252.0) if np.isfinite(bars_per_day) else 252.0


def compute_performance(df_bt: pd.DataFrame, trades: List[dict], init_cap: float) -> dict:
    net_ret = (df_bt["Equity_Net"].iloc[-1] / init_cap - 1) * 100
    rets = df_bt["Equity_Net"].pct_change().dropna()

    ann = infer_bars_per_year(df_bt.index)
    vol_ann = rets.std() * np.sqrt(ann) * 100
    sharpe = (rets.mean() * np.sqrt(ann)) / (rets.std() + 1e-12)
    dd = (df_bt["Equity_Net"] - df_bt["Equity_Net"].cummax()) / df_bt["Equity_Net"].cummax()
    max_dd = dd.min() * 100
    calmar = (net_ret/100) / abs(max_dd/100) if max_dd < 0 else np.nan
    gross_ret = (df_bt["Equity_Gross"].iloc[-1] / init_cap - 1) * 100
    bh_ret = (df_bt["Close"].iloc[-1] / df_bt["Close"].iloc[0] - 1) * 100
    fees = sum(t.get("Fees",0.0) for t in trades)
    phase = "Open" if trades and trades[-1]["Typ"] == "Entry" else "Flat"
    completed = sum(1 for t in trades if t["Typ"] == "Exit")
    net_eur = df_bt["Equity_Net"].iloc[-1] - init_cap
    cagr = _cagr_from_path(df_bt["Equity_Net"], ann)
    sortino = _sortino(rets, ann)

    return {
        "Strategy Net (%)": round(net_ret, 2),
        "Strategy Gross (%)": round(gross_ret, 2),
        "Buy & Hold Net (%)": round(bh_ret, 2),
        "Volatility (%)": round(vol_ann, 2),
        "Sharpe-Ratio": round(sharpe, 2),
        "Sortino-Ratio": round(sortino, 2) if np.isfinite(sortino) else np.nan,
        "Max Drawdown (%)": round(max_dd, 2),
        "Calmar-Ratio": round(calmar, 2) if np.isfinite(calmar) else np.nan,
        "Fees (€)": round(fees, 2),
        "Phase": phase,
        "Number of Trades": int(completed),
        "Net P&L (€)": round(net_eur, 2),
        "CAGR (%)": round(100*(cagr if np.isfinite(cagr) else np.nan), 2),
    }


def make_features_and_train_intraday(
    df: pd.DataFrame,
    lookback_bars: int,
    horizon_bars: int,
    threshold: float,
    model_params: dict,
    entry_prob: float,
    exit_prob: float,
    min_hold_bars: int = 0,
    cooldown_bars: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[dict], dict]:
    feat = make_features(df, lookback_bars, horizon_bars)

    hist = feat.iloc[:-1].dropna(subset=["FutureRet"]).copy()
    if len(hist) < max(60, lookback_bars + horizon_bars + 10):
        raise ValueError("Zu wenige Intraday-Balken für Training.")

    hist["Target"] = (hist["FutureRet"] > threshold).astype(int)

    pos = int(hist["Target"].sum())
    neg = int(len(hist) - pos)
    if pos == 0 or neg == 0:
        feat["SignalProb"] = 0.5
        feat_bt = feat.iloc[:-1].copy()
        df_bt, trades = backtest_next_bar(
            feat_bt, entry_prob, exit_prob, COMMISSION, SLIPPAGE_BPS,
            INIT_CAP, POS_FRAC, min_hold_bars=int(min_hold_bars), cooldown_bars=int(cooldown_bars)
        )
        metrics = compute_performance(df_bt, trades, INIT_CAP)
        metrics["Note"] = "Training übersprungen: Target nur eine Klasse; P=0.5 genutzt."
        return feat, df_bt, trades, metrics

    X_cols = ["Range","SlopeHigh","SlopeLow"]
    X_train, y_train = hist[X_cols].values, hist["Target"].values
    scaler = StandardScaler().fit(X_train)
    model  = GradientBoostingClassifier(**model_params).fit(scaler.transform(X_train), y_train)
    feat["SignalProb"] = model.predict_proba(scaler.transform(feat[X_cols].values))[:,1]

    feat_bt = feat.iloc[:-1].copy()
    df_bt, trades = backtest_next_bar(
        feat_bt, entry_prob, exit_prob, COMMISSION, SLIPPAGE_BPS,
        INIT_CAP, POS_FRAC, min_hold_bars=int(min_hold_bars), cooldown_bars=int(cooldown_bars)
    )
    metrics = compute_performance(df_bt, trades, INIT_CAP)
    return feat, df_bt, trades, metrics


# ─────────────────────────────────────────────────────────────
# Sidebar – Controls
# ─────────────────────────────────────────────────────────────
st.sidebar.header("Parameter")

# Tickers
_tks_in = st.sidebar.text_input(
    "Tickers (Komma-getrennt)", value="BABA, VOW3.DE, INTC, BIDU, LUMN"
)
TICKERS = _normalize_tickers([t for t in _tks_in.split(',') if t.strip()])
if not TICKERS:
    st.stop()

# Datenmodus
mode = st.sidebar.selectbox("Daten-Frequenz", ["Intraday (10 Handelstage)", "Daily"], index=0)

# Intraday Settings
INTRA_DAYS = st.sidebar.slider("Intraday Lookback (Handelstage)", 2, 30, 10, step=1)
INTRA_INTERVAL = st.sidebar.selectbox("Intraday-Intervall", ["1m","2m","5m","15m"], index=2)
if INTRA_INTERVAL == "1m" and INTRA_DAYS > 7:
    st.sidebar.warning("Yahoo begrenzt 1m ~7 Tage. Für 10 Tage 2m/5m nutzen.")

# Modellparameter
LOOKBACK_BARS = st.sidebar.number_input("Lookback (Bars)", 10, 5_000, 120, step=10)
HORIZON_BARS  = st.sidebar.number_input("Horizon (Bars)", 1, 1_000, 20, step=1)
THRESH   = st.sidebar.number_input("Threshold für Target", 0.0, 0.10, 0.02, step=0.005, format="%.3f")
ENTRY_PROB = st.sidebar.slider("Entry Threshold (P)", 0.0, 1.0, 0.63, step=0.01)
EXIT_PROB  = st.sidebar.slider("Exit Threshold (P)",  0.0, 1.0, 0.46, step=0.01)
if EXIT_PROB >= ENTRY_PROB:
    st.sidebar.error("Exit-Threshold muss unter Entry-Threshold liegen.")
    st.stop()

# Trading-Kosten & Sizing
MIN_HOLD_BARS = st.sidebar.number_input("Mindesthaltedauer (Bars)", 0, 10000, 12, step=1)
COOLDOWN_BARS = st.sidebar.number_input("Cooling Phase (Bars)", 0, 10000, 6, step=1)
COMMISSION   = st.sidebar.number_input("Commission (ad valorem)", 0.0, 0.02, 0.0010, step=0.0001, format="%.4f")
SLIPPAGE_BPS = st.sidebar.number_input("Slippage (bp je Ausführung)", 0, 100, 2, step=1)
POS_FRAC     = st.sidebar.slider("Positionsgröße (% des Kapitals)", 0.1, 1.0, 1.0, step=0.1)
INIT_CAP     = st.sidebar.number_input("Initial Capital  (€)", min_value=1000.0, value=10_000.0, step=1000.0, format="%.2f")

# Modell (GBM)
st.sidebar.markdown("**Modellparameter**")
n_estimators  = st.sidebar.number_input("n_estimators",  10, 500, 120, step=10)
learning_rate = st.sidebar.number_input("learning_rate", 0.01, 1.0, 0.10, step=0.01, format="%.2f")
max_depth     = st.sidebar.number_input("max_depth",     1, 10, 3, step=1)
MODEL_PARAMS = dict(n_estimators=int(n_estimators), learning_rate=float(learning_rate),
                    max_depth=int(max_depth), random_state=42)

c1, c2 = st.sidebar.columns(2)
if c1.button("🔄 Cache leeren"):
    st.cache_data.clear(); st.rerun()

# ─────────────────────────────────────────────────────────────
# Haupt – Pipeline
# ─────────────────────────────────────────────────────────────
st.markdown("<h1 style='font-size: 32px;'>📈 AI NEXTLEVEL • Intraday 10d (RTH)</h1>", unsafe_allow_html=True)

price_map: Dict[str, pd.DataFrame] = {}
with st.spinner("Lade Intraday-Daten …"):
    for tk in TICKERS:
        try:
            df = get_intraday_past_n_days(tk, days=int(INTRA_DAYS), interval=INTRA_INTERVAL, regular_only=True)
            if df is None or df.empty:
                st.warning(f"Keine Intraday-Daten für {tk} ({INTRA_INTERVAL}).")
                continue
            # Sicherstellen, dass Spalten vollständig sind
            need = ["Open","High","Low","Close"]
            if not set(need).issubset(df.columns):
                raise ValueError(f"OHLC unvollständig für {tk}")
            price_map[tk] = df
        except Exception as e:
            st.error(f"Fehler beim Laden von {tk}: {e}")

if not price_map:
    st.stop()

results = []
all_trades: Dict[str, List[dict]] = {}
all_feat:  Dict[str, pd.DataFrame] = {}
all_bt:    Dict[str, pd.DataFrame] = {}

for ticker, df in price_map.items():
    with st.expander(f"🔍 Analyse {ticker} – {get_ticker_name(ticker)}", expanded=False):
        try:
            feat, df_bt, trades, metrics = make_features_and_train_intraday(
                df, int(LOOKBACK_BARS), int(HORIZON_BARS), float(THRESH), MODEL_PARAMS,
                float(ENTRY_PROB), float(EXIT_PROB),
                min_hold_bars=int(MIN_HOLD_BARS), cooldown_bars=int(COOLDOWN_BARS)
            )
            metrics["Ticker"] = ticker
            results.append(metrics)
            all_trades[ticker] = trades
            all_feat[ticker] = feat
            all_bt[ticker] = df_bt

            # KPI Tiles
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("Strategie Netto (%)", f"{metrics['Strategy Net (%)']:.2f}")
            c2.metric("Buy & Hold (%)",      f"{metrics['Buy & Hold Net (%)']:.2f}")
            c3.metric("Sharpe",               f"{metrics['Sharpe-Ratio']:.2f}")
            c4.metric("Sortino",              f"{metrics['Sortino-Ratio']:.2f}" if np.isfinite(metrics["Sortino-Ratio"]) else "–")
            c5.metric("Max DD (%)",           f"{metrics['Max Drawdown (%)']:.2f}")
            c6.metric("Trades (RT)",          f"{int(metrics['Number of Trades'])}")

            # Preis + Signale (Intraday, farbcodiert)
            df_plot = feat.copy()
            price_fig = go.Figure()
            price_fig.add_trace(go.Scatter(
                x=df_plot.index, y=df_plot["Close"], mode="lines", name="Close",
                line=dict(color="rgba(0,0,0,0.35)", width=1),
                hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Close: %{y:.4f}<extra></extra>"
            ))
            signal_probs = df_plot["SignalProb"]
            norm = (signal_probs - signal_probs.min()) / (signal_probs.max() - signal_probs.min() + 1e-9)
            for i in range(len(df_plot) - 1):
                seg_x = df_plot.index[i:i+2]
                seg_y = df_plot["Close"].iloc[i:i+2]
                color_seg = px.colors.sample_colorscale(px.colors.diverging.RdYlGn, float(norm.iloc[i]))[0]
                price_fig.add_trace(go.Scatter(x=seg_x, y=seg_y, mode="lines", showlegend=False,
                                               line=dict(color=color_seg, width=2), hoverinfo="skip"))
            trades_df = pd.DataFrame(trades)
            if not trades_df.empty:
                trades_df["Date"] = pd.to_datetime(trades_df["Date"])
                entries = trades_df[trades_df["Typ"]=="Entry"]; exits = trades_df[trades_df["Typ"]=="Exit"]
                price_fig.add_trace(go.Scatter(
                    x=entries["Date"], y=entries["Price"], mode="markers", name="Entry",
                    marker_symbol="triangle-up", marker=dict(size=10, color="green"),
                    hovertemplate="Entry<br>%{x|%Y-%m-%d %H:%M}<br>%{y:.4f}<extra></extra>"
                ))
                price_fig.add_trace(go.Scatter(
                    x=exits["Date"], y=exits["Price"], mode="markers", name="Exit",
                    marker_symbol="triangle-down", marker=dict(size=10, color="red"),
                    hovertemplate="Exit<br>%{x|%Y-%m-%d %H:%M}<br>%{y:.4f}<extra></extra>"
                ))
            price_fig.update_layout(
                title=f"{ticker}: Intraday Close + Signal-Wahrscheinlichkeit",
                xaxis_title="Zeit", yaxis_title="Preis",
                height=420, margin=dict(t=50, b=30, l=40, r=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(price_fig, use_container_width=True)

            # Equity-Kurve vs. Buy&Hold
            eq = go.Figure()
            eq.add_trace(go.Scatter(x=df_bt.index, y=df_bt["Equity_Net"], name="Strategy Net Equity",
                        mode="lines", hovertemplate="%{x|%Y-%m-%d %H:%M}: %{y:.2f}€<extra></extra>"))
            bh_curve = INIT_CAP * df_bt["Close"] / df_bt["Close"].iloc[0]
            eq.add_trace(go.Scatter(x=df_bt.index, y=bh_curve, name="Buy & Hold", mode="lines",
                                    line=dict(dash="dash", color="black")))
            eq.update_layout(title=f"{ticker}: Net Equity vs. Buy & Hold", xaxis_title="Zeit", yaxis_title="Equity (€)",
                             height=380, margin=dict(t=50, b=30, l=40, r=20),
                             legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(eq, use_container_width=True)

            # Trades-Tabelle
            with st.expander(f"Trades (Next Bar) • {ticker}", expanded=False):
                if not trades_df.empty:
                    df_tr = trades_df.copy()
                    df_tr["Ticker"] = ticker
                    df_tr["Name"] = get_ticker_name(ticker)
                    df_tr["DateStr"] = pd.to_datetime(df_tr["Date"]).dt.strftime("%d.%m.%Y %H:%M")
                    df_tr["CumPnL"] = (
                        df_tr.where(df_tr["Typ"] == "Exit")["Net P&L"].cumsum().fillna(method="ffill").fillna(0)
                    )
                    df_tr = df_tr.rename(columns={"Net P&L":"PnL","Prob":"Signal Prob","HoldBars":"Hold (bars)"})
                    disp_cols = ["Ticker","Name","DateStr","Typ","Price","Shares","Signal Prob","Hold (bars)","PnL","CumPnL","Fees"]
                    st.dataframe(df_tr[disp_cols], use_container_width=True)
                    st.download_button(
                        label=f"Trades {ticker} als CSV",
                        data=to_csv_eu(df_tr[["Ticker","Name","Date","Typ","Price","Shares","Signal Prob","Hold (bars)","PnL","CumPnL","Fees"]], float_format="%.6f"),
                        file_name=f"trades_{ticker}_intraday.csv", mime="text/csv",
                    )
                else:
                    st.info("Keine Trades vorhanden.")

        except Exception as e:
            st.error(f"Fehler bei {ticker}: {e}")

# ─────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────
if results:
    summary_df = pd.DataFrame(results).set_index("Ticker")

    st.subheader("📊 Summary aller Ticker (Intraday • Next Bar)")
    cols = st.columns(4)
    total_net_pnl   = summary_df["Net P&L (€)"].sum()
    total_fees      = summary_df["Fees (€)"].sum()
    total_gross_pnl = float(total_net_pnl + total_fees)
    total_trades    = int(summary_df["Number of Trades"].sum())
    total_capital   = INIT_CAP * len(summary_df)

    cols[0].metric("Cumulative Net P&L (€)",  f"{total_net_pnl:,.2f}")
    cols[1].metric("Trading Costs (€)",       f"{total_fees:,.2f}")
    cols[2].metric("Cumulative Gross P&L (€)",f"{total_gross_pnl:,.2f}")
    cols[3].metric("Round-Trips",             f"{total_trades}")

    styled = (
        summary_df.style
        .format({
            "Strategy Net (%)":"{:.2f}","Strategy Gross (%)":"{:.2f}",
            "Buy & Hold Net (%)":"{:.2f}","Volatility (%)":"{:.2f}",
            "Sharpe-Ratio":"{:.2f}","Sortino-Ratio":"{:.2f}",
            "Max Drawdown (%)":"{:.2f}","Calmar-Ratio":"{:.2f}",
            "Fees (€)":"{:.2f}","Net P&L (€)":"{:.2f}","CAGR (%)":"{:.2f}"
        })
    )
    st.dataframe(summary_df, use_container_width=True)
    st.download_button(
        "Summary als CSV",
        to_csv_eu(summary_df.reset_index()),
        file_name="strategy_summary_intraday.csv", mime="text/csv"
    )
else:
    st.warning("Keine Ergebnisse. Prüfe Ticker/Intervall/Lookback.")
