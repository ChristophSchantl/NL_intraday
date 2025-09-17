# streamlit_app.py
# -*- coding: utf-8 -*-

"""
SHI – STOCK CHECK • Intraday 10d (RTH only)
- Nur Intraday-Analyse, strikt letzte 10 Handelstage.
- CSV/Upload, Backtest (Next Bar), KPIs, Trades, Summary, Round-Trips, Histogramme, Korrelation.
"""

# ─────────────────────────────────────────────────────────────
# Imports & Global Config
# ─────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*figure layout has changed to tight.*")

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import time
from typing import Tuple, List, Dict, Optional
from zoneinfo import ZoneInfo
from functools import lru_cache

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="SHI – STOCK CHECK • Intraday 10d (RTH)", layout="wide")
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

def parse_ticker_csv(path_or_buffer) -> List[str]:
    try:
        df = pd.read_csv(path_or_buffer)
    except Exception:
        df = pd.read_csv(path_or_buffer, sep=";")
    if df.empty:
        return []
    cols_lower = {c.lower(): c for c in df.columns}
    for key in ("ticker", "symbol", "symbols", "isin", "code"):
        if key in cols_lower:
            col = cols_lower[key]
            return _normalize_tickers(df[col].astype(str).tolist())
    first = df.columns[0]
    return _normalize_tickers(df[first].astype(str).tolist())

def show_styled_or_plain(df: pd.DataFrame, styler):
    try:
        html = getattr(styler, "to_html", None)
        if callable(html):
            st.markdown(html(), unsafe_allow_html=True)
        else:
            raise AttributeError("Styler ohne to_html()")
    except Exception as e:
        st.warning(f"Styled-Tabelle nicht renderbar, Fallback DataFrame. ({e})")
        st.dataframe(df, use_container_width=True)

def slope(arr: np.ndarray) -> float:
    x = np.arange(len(arr))
    return np.polyfit(x, arr, 1)[0] if len(arr) >= 2 else 0.0

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
# Intraday: Börsenprofile + RTH-Filter
# ─────────────────────────────────────────────────────────────
_EX_SUFFIX_MAP: Dict[str, Dict] = {
    ".DE": {"tz": "Europe/Berlin",  "sessions": [("09:00","17:30")]},
    ".F":  {"tz": "Europe/Berlin",  "sessions": [("09:00","17:30")]},
    ".SW": {"tz": "Europe/Zurich",  "sessions": [("09:00","17:30")]},
    ".PA": {"tz": "Europe/Paris",   "sessions": [("09:00","17:30")]},
    ".AS": {"tz": "Europe/Amsterdam","sessions": [("09:00","17:30")]},
    ".MI": {"tz": "Europe/Rome",    "sessions": [("09:00","17:30")]},
    ".L":  {"tz": "Europe/London",  "sessions": [("08:00","16:30")]},
    ".IR": {"tz": "Europe/Dublin",  "sessions": [("08:00","16:30")]},
    ".TO": {"tz": "America/Toronto","sessions": [("09:30","16:00")]},
    ".V":  {"tz": "America/Toronto","sessions": [("09:30","16:00")]},
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
    sx = _ticker_suffix(ticker)
    if sx and sx in _EX_SUFFIX_MAP:
        return _EX_SUFFIX_MAP[sx]
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
    df = df.copy(); df.index = local
    mask = np.zeros(len(df), dtype=bool)
    times = df.index.time
    for o, c in sessions:
        t1, t2 = _to_time(o), _to_time(c)
        mask |= (times >= t1) & (times <= t2)
    df = df.loc[mask]
    df.index = df.index.tz_convert(LOCAL_TZ)
    return df

# ─────────────────────────────────────────────────────────────
# Intraday Loader (fix 10 Handelstage)
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=180)
def get_intraday_past_n_days(ticker: str, interval: str = "5m", days: int = 10, regular_only: bool = True) -> pd.DataFrame:
    tk = yf.Ticker(ticker)
    period_days = max(days + 5, 7)
    intr = tk.history(period=f"{period_days}d", interval=interval,
                      auto_adjust=True, actions=False, prepost=not regular_only)
    if intr.empty:
        return intr
    intr = intr.sort_index()
    prof = infer_exchange_profile(ticker)
    intr = apply_rth_filter(intr, prof["tz"], prof["sessions"]) if regular_only else intr
    uniq = pd.Index(intr.index.normalize().unique())
    keep = set(uniq[-days:])
    intr = intr.loc[intr.index.normalize().isin(keep)]
    intr = intr[~intr.index.duplicated(keep='last')]
    return intr

# ─────────────────────────────────────────────────────────────
# Features, Training & Backtest (Next Bar)
# ─────────────────────────────────────────────────────────────
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
):
    df = df.copy()
    n = len(df)
    if n < 2:
        raise ValueError("Zu wenige Datenpunkte für Backtest.")

    cash_gross = init_cap; cash_net = init_cap
    shares = 0.0; in_pos = False
    cost_basis_gross = 0.0; cost_basis_net = 0.0
    last_entry_idx = None; last_exit_idx = None

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
                cool_ok = (i - last_exit_idx) >= int(cooldown_bars)

            # Entry
            if (not in_pos) and (prob_prev > entry_thr) and cool_ok:
                invest_net = cash_net * pos_frac
                fee_entry  = invest_net * commission
                target_shares = max((invest_net - fee_entry) / slip_buy, 0.0)
                if target_shares > 0 and (target_shares * slip_buy + fee_entry) <= cash_net + 1e-9:
                    shares = target_shares
                    cost_basis_gross = shares * slip_buy
                    cost_basis_net   = shares * slip_buy + fee_entry
                    cash_gross -= cost_basis_gross; cash_net -= cost_basis_net
                    in_pos = True; last_entry_idx = i
                    trades.append({"Date": date_exec, "Typ": "Entry", "Price": round(slip_buy, 6),
                                   "Shares": round(shares, 4), "Gross P&L": 0.0,
                                   "Fees": round(fee_entry, 2), "Net P&L": 0.0,
                                   "kum P&L": round(cum_pl_net, 2), "Prob": round(prob_prev, 4),
                                   "HoldBars": np.nan})
            # Exit
            elif in_pos and prob_prev < exit_thr:
                held_bars = (i - last_entry_idx) if last_entry_idx is not None else 0
                if int(min_hold_bars) > 0 and held_bars < int(min_hold_bars):
                    pass
                else:
                    gross_value = shares * slip_sell
                    fee_exit    = gross_value * commission
                    pnl_gross   = gross_value - cost_basis_gross
                    pnl_net     = (gross_value - fee_exit) - cost_basis_net
                    cash_gross += gross_value; cash_net += (gross_value - fee_exit)
                    in_pos = False; shares = 0.0; cost_basis_gross = 0.0; cost_basis_net = 0.0
                    cum_pl_net += pnl_net
                    trades.append({"Date": date_exec, "Typ": "Exit", "Price": round(slip_sell, 6),
                                   "Shares": 0.0, "Gross P&L": round(pnl_gross, 2),
                                   "Fees": round(fee_exit, 2), "Net P&L": round(pnl_net, 2),
                                   "kum P&L": round(cum_pl_net, 2), "Prob": round(prob_prev, 4),
                                   "HoldBars": int(held_bars)})
                    last_exit_idx = i; last_entry_idx = None

        close_today = float(df["Close"].iloc[i])
        equity_gross.append(cash_gross + (shares * close_today if in_pos else 0.0))
        equity_net.append(cash_net + (shares * close_today if in_pos else 0.0))

    df_bt = df.copy()
    df_bt["Equity_Gross"] = equity_gross
    df_bt["Equity_Net"]   = equity_net
    return df_bt, trades

def infer_bars_per_year(idxf: pd.DatetimeIndex) -> float:
    if len(idxf) < 10:
        return 252.0
    df = pd.DataFrame(index=idxf)
    by_day = df.groupby(idxf.normalize()).size()
    bars_per_day = float(by_day.median()) if not by_day.empty else np.nan
    return (bars_per_day * 252.0) if np.isfinite(bars_per_day) else 252.0

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

def _winrate_roundtrips(trades: List[dict]) -> float:
    if not trades:
        return np.nan
    pnl = []
    entry = None
    for ev in trades:
        if ev["Typ"] == "Entry":
            entry = ev
        elif ev["Typ"] == "Exit" and entry is not None:
            pnl.append(float(ev.get("Net P&L", 0.0))); entry = None
    if not pnl:
        return np.nan
    pnl = np.array(pnl, dtype=float)
    return (pnl > 0).mean()

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
    winrate = _winrate_roundtrips(trades)
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
        "Winrate (%)": round(100*(winrate if np.isfinite(winrate) else np.nan), 2),
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
):
    feat = make_features(df, lookback_bars, horizon_bars)
    hist = feat.iloc[:-1].dropna(subset=["FutureRet"]).copy()
    if len(hist) < max(60, lookback_bars + horizon_bars + 10):
        raise ValueError("Zu wenige Intraday-Balken für Training.")
    hist["Target"] = (hist["FutureRet"] > threshold).astype(int)

    pos = int(hist["Target"].sum()); neg = int(len(hist) - pos)
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

def compute_round_trips(all_trades: Dict[str, List[dict]]) -> pd.DataFrame:
    rows = []
    for tk, tr in all_trades.items():
        name = get_ticker_name(tk)
        current_entry = None
        for ev in tr:
            if ev["Typ"] == "Entry":
                current_entry = ev
            elif ev["Typ"] == "Exit" and current_entry is not None:
                entry_date = pd.to_datetime(current_entry["Date"])
                exit_date  = pd.to_datetime(ev["Date"])
                shares     = float(current_entry.get("Shares", 0.0))
                entry_p    = float(current_entry.get("Price", np.nan))
                exit_p     = float(ev.get("Price", np.nan))
                fee_e      = float(current_entry.get("Fees", 0.0))
                fee_x      = float(ev.get("Fees", 0.0))
                pnl_net    = float(ev.get("Net P&L", 0.0))
                cost_net   = shares * entry_p + fee_e
                ret_pct    = (pnl_net / cost_net * 100.0) if cost_net else np.nan
                hold_bars  = ev.get("HoldBars", np.nan)
                rows.append({
                    "Ticker": tk, "Name": name,
                    "Entry Date": entry_date, "Exit Date": exit_date,
                    "Hold (bars)": hold_bars if pd.notna(hold_bars) else np.nan,
                    "Entry Prob": current_entry.get("Prob", np.nan),
                    "Exit Prob":  ev.get("Prob", np.nan),
                    "Shares": round(shares, 4),
                    "Entry Price": round(entry_p, 6), "Exit Price": round(exit_p, 6),
                    "PnL Net (€)": round(pnl_net, 2), "Fees (€)": round(fee_e + fee_x, 2),
                    "Return (%)": round(ret_pct, 2),
                })
                current_entry = None
    return pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────
# Sidebar – Controls (Intraday only)
# ─────────────────────────────────────────────────────────────
st.sidebar.header("Parameter")

ticker_source = st.sidebar.selectbox("Ticker-Quelle", ["Manuell (Textfeld)", "CSV-Upload"], index=0)
tickers_final: List[str] = []
if ticker_source == "Manuell (Textfeld)":
    tickers_input = st.sidebar.text_input("Tickers (Komma-getrennt)", value="BABA, VOW3.DE, INTC, BIDU, LUMN")
    tickers_final = _normalize_tickers([t for t in tickers_input.split(",") if t.strip()])
else:
    st.sidebar.caption("CSV(s) mit Spalte **ticker** oder erste Spalte.")
    uploads = st.sidebar.file_uploader("CSV-Dateien", type=["csv"], accept_multiple_files=True)
    collected = []
    if uploads:
        for up in uploads:
            try:
                collected += parse_ticker_csv(up)
            except Exception as e:
                st.sidebar.error(f"Fehler beim Lesen von '{up.name}': {e}")
    base = _normalize_tickers(collected)
    extra_csv = st.sidebar.text_input("Weitere Ticker manuell (Komma)", value="", key="extra_csv")
    extras = _normalize_tickers([t for t in extra_csv.split(",") if t.strip()]) if extra_csv else []
    tickers_final = _normalize_tickers(base + extras)
    if tickers_final:
        st.sidebar.caption(f"Gefundene Ticker: {len(tickers_final)}")
        tickers_final = st.sidebar.multiselect("Auswahl verfeinern", options=tickers_final, default=tickers_final)

if not tickers_final:
    tickers_final = _normalize_tickers(["BABA", "VOW3.DE", "INTC", "BIDU", "LUMN"])

st.sidebar.download_button("Kombinierte Ticker als CSV",
    to_csv_eu(pd.DataFrame({"ticker": tickers_final})),
    file_name="tickers_combined.csv", mime="text/csv"
)
TICKERS = tickers_final

# Fix: immer exakt 10 Handelstage; 1m nicht anbieten
INTRA_DAYS = 10
INTRA_INTERVAL = st.sidebar.selectbox("Intraday-Intervall", ["2m","5m","15m"], index=1)

# Signal-Parameter
LOOKBACK_BARS = st.sidebar.number_input("Lookback (Bars)", 10, 5000, 120, step=10)
HORIZON_BARS  = st.sidebar.number_input("Horizon (Bars)", 1, 1000, 20, step=1)
THRESH        = st.sidebar.number_input("Threshold für Target", 0.0, 0.10, 0.02, step=0.005, format="%.3f")
ENTRY_PROB    = st.sidebar.slider("Entry Threshold (P)", 0.0, 1.0, 0.63, step=0.01)
EXIT_PROB     = st.sidebar.slider("Exit Threshold (P)",  0.0, 1.0, 0.46, step=0.01)
if EXIT_PROB >= ENTRY_PROB:
    st.sidebar.error("Exit-Threshold muss unter Entry-Threshold liegen."); st.stop()

MIN_HOLD_BARS = st.sidebar.number_input("Mindesthaltedauer (Bars)", 0, 10000, 12, step=1)
COOLDOWN_BARS = st.sidebar.number_input("Cooling Phase (Bars)", 0, 10000, 6, step=1)

COMMISSION   = st.sidebar.number_input("Commission (ad valorem)", 0.0, 0.02, 0.0010, step=0.0001, format="%.4f")
SLIPPAGE_BPS = st.sidebar.number_input("Slippage (bp je Ausführung)", 0, 100, 2, step=1)
POS_FRAC     = st.sidebar.slider("Positionsgröße (% des Kapitals)", 0.1, 1.0, 1.0, step=0.1)
INIT_CAP     = st.sidebar.number_input("Initial Capital  (€)", min_value=1000.0, value=10_000.0, step=1000.0, format="%.2f")

st.sidebar.markdown("**Modellparameter**")
n_estimators  = st.sidebar.number_input("n_estimators",  10, 500, 120, step=10)
learning_rate = st.sidebar.number_input("learning_rate", 0.01, 1.0, 0.10, step=0.01, format="%.2f")
max_depth     = st.sidebar.number_input("max_depth",     1, 10, 3, step=1)
MODEL_PARAMS = dict(n_estimators=int(n_estimators), learning_rate=float(learning_rate),
                    max_depth=int(max_depth), random_state=42)

c1, c2 = st.sidebar.columns(2)
if c1.button("🔄 Cache leeren"):
    st.cache_data.clear(); st.rerun()
if c2.button("📥 Summary CSV laden"):
    st.experimental_set_query_params(download="summary")

# ─────────────────────────────────────────────────────────────
# Haupt – Pipeline (Intraday only)
# ─────────────────────────────────────────────────────────────
st.markdown("<h1 style='font-size: 32px;'>📈 AI NEXTLEVEL • Intraday 10d (RTH)</h1>", unsafe_allow_html=True)

price_map: Dict[str, pd.DataFrame] = {}
with st.spinner(f"Lade Intraday-Daten (RTH) für {len(TICKERS)} Ticker …"):
    for tk in TICKERS:
        try:
            df = get_intraday_past_n_days(tk, interval=INTRA_INTERVAL, days=INTRA_DAYS, regular_only=True)
            if df is None or df.empty:
                st.warning(f"Keine Intraday-Daten für {tk} ({INTRA_INTERVAL})."); continue
            need = ["Open","High","Low","Close"]
            if not set(need).subset(df.columns):
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
live_forecasts_run: List[dict] = []

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

            # Live row
            live_ts    = pd.Timestamp(feat.index[-1])
            live_prob  = float(feat["SignalProb"].iloc[-1])
            live_close = float(feat["Close"].iloc[-1])
            live_forecasts_run.append({
                "AsOf": live_ts.strftime("%Y-%m-%d %H:%M"),
                "Ticker": ticker, "Name": get_ticker_name(ticker),
                f"P(>{THRESH:.3f} in {HORIZON_BARS} bars)": round(live_prob, 4),
                "Action": ("Enter / Add" if live_prob > ENTRY_PROB else ("Exit / Reduce" if live_prob < EXIT_PROB else "Hold / No Trade")),
                "Close": round(live_close, 4), "Bar": "intraday"
            })

            # KPI Tiles
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("Strategie Netto (%)", f"{metrics['Strategy Net (%)']:.2f}")
            c2.metric("Buy & Hold (%)",      f"{metrics['Buy & Hold Net (%)']:.2f}")
            c3.metric("Sharpe",               f"{metrics['Sharpe-Ratio']:.2f}")
            c4.metric("Sortino",              f"{metrics['Sortino-Ratio']:.2f}" if np.isfinite(metrics["Sortino-Ratio"]) else "–")
            c5.metric("Max DD (%)",           f"{metrics['Max Drawdown (%)']:.2f}")
            c6.metric("Trades (RT)",          f"{int(metrics['Number of Trades'])}")

            # Preis + Signale
            df_plot = feat.copy()
            price_fig = go.Figure()
            price_fig.add_trace(go.Scatter(
                x=df_plot.index, y=df_plot["Close"], mode="lines", name="Close",
                line=dict(color="rgba(0,0,0,0.35)", width=1),
                hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Close: %{y:.4f}<extra></extra>"
            ))
            signal_probs = pd.to_numeric(df_plot["SignalProb"], errors="coerce").fillna(0.5)
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
            eq.add_trace(go.Scatter(x=df_bt.index, y=df_bt["Equity_Net"], name="Strategy Net Equity (Next Bar)",
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
                    styled = df_tr[disp_cols].rename(columns={"DateStr":"Date"}).style.format({
                        "Price":"{:.4f}","Shares":"{:.4f}","Signal Prob":"{:.4f}",
                        "PnL":"{:.2f}","CumPnL":"{:.2f}","Fees":"{:.2f}"
                    })
                    show_styled_or_plain(df_tr[disp_cols].rename(columns={"DateStr":"Date"}), styled)
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
# 🔮 Live-Forecast Board
# ─────────────────────────────────────────────────────────────
if live_forecasts_run:
    live_df = (
        pd.DataFrame(live_forecasts_run)
          .drop_duplicates(subset=["Ticker"], keep="last")
          .sort_values(["AsOf", "Ticker"])
          .reset_index(drop=True)
    )
    prob_col = [c for c in live_df.columns if c.startswith("P(")]
    prob_col = prob_col[0] if prob_col else "Prob"
    if prob_col not in live_df.columns:
        live_df[prob_col] = np.nan

    def style_live_board(df: pd.DataFrame, prob_col: str, entry_threshold: float):
        def _row_color(row):
            act = str(row.get("Action","")).lower()
            if "enter" in act: return ["background-color: #D7F3F7"] * len(row)
            if "exit"  in act: return ["background-color: #FFE8E8"] * len(row)
            try:
                if float(row.get(prob_col, np.nan)) >= float(entry_threshold):
                    return ["background-color: #E6F7FF"] * len(row)
            except Exception:
                pass
            return ["background-color: #F7F7F7"] * len(row)
        return (
            df.style
              .format({prob_col:"{:.4f}","Close":"{:.4f}"})
              .apply(_row_color, axis=1)
              .set_properties(subset=["Action"], **{"font-weight": "600"})
        )

    st.markdown("### 🟣 Live–Forecast Board – Intraday")
    styled_live = style_live_board(live_df, prob_col, ENTRY_PROB)
    show_styled_or_plain(live_df, styled_live)
    st.download_button("Live-Forecasts als CSV", to_csv_eu(live_df),
                       file_name="live_forecasts_today_intraday.csv", mime="text/csv")

# ─────────────────────────────────────────────────────────────
# Summary / Open Positions / Round-Trips / Histogramme / Korrelation
# ─────────────────────────────────────────────────────────────
if results:
    summary_df = pd.DataFrame(results).set_index("Ticker")
    summary_df["Net P&L (%)"] = (summary_df["Net P&L (€)"] / INIT_CAP) * 100

    st.subheader("📊 Summary aller Ticker (Intraday • Next Bar)")
    cols = st.columns(4)
    total_net_pnl   = summary_df["Net P&L (€)"].sum()
    total_fees      = summary_df["Fees (€)"].sum()
    total_gross_pnl = float(total_net_pnl + total_fees)
    total_trades    = int(summary_df["Number of Trades"].sum())
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
            "Fees (€)":"{:.2f}","Net P&L (%)":"{:.2f}","Net P&L (€)":"{:.2f}",
            "CAGR (%)":"{:.2f}","Winrate (%)":"{:.2f}"
        })
    )
    show_styled_or_plain(summary_df, styled)
    st.download_button(
        "Summary als CSV",
        to_csv_eu(summary_df.reset_index()),
        file_name="strategy_summary_intraday.csv", mime="text/csv"
    )

    # Open Positions
    st.subheader("📋 Open Positions (Intraday)")
    open_positions = []
    for ticker, trades in all_trades.items():
        if trades and trades[-1]["Typ"] == "Entry":
            last_entry = next(t for t in reversed(trades) if t["Typ"] == "Entry")
            entry_ts = pd.to_datetime(last_entry["Date"])
            prob = float(all_feat[ticker]["SignalProb"].iloc[-1])
            last_close = float(all_bt[ticker]["Close"].iloc[-1])
            upnl = (last_close - float(last_entry["Price"])) * float(last_entry["Shares"])
            open_positions.append({
                "Ticker": ticker, "Name": get_ticker_name(ticker),
                "Entry Date": entry_ts,
                "Entry Price": round(float(last_entry["Price"]), 6),
                "Current Prob.": round(prob, 4),
                "Unrealized PnL (€)": round(upnl, 2),
            })
    if open_positions:
        open_df = pd.DataFrame(open_positions).sort_values("Entry Date", ascending=False)
        open_df_display = open_df.copy()
        open_df_display["Entry Date"] = open_df_display["Entry Date"].dt.strftime("%Y-%m-%d %H:%M")
        styled_open = open_df_display.style.format({
            "Entry Price":"{:.4f}", "Current Prob.":"{:.4f}", "Unrealized PnL (€)":"{:.2f}",
        })
        show_styled_or_plain(open_df_display, styled_open)
        st.download_button("Offene Positionen als CSV", to_csv_eu(open_df),
                           file_name="open_positions_intraday.csv", mime="text/csv")
    else:
        st.success("Keine offenen Positionen.")

    # Round-Trips
    rt_df = compute_round_trips(all_trades)
    if not rt_df.empty:
        st.subheader("🔁 Abgeschlossene Trades (Round-Trips) – Filter")
        rt_df["Entry Date"] = pd.to_datetime(rt_df["Entry Date"])
        rt_df["Exit Date"]  = pd.to_datetime(rt_df["Exit Date"])
        for c in ["Entry Prob","Exit Prob","Return (%)","PnL Net (€)","Fees (€)","Hold (bars)"]:
            if c not in rt_df.columns: rt_df[c] = np.nan

        r_min_d, r_max_d = rt_df["Entry Date"].min().date(), rt_df["Entry Date"].max().date()
        r_ticks = sorted(rt_df["Ticker"].unique().tolist())

        def finite_minmax(series, fallback=(0.0, 1.0)):
            s = pd.to_numeric(series, errors="coerce")
            lo, hi = float(np.nanmin(s.values)), float(np.nanmax(s.values))
            if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                lo, hi = fallback
            return lo, hi

        r1, r2, r3 = st.columns([1.1, 1.1, 1.5])
        with r1:
            rt_tick_sel = st.multiselect("Ticker", options=r_ticks, default=r_ticks)
            hb_lo, hb_hi = finite_minmax(rt_df["Hold (bars)"], (0.0, 10_000.0))
            rt_hold = st.slider("Haltedauer (Bars)", float(hb_lo), float(hb_hi), (float(hb_lo), float(hb_hi)), step=1.0)
        with r2:
            rt_date = st.date_input("Zeitraum (Entry-Datum)", value=(r_min_d, r_max_d),
                                    min_value=r_min_d, max_value=r_max_d, key="rt_date")
            ep_lo, ep_hi = finite_minmax(rt_df["Entry Prob"], (0.0, 1.0))
            xp_lo, xp_hi = finite_minmax(rt_df["Exit Prob"],  (0.0, 1.0))
            rt_ep = st.slider("Entry-Prob.", 0.0, 1.0, (max(0.0, ep_lo), min(1.0, ep_hi)), step=0.01)
            rt_xp = st.slider("Exit-Prob.",  0.0, 1.0, (max(0.0, xp_lo), min(1.0, xp_hi)), step=0.01)
        with r3:
            ret_lo, ret_hi = finite_minmax(rt_df["Return (%)"], (-100.0, 200.0))
            pnl_lo, pnl_hi = finite_minmax(rt_df["PnL Net (€)"], (-INIT_CAP, INIT_CAP))
            rt_ret = st.slider("Return (%)", float(ret_lo), float(ret_hi), (float(ret_lo), float(ret_hi)), step=0.5)
            rt_pnl = st.slider("PnL Net (€)", float(pnl_lo), float(pnl_hi), (float(pnl_lo), float(pnl_hi)), step=10.0)

        rds, rde = (rt_date if isinstance(rt_date, tuple) else (r_min_d, r_max_d))
        mask_rt = (
            rt_df["Ticker"].isin(rt_tick_sel) &
            (rt_df["Entry Date"].dt.date.between(rds, rde)) &
            (pd.to_numeric(rt_df["Hold (bars)"], errors="coerce").fillna(-9e9).between(rt_hold[0], rt_hold[1])) &
            (rt_df["Entry Prob"].fillna(0.0).between(rt_ep[0], rt_ep[1])) &
            (rt_df["Exit Prob"].fillna(0.0).between(rt_xp[0], rt_xp[1])) &
            (pd.to_numeric(rt_df["Return (%)"], errors="coerce").fillna(-9e9).between(rt_ret[0], rt_ret[1])) &
            (pd.to_numeric(rt_df["PnL Net (€)"], errors="coerce").fillna(-9e9).between(rt_pnl[0], rt_pnl[1]))
        )

        rt_f = rt_df.loc[mask_rt].copy()
        rt_f_disp = rt_f.copy()
        rt_f_disp["Entry Date"] = rt_f_disp["Entry Date"].dt.strftime("%Y-%m-%d")
        rt_f_disp["Exit Date"]  = rt_f_disp["Exit Date"].dt.strftime("%Y-%m-%d")

        styled_rt = rt_f_disp.style.format({
            "Shares":"{:.4f}",
            "Entry Price":"{:.6f}","Exit Price":"{:.6f}",
            "PnL Net (€)":"{:.2f}","Fees (€)":"{:.2f}","Return (%)":"{:.2f}",
            "Entry Prob":"{:.4f}","Exit Prob":"{:.4f}","Hold (bars)":"{:.0f}"
        })
        show_styled_or_plain(rt_f_disp, styled_rt)
        st.download_button(
            "Round-Trips (gefiltert) als CSV",
            to_csv_eu(rt_f_disp),
            file_name="round_trips_filtered_intraday.csv", mime="text/csv"
        )

        # Histogramme
        st.markdown("### 📊 Verteilung der Round-Trip-Ergebnisse")
        bins = st.slider("Anzahl Bins", 10, 100, 30, step=5, key="rt_bins")

        ret = pd.to_numeric(rt_f.get("Return (%)"), errors="coerce").dropna()
        pnl = pd.to_numeric(rt_f.get("PnL Net (€)"), errors="coerce").dropna()

        def pct(x): return f"{x:.2f}%"
        cstats = st.columns(5)
        cstats[0].metric("Anzahl", f"{len(ret)}")
        cstats[1].metric("Winrate", pct(100.0 * (ret > 0).mean()) if len(ret) else "–")
        cstats[2].metric("Ø Return", pct(ret.mean()) if len(ret) else "–")
        cstats[3].metric("Median",  pct(ret.median()) if len(ret) else "–")
        cstats[4].metric("Std-Abw.", pct(ret.std()) if len(ret) else "–")

        col_h1, col_h2 = st.columns(2)
        with col_h1:
            if ret.empty:
                st.info("Keine Rendite-Werte vorhanden.")
            else:
                fig_ret = go.Figure(go.Histogram(x=ret, nbinsx=bins, marker_line_width=0))
                fig_ret.add_vline(x=0, line_dash="dash", opacity=0.5)
                fig_ret.add_vline(x=float(ret.mean()), line_dash="dot", opacity=0.9)
                fig_ret.update_layout(
                    title="Histogramm: Return (%)",
                    xaxis_title="Return (%)", yaxis_title="Häufigkeit",
                    height=360, margin=dict(t=40, l=40, r=20, b=40), showlegend=False
                )
                st.plotly_chart(fig_ret, use_container_width=True)
        with col_h2:
            if pnl.empty:
                st.info("Keine PnL-Werte vorhanden.")
            else:
                fig_pnl = go.Figure(go.Histogram(x=pnl, nbinsx=bins, marker_line_width=0))
                fig_pnl.add_vline(x=0, line_dash="dash", opacity=0.5)
                fig_pnl.add_vline(x=float(pnl.mean()), line_dash="dot", opacity=0.9)
                fig_pnl.update_layout(
                    title="Histogramm: PnL Net (€)",
                    xaxis_title="PnL Net (€)", yaxis_title="Häufigkeit",
                    height=360, margin=dict(t=40, l=40, r=20, b=40), showlegend=False
                )
                st.plotly_chart(fig_pnl, use_container_width=True)

        # Korrelation auf Bar-Returns
        st.markdown("### 🔗 Portfolio-Korrelation (Bar-Returns)")
        price_series = []
        for tk, dfbt in all_bt.items():
            if isinstance(dfbt, pd.DataFrame) and "Close" in dfbt.columns and len(dfbt) >= 2:
                s = dfbt["Close"].copy(); s.name = tk
                price_series.append(s)
        if len(price_series) < 2:
            st.info("Mindestens zwei Ticker mit Daten nötig.")
        else:
            prices = pd.concat(price_series, axis=1, join="outer").sort_index().ffill()
            rets = prices.pct_change().dropna(how="all")
            enough = [c for c in rets.columns if rets[c].count() >= 10]
            rets = rets[enough]
            common_rows = rets.dropna(how="any")
            if rets.shape[1] < 2 or len(common_rows) < 10:
                st.info("Zu wenige Datenüberschneidungen für eine Korrelationsmatrix.")
            else:
                corr = common_rows.corr(method="pearson", min_periods=10)
                fig_corr = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu", zmin=-1, zmax=1)
                fig_corr.update_layout(height=520, margin=dict(t=40, l=40, r=30, b=40), coloraxis_colorbar=dict(title="ρ"))
                st.plotly_chart(fig_corr, use_container_width=True)
else:
    st.warning("Keine Ergebnisse. Prüfe Ticker/Intervall/Lookback.")
