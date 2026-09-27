import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ============================================================
# SHERAZ AI FUTURES AGENT V3
# Market dashboard + candlesticks + movers + signal analyst
# Public market data only. No automatic/live order execution.
# ============================================================

st.set_page_config(
    page_title="Sheraz AI Futures Agent V3",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

BINANCE_SPOT = "https://api.binance.com"
BINANCE_FUT = "https://fapi.binance.com"
BYBIT = "https://api.bybit.com"
OKX = "https://www.okx.com"

INTERVALS = {
    "1m": "1m",
    "3m": "3m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
}

HEADERS = {"User-Agent": "Mozilla/5.0 Sheraz-AI-Futures-Agent/3.0"}

st.markdown(
    """
    <style>
    .block-container {padding-top: 1rem; padding-bottom: 2rem;}
    .metric-card {
        padding: 12px 14px;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,.25);
        margin-bottom: 8px;
    }
    .small-note {font-size: .82rem; opacity: .75;}
    </style>
    """,
    unsafe_allow_html=True,
)

def http_get(url, params=None, timeout=8):
    return requests.get(url, params=params, timeout=timeout, headers=HEADERS)

def clean_symbol(s):
    s = str(s).upper().strip().replace("/", "").replace("-", "")
    return s

def fmt_price(x):
    if x is None or not np.isfinite(float(x)):
        return "-"
    x = float(x)
    if abs(x) >= 1000:
        return f"{x:,.2f}"
    if abs(x) >= 1:
        return f"{x:,.4f}"
    if abs(x) >= 0.01:
        return f"{x:,.6f}"
    return f"{x:,.8f}"

def fmt_pct(x):
    if x is None or not np.isfinite(float(x)):
        return "-"
    return f"{float(x):+.2f}%"

def pct(a, b):
    if b == 0 or b is None:
        return np.nan
    return (a / b - 1.0) * 100.0

# ------------------------------------------------------------
# Symbol lists
# ------------------------------------------------------------

@st.cache_data(ttl=300)
def get_symbols(market_type):
    if market_type == "Futures":
        # Binance USD-M futures
        try:
            r = http_get(f"{BINANCE_FUT}/fapi/v1/exchangeInfo")
            if r.status_code == 200:
                rows = []
                for s in r.json().get("symbols", []):
                    if (
                        s.get("status") == "TRADING"
                        and s.get("quoteAsset") == "USDT"
                    ):
                        rows.append(s["symbol"])
                if rows:
                    return sorted(rows)
        except Exception:
            pass

        # Bybit linear fallback
        try:
            r = http_get(
                f"{BYBIT}/v5/market/instruments-info",
                {"category": "linear", "limit": 1000},
            )
            if r.status_code == 200:
                rows = [
                    x["symbol"]
                    for x in r.json().get("result", {}).get("list", [])
                    if x.get("status") == "Trading"
                    and x.get("quoteCoin") == "USDT"
                ]
                if rows:
                    return sorted(set(rows))
        except Exception:
            pass

        return ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]

    # Binance spot
    try:
        r = http_get(f"{BINANCE_SPOT}/api/v3/exchangeInfo")
        if r.status_code == 200:
            rows = []
            for s in r.json().get("symbols", []):
                if (
                    s.get("status") == "TRADING"
                    and s.get("quoteAsset") == "USDT"
                    and s.get("isSpotTradingAllowed", True)
                ):
                    rows.append(s["symbol"])
            if rows:
                return sorted(rows)
    except Exception:
        pass

    return ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "KAVAUSDT"]

# ------------------------------------------------------------
# Klines
# ------------------------------------------------------------

def _binance_klines(market_type, symbol, interval, limit):
    base = BINANCE_FUT if market_type == "Futures" else BINANCE_SPOT
    path = "/fapi/v1/klines" if market_type == "Futures" else "/api/v3/klines"
    r = http_get(
        f"{base}{path}",
        {"symbol": symbol, "interval": interval, "limit": min(limit, 1000)},
    )
    if r.status_code != 200:
        return None
    data = r.json()
    if not isinstance(data, list) or not data:
        return None
    d = pd.DataFrame(
        data,
        columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore",
        ],
    )
    for c in ["open", "high", "low", "close", "volume", "quote_volume"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["open_time"] = pd.to_numeric(d["open_time"], errors="coerce")
    d["data_source"] = "Binance " + market_type
    return d

def _bybit_klines(market_type, symbol, interval, limit):
    category = "linear" if market_type == "Futures" else "spot"
    bybit_interval = {"1m":"1","3m":"3","5m":"5","15m":"15","30m":"30","1h":"60","4h":"240","1d":"D"}.get(interval, "5")
    r = http_get(
        f"{BYBIT}/v5/market/kline",
        {
            "category": category,
            "symbol": symbol,
            "interval": bybit_interval,
            "limit": min(limit, 1000),
        },
    )
    if r.status_code != 200:
        return None
    result = r.json()
    if result.get("retCode") != 0:
        return None
    rows = result.get("result", {}).get("list", [])
    if not rows:
        return None
    rows = list(reversed(rows))
    d = pd.DataFrame(
        rows,
        columns=["open_time","open","high","low","close","volume","turnover"],
    )
    for c in ["open","high","low","close","volume","turnover"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["open_time"] = pd.to_numeric(d["open_time"], errors="coerce")
    d["data_source"] = "Bybit " + market_type
    return d

def _okx_klines(market_type, symbol, interval, limit):
    # OKX uses SWAP for futures; spot uses -USDT.
    if market_type == "Futures":
        inst = symbol.replace("USDT", "-USDT-SWAP")
    else:
        inst = symbol.replace("USDT", "-USDT")
    bar = {"1m":"1m","3m":"3m","5m":"5m","15m":"15m","30m":"30m","1h":"1H","4h":"4H","1d":"1D"}.get(interval, "5m")
    r = http_get(
        f"{OKX}/api/v5/market/candles",
        {"instId": inst, "bar": bar, "limit": min(limit, 300)},
    )
    if r.status_code != 200:
        return None
    result = r.json()
    if result.get("code") != "0":
        return None
    rows = result.get("data", [])
    if not rows:
        return None
    rows = list(reversed(rows))
    d = pd.DataFrame(
        rows,
        columns=[
            "open_time","open","high","low","close","volume",
            "volume_currency","volume_quote","confirm",
        ],
    )
    for c in ["open","high","low","close","volume","volume_currency","volume_quote"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["open_time"] = pd.to_numeric(d["open_time"], errors="coerce")
    d["data_source"] = "OKX " + market_type
    return d

@st.cache_data(ttl=15)
def klines(market_type, symbol, interval, limit=250):
    symbol = clean_symbol(symbol)
    for fn in (
        lambda: _binance_klines(market_type, symbol, interval, limit),
        lambda: _bybit_klines(market_type, symbol, interval, limit),
        lambda: _okx_klines(market_type, symbol, interval, limit),
    ):
        try:
            d = fn()
            if d is not None and len(d) > 20:
                return d
        except Exception:
            pass
    raise RuntimeError(f"{symbol} market data was not available.")

# ------------------------------------------------------------
# 24h tickers / full market
# ------------------------------------------------------------

def _binance_tickers(market_type):
    base = BINANCE_FUT if market_type == "Futures" else BINANCE_SPOT
    path = "/fapi/v1/ticker/24hr" if market_type == "Futures" else "/api/v3/ticker/24hr"
    r = http_get(f"{base}{path}")
    if r.status_code != 200:
        return None
    data = r.json()
    if not isinstance(data, list):
        return None
    rows = []
    for x in data:
        symbol = x.get("symbol", "")
        if not symbol.endswith("USDT"):
            continue
        rows.append({
            "symbol": symbol,
            "price": float(x.get("lastPrice", 0) or 0),
            "change_pct": float(x.get("priceChangePercent", 0) or 0),
            "high_24h": float(x.get("highPrice", 0) or 0),
            "low_24h": float(x.get("lowPrice", 0) or 0),
            "volume": float(x.get("volume", 0) or 0),
            "quote_volume": float(x.get("quoteVolume", 0) or 0),
            "trades": float(x.get("count", 0) or 0),
            "source": "Binance " + market_type,
        })
    return pd.DataFrame(rows)

def _bybit_tickers(market_type):
    category = "linear" if market_type == "Futures" else "spot"
    r = http_get(f"{BYBIT}/v5/market/tickers", {"category": category})
    if r.status_code != 200:
        return None
    result = r.json()
    if result.get("retCode") != 0:
        return None
    rows = []
    for x in result.get("result", {}).get("list", []):
        symbol = x.get("symbol", "")
        if not symbol.endswith("USDT"):
            continue
        rows.append({
            "symbol": symbol,
            "price": float(x.get("lastPrice", 0) or 0),
            "change_pct": float(x.get("price24hPcnt", 0) or 0) * 100,
            "high_24h": float(x.get("highPrice24h", 0) or 0),
            "low_24h": float(x.get("lowPrice24h", 0) or 0),
            "volume": float(x.get("volume24h", 0) or 0),
            "quote_volume": float(x.get("turnover24h", 0) or 0),
            "trades": np.nan,
            "source": "Bybit " + market_type,
        })
    return pd.DataFrame(rows)

@st.cache_data(ttl=20)
def market_tickers(market_type):
    for fn in (
        lambda: _binance_tickers(market_type),
        lambda: _bybit_tickers(market_type),
    ):
        try:
            d = fn()
            if d is not None and not d.empty:
                return d
        except Exception:
            pass
    return pd.DataFrame(columns=[
        "symbol","price","change_pct","high_24h","low_24h",
        "volume","quote_volume","trades","source"
    ])


@st.cache_data(ttl=5)
def order_book(market_type, symbol, limit=10):
    base = BINANCE_FUT if market_type == "Futures" else BINANCE_SPOT
    path = "/fapi/v1/depth" if market_type == "Futures" else "/api/v3/depth"
    try:
        r = http_get(f"{base}{path}", {"symbol": symbol, "limit": limit})
        if r.status_code == 200:
            x = r.json()
            bids = pd.DataFrame(x.get("bids", []), columns=["Price", "Qty"])
            asks = pd.DataFrame(x.get("asks", []), columns=["Price", "Qty"])
            for z in (bids, asks):
                if not z.empty:
                    z["Price"] = pd.to_numeric(z["Price"], errors="coerce")
                    z["Qty"] = pd.to_numeric(z["Qty"], errors="coerce")
            return bids, asks
    except Exception:
        pass
    return pd.DataFrame(columns=["Price","Qty"]), pd.DataFrame(columns=["Price","Qty"])

@st.cache_data(ttl=20)
def funding_rate(symbol):
    try:
        r = http_get(f"{BINANCE_FUT}/fapi/v1/premiumIndex", {"symbol": symbol})
        if r.status_code == 200:
            x = r.json()
            return float(x.get("lastFundingRate", 0) or 0) * 100
    except Exception:
        pass
    return np.nan

# ------------------------------------------------------------
# Technical analysis
# ------------------------------------------------------------

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)

def atr(d, period=14):
    prev = d["close"].shift(1)
    tr = pd.concat(
        [
            d["high"] - d["low"],
            (d["high"] - prev).abs(),
            (d["low"] - prev).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()

def add_indicators(d):
    d = d.copy()
    d["ema9"] = d["close"].ewm(span=9, adjust=False).mean()
    d["ema21"] = d["close"].ewm(span=21, adjust=False).mean()
    d["ema50"] = d["close"].ewm(span=50, adjust=False).mean()
    d["rsi14"] = rsi(d["close"], 14)
    d["atr14"] = atr(d, 14)
    d["vol_ma20"] = d["volume"].rolling(20).mean()
    d["vol_ratio"] = d["volume"] / d["vol_ma20"].replace(0, np.nan)
    return d

def support_resistance(d, lookback=60):
    x = d.tail(lookback)
    support = float(x["low"].min())
    resistance = float(x["high"].max())
    return support, resistance

def build_signal(d):
    d = add_indicators(d)
    x = d.iloc[-1]
    price = float(x["close"])
    ema9 = float(x["ema9"])
    ema21 = float(x["ema21"])
    ema50 = float(x["ema50"])
    rsi14 = float(x["rsi14"])
    atr14 = float(x["atr14"]) if np.isfinite(x["atr14"]) else price * 0.005
    vol_ratio = float(x["vol_ratio"]) if np.isfinite(x["vol_ratio"]) else 1.0
    support, resistance = support_resistance(d)

    long_score = 0
    short_score = 0
    reasons_long = []
    reasons_short = []

    if ema9 > ema21:
        long_score += 2
        reasons_long.append("EMA9 is above EMA21")
    else:
        short_score += 2
        reasons_short.append("EMA9 is below EMA21")

    if price > ema50:
        long_score += 1
        reasons_long.append("Price is above EMA50")
    elif price < ema50:
        short_score += 1
        reasons_short.append("Price is below EMA50")

    if 52 <= rsi14 <= 68:
        long_score += 1
        reasons_long.append("RSI is in a bullish zone")
    elif 32 <= rsi14 <= 48:
        short_score += 1
        reasons_short.append("RSI is in a bearish zone")
    elif rsi14 > 75:
        short_score += 1
        reasons_short.append("RSI is very high")
    elif rsi14 < 25:
        long_score += 1
        reasons_long.append("RSI is very low")

    if vol_ratio >= 1.4:
        if price >= ema9:
            long_score += 1
            reasons_long.append("Volume is significantly above normal")
        else:
            short_score += 1
            reasons_short.append("Volume is significantly above normal")

    # Setup is deliberately conservative: no trade when scores are close.
    if long_score >= 4 and long_score >= short_score + 2:
        direction = "LONG"
        entry = price
        stop = min(support, entry - 1.25 * atr14)
        risk = max(entry - stop, atr14 * 0.75)
        tp1 = entry + 1.5 * risk
        tp2 = entry + 2.5 * risk
        reasons = reasons_long
        score = long_score
    elif short_score >= 4 and short_score >= long_score + 2:
        direction = "SHORT"
        entry = price
        stop = max(resistance, entry + 1.25 * atr14)
        risk = max(stop - entry, atr14 * 0.75)
        tp1 = entry - 1.5 * risk
        tp2 = entry - 2.5 * risk
        reasons = reasons_short
        score = short_score
    else:
        direction = "NO TRADE"
        entry = price
        stop = np.nan
        tp1 = np.nan
        tp2 = np.nan
        risk = np.nan
        reasons = ["EMA/RSI/price structure is not clearly aligned"]
        score = max(long_score, short_score)

    return {
        "direction": direction,
        "score": int(score),
        "price": price,
        "ema9": ema9,
        "ema21": ema21,
        "ema50": ema50,
        "rsi": rsi14,
        "atr": atr14,
        "vol_ratio": vol_ratio,
        "support": support,
        "resistance": resistance,
        "entry": entry,
        "stop": stop,
        "tp1": tp1,
        "tp2": tp2,
        "risk": risk,
        "rr1": 1.5 if direction != "NO TRADE" else np.nan,
        "rr2": 2.5 if direction != "NO TRADE" else np.nan,
        "reasons": reasons,
    }

# ------------------------------------------------------------
# Chart
# ------------------------------------------------------------

def candle_chart(d, symbol, interval, direction):
    x = d.tail(180).copy()
    x["time"] = pd.to_datetime(x["open_time"], unit="ms", utc=True)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.72, 0.28],
    )

    fig.add_trace(
        go.Candlestick(
            x=x["time"],
            open=x["open"],
            high=x["high"],
            low=x["low"],
            close=x["close"],
            name="Candles",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=x["time"], y=x["ema9"], name="EMA9", mode="lines"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=x["time"], y=x["ema21"], name="EMA21", mode="lines"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=x["time"], y=x["ema50"], name="EMA50", mode="lines"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Bar(x=x["time"], y=x["volume"], name="Volume"),
        row=2, col=1,
    )

    fig.update_layout(
        title=f"{symbol} - {interval} - {direction}",
        height=650,
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=45, b=10),
        legend=dict(orientation="h"),
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    return fig

# ------------------------------------------------------------
# AI analyst
# ------------------------------------------------------------

def local_answer(question, symbol, market_type, interval, signal, ticker_row):
    q = question.lower()
    if any(k in q for k in ["long", "buy"]):
        if signal["direction"] == "LONG":
            return (
                f"{symbol} currently has a LONG-side structure. "
                f"Price {fmt_price(signal['price'])}, EMA9 {fmt_price(signal['ema9'])} "
                f"and RSI {signal['rsi']:.1f}. Suggested entry is around the current price, "
                f"SL {fmt_price(signal['stop'])}, TP1 {fmt_price(signal['tp1'])}, "
                f"TP2 {fmt_price(signal['tp2'])}. Do not place a market order without confirmation."
            )
        return (
            f"{symbol} does not have a clear LONG setup right now. Current signal: {signal['direction']}. "
            f"Price {fmt_price(signal['price'])}, RSI {signal['rsi']:.1f}. "
            f"Wait for confirmation before considering a trade."
        )
    if any(k in q for k in ["short", "sell"]):
        if signal["direction"] == "SHORT":
            return (
                f"{symbol} currently has a SHORT-side structure. "
                f"Price {fmt_price(signal['price'])}, EMA9 {fmt_price(signal['ema9'])} "
                f"and RSI {signal['rsi']:.1f}. Suggested entry is around the current price, "
                f"SL {fmt_price(signal['stop'])}, TP1 {fmt_price(signal['tp1'])}, "
                f"TP2 {fmt_price(signal['tp2'])}. Do not place a market order without confirmation."
            )
        return (
            f"{symbol} does not have a clear SHORT setup right now. Current signal: {signal['direction']}. "
            f"Price {fmt_price(signal['price'])}, RSI {signal['rsi']:.1f}. "
            f"Wait for confirmation before considering a trade."
        )
    return (
        f"{symbol} ({market_type}, {interval}) current summary: "
        f"Price {fmt_price(signal['price'])}, direction {signal['direction']}, "
        f"EMA9 {fmt_price(signal['ema9'])}, EMA21 {fmt_price(signal['ema21'])}, "
        f"EMA50 {fmt_price(signal['ema50'])}, RSI {signal['rsi']:.1f}, "
        f"Support {fmt_price(signal['support'])}, Resistance {fmt_price(signal['resistance'])}. "
        f"24h change {fmt_pct(ticker_row.get('change_pct', np.nan))}."
    )

def ai_answer(question, context, fallback):
    api_key = st.secrets.get("OPENAI_API_KEY", "")
    model = st.secrets.get("OPENAI_MODEL", "gpt-4o-mini")
    if not api_key or OpenAI is None:
        return fallback

    try:
        client = OpenAI(api_key=api_key)
        prompt = f"""
You are Sheraz AI Futures Agent, a market-analysis assistant.
Answer in clear English. Use ONLY the supplied live market snapshot for factual claims.
Do not pretend to know future prices. Explain uncertainty.
When the user asks long/short, give a conditional setup with entry zone,
stop-loss and targets only if the technical structure supports it; otherwise say NO TRADE.
Never place an order and never claim an order was placed.

LIVE SNAPSHOT:
{context}

USER QUESTION:
{question}
"""
        resp = client.responses.create(model=model, input=prompt)
        return resp.output_text
    except Exception as e:
        return fallback + f"\n\nAI API is unavailable, so the built-in analysis is being shown."

# ------------------------------------------------------------
# App
# ------------------------------------------------------------

st.title(" Sheraz AI Futures Agent V3")
st.caption("Live market dashboard | Candlesticks | Pump/Dump | Scanner | AI Analyst | Automatic order execution disabled")

with st.sidebar:
    st.header(" Market Controls")
    market_type = st.radio("Market", ["Futures", "Spot"], index=0)
    interval = st.selectbox("Timeframe", list(INTERVALS.keys()), index=2)
    symbols = get_symbols(market_type)

    search = st.text_input("Coin search", "BTCUSDT").upper().strip()
    matches = [s for s in symbols if search in s]
    if not matches:
        matches = symbols[:20]

    default_symbol = search if search in symbols else matches[0]
    symbol = st.selectbox(
        "Coin",
        matches,
        index=matches.index(default_symbol) if default_symbol in matches else 0,
    )

    limit = st.slider("Candles", 100, 500, 250, 50)
    auto_refresh = st.checkbox(" Auto refresh (15 sec)", value=False)
    if st.button(" Refresh now", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.info(
        "This version provides market data and technical analysis. "
        "Live/automatic order execution    disabled is."
    )

# Auto-refresh is intentionally simple and safe.
if auto_refresh:
    st.markdown(
        "<meta http-equiv='refresh' content='15'>",
        unsafe_allow_html=True,
    )

try:
    d = klines(market_type, symbol, interval, limit)
    d = add_indicators(d)
    sig = build_signal(d)
except Exception as e:
    st.error(str(e))
    st.stop()

ticks = market_tickers(market_type)
ticker_match = ticks[ticks["symbol"] == symbol]
if ticker_match.empty:
    ticker = {
        "price": sig["price"],
        "change_pct": np.nan,
        "high_24h": np.nan,
        "low_24h": np.nan,
        "quote_volume": np.nan,
        "volume": np.nan,
        "source": d["data_source"].iloc[-1],
    }
else:
    ticker = ticker_match.iloc[0].to_dict()

# ------------------------------------------------------------
# Top market view
# ------------------------------------------------------------

st.subheader(" Market Overview")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Price", fmt_price(ticker["price"]))
c2.metric("24h", fmt_pct(ticker.get("change_pct", np.nan)))
c3.metric("24h High", fmt_price(ticker.get("high_24h", np.nan)))
c4.metric("24h Low", fmt_price(ticker.get("low_24h", np.nan)))
c5.metric("Data", ticker.get("source", "Unknown"))

# ------------------------------------------------------------
# Main chart + signal
# ------------------------------------------------------------

left, right = st.columns([2.2, 1])

with left:
    st.plotly_chart(
        candle_chart(d, symbol, interval, sig["direction"]),
        use_container_width=True,
    )
    st.caption(
        f"Data source: {d['data_source'].iloc[-1]}  "
        f"Last candle: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    )

with right:
    st.subheader(" AI Technical Signal")
    st.metric("Direction", sig["direction"])
    st.metric("RSI", f"{sig['rsi']:.1f}")
    st.metric("EMA9 / EMA21", f"{fmt_price(sig['ema9'])} / {fmt_price(sig['ema21'])}")

    if sig["direction"] != "NO TRADE":
        st.success(
            f"{sig['direction']}\n\n"
            f"Entry: {fmt_price(sig['entry'])}\n\n"
            f"SL: {fmt_price(sig['stop'])}\n\n"
            f"TP1: {fmt_price(sig['tp1'])}\n\n"
            f"TP2: {fmt_price(sig['tp2'])}"
        )
    else:
        st.warning("NO TRADE - market structure is not clear")

    st.write("**Key Reasons:**")
    for reason in sig["reasons"]:
        st.write(" " + reason)

    st.write(
        f"Support: **{fmt_price(sig['support'])}**  \n"
        f"Resistance: **{fmt_price(sig['resistance'])}**  \n"
        f"Volume ratio: **{sig['vol_ratio']:.2f}x**"
    )


# ------------------------------------------------------------
# Order book / derivatives data
# ------------------------------------------------------------

ob_left, ob_mid, ob_right = st.columns([1, 1, 1])
bids, asks = order_book(market_type, symbol, 10)

with ob_left:
    st.subheader(" Order Book")
    if not bids.empty:
        b = bids.copy()
        b["Price"] = b["Price"].map(fmt_price)
        b["Qty"] = b["Qty"].map(lambda x: f"{x:,.4f}")
        st.write("Bids")
        st.dataframe(b, use_container_width=True, hide_index=True)
    else:
        st.caption("Order book is not available.")

with ob_mid:
    st.subheader(" Ask Side")
    if not asks.empty:
        a = asks.copy()
        a["Price"] = a["Price"].map(fmt_price)
        a["Qty"] = a["Qty"].map(lambda x: f"{x:,.4f}")
        st.dataframe(a, use_container_width=True, hide_index=True)
    else:
        st.caption("Ask data is not available.")

with ob_right:
    st.subheader(" Futures Info")
    if market_type == "Futures":
        fr = funding_rate(symbol)
        st.metric("Funding Rate", f"{fr:+.4f}%" if np.isfinite(fr) else "-")
        st.caption("Funding rate is an indicator of the periodic futures cost/credit.")
    else:
        st.metric("Market", "Spot")
        st.caption("Spot markets do not have a funding rate.")

# ------------------------------------------------------------
# Pump / Dump
# ------------------------------------------------------------

st.subheader(" Pump / Dump Scanner")
if ticks.empty:
    st.warning("Full market ticker is not available right now.")
else:
    movers = ticks.copy()
    movers["abs_change"] = movers["change_pct"].abs()
    movers = movers.sort_values("abs_change", ascending=False)

    p1, p2 = st.columns(2)
    with p1:
        st.write("###  Top Pumpers")
        pump = ticks.sort_values("change_pct", ascending=False).head(15).copy()
        pump["Price"] = pump["price"].map(fmt_price)
        pump["24h %"] = pump["change_pct"].map(fmt_pct)
        pump["Quote Volume"] = pump["quote_volume"].map(lambda x: f"{x:,.0f}")
        st.dataframe(
            pump[["symbol", "Price", "24h %", "Quote Volume"]],
            use_container_width=True,
            hide_index=True,
        )

    with p2:
        st.write("###  Top Dumpers")
        dump = ticks.sort_values("change_pct", ascending=True).head(15).copy()
        dump["Price"] = dump["price"].map(fmt_price)
        dump["24h %"] = dump["change_pct"].map(fmt_pct)
        dump["Quote Volume"] = dump["quote_volume"].map(lambda x: f"{x:,.0f}")
        st.dataframe(
            dump[["symbol", "Price", "24h %", "Quote Volume"]],
            use_container_width=True,
            hide_index=True,
        )

# ------------------------------------------------------------
# Full market list
# ------------------------------------------------------------

st.subheader(" Exchange Market List - USDT Pairs")
if not ticks.empty:
    table = ticks.copy()
    table["Price"] = table["price"].map(fmt_price)
    table["24h %"] = table["change_pct"].map(fmt_pct)
    table["High"] = table["high_24h"].map(fmt_price)
    table["Low"] = table["low_24h"].map(fmt_price)
    table["Quote Volume"] = table["quote_volume"].map(lambda x: f"{x:,.0f}")
    table = table.sort_values("quote_volume", ascending=False)
    st.dataframe(
        table[["symbol", "Price", "24h %", "High", "Low", "Quote Volume"]],
        use_container_width=True,
        hide_index=True,
        height=430,
    )

# ------------------------------------------------------------
# Multi-coin scanner
# ------------------------------------------------------------

st.subheader(" Multi-Coin Technical Scanner")
scan_text = st.text_input(
    "Coins to scan (comma separated)",
    "BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,KAVAUSDT",
)
scan_button = st.button("Scan Coins", type="primary")

if scan_button:
    scan_rows = []
    requested = [clean_symbol(x) for x in scan_text.split(",") if x.strip()]
    for s in requested:
        try:
            kd = klines(market_type, s, interval, 180)
            sg = build_signal(kd)
            tm = ticks[ticks["symbol"] == s]
            ch = float(tm.iloc[0]["change_pct"]) if not tm.empty else np.nan
            scan_rows.append({
                "Coin": s,
                "Direction": sg["direction"],
                "Price": fmt_price(sg["price"]),
                "24h %": fmt_pct(ch),
                "RSI": round(sg["rsi"], 1),
                "EMA9": fmt_price(sg["ema9"]),
                "EMA21": fmt_price(sg["ema21"]),
                "Support": fmt_price(sg["support"]),
                "Resistance": fmt_price(sg["resistance"]),
            })
        except Exception as e:
            scan_rows.append({
                "Coin": s,
                "Direction": "DATA ERROR",
                "Price": "-",
                "24h %": "-",
                "RSI": "-",
                "EMA9": "-",
                "EMA21": "-",
                "Support": "-",
                "Resistance": "-",
            })
    st.dataframe(pd.DataFrame(scan_rows), use_container_width=True, hide_index=True)

# ------------------------------------------------------------
# AI chat
# ------------------------------------------------------------

st.subheader(" Market AI - Ask Me")
st.caption(
    "Examples: 'What is BTCUSDT doing now?', "
    "'Where is the KAVAUSDT long setup?', "
    "'Is short better now or should I wait?'"
)

context = f"""
Symbol: {symbol}
Market: {market_type}
Timeframe: {interval}
Price: {fmt_price(sig['price'])}
24h Change: {fmt_pct(ticker.get('change_pct', np.nan))}
24h High: {fmt_price(ticker.get('high_24h', np.nan))}
24h Low: {fmt_price(ticker.get('low_24h', np.nan))}
EMA9: {fmt_price(sig['ema9'])}
EMA21: {fmt_price(sig['ema21'])}
EMA50: {fmt_price(sig['ema50'])}
RSI14: {sig['rsi']:.2f}
ATR14: {fmt_price(sig['atr'])}
Support: {fmt_price(sig['support'])}
Resistance: {fmt_price(sig['resistance'])}
Volume ratio: {sig['vol_ratio']:.2f}x
Signal: {sig['direction']}
Entry: {fmt_price(sig['entry'])}
Stop: {fmt_price(sig['stop'])}
TP1: {fmt_price(sig['tp1'])}
TP2: {fmt_price(sig['tp2'])}
"""

question = st.text_input(
    "Ask your market question",
    placeholder="Example: Should I long BTCUSDT now or wait?",
)

if st.button(" Ask AI", type="primary"):
    fallback = local_answer(
        question or "market overview",
        symbol,
        market_type,
        interval,
        sig,
        ticker,
    )
    answer = ai_answer(question or "market overview", context, fallback)
    st.markdown(answer)

st.divider()
st.caption(
    "This tool performs technical analysis using live market data. "
    "It does not guarantee future prices or profit. Live/automatic order placement is disabled."
)
