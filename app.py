import os, json
from datetime import datetime, timezone
import requests
import numpy as np
import pandas as pd
import streamlit as st

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

st.set_page_config(
    page_title="Sheraz AI Futures Agent",
    page_icon="🤖",
    layout="wide"
)

BINANCE = "https://fapi.binance.com"
BYBIT = "https://api.bybit.com"


@st.cache_data(ttl=15)
def klines(symbol, interval, limit=250):
    """
    پہلے Binance Futures سے data لینے کی کوشش۔
    اگر Binance 451/403/connection error دے تو Bybit Linear Futures
    سے public market data استعمال کیا جائے گا۔
    """

    # -----------------------------
    # 1) Binance Futures
    # -----------------------------
    try:
        r = requests.get(
            f"{BINANCE}/fapi/v1/klines",
            params={
                "symbol": symbol,
                "interval": interval,
                "limit": limit
            },
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0"}
        )

        if r.status_code == 200:
            data = r.json()

            cols = [
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "trades",
                "taker_buy_base", "taker_buy_quote", "ignore"
            ]

            d = pd.DataFrame(data, columns=cols)

            for c in [
                "open", "high", "low", "close",
                "volume", "quote_volume"
            ]:
                d[c] = pd.to_numeric(d[c], errors="coerce")

            d["data_source"] = "Binance Futures"

            return d

    except Exception:
        pass

    # -----------------------------
    # 2) Bybit Linear Futures
    # -----------------------------
    bybit_interval = {
        "1m": "1",
        "5m": "5",
        "15m": "15",
        "30m": "30",
        "1h": "60"
    }.get(interval, "5")

    try:
        r = requests.get(
            f"{BYBIT}/v5/market/kline",
            params={
                "category": "linear",
                "symbol": symbol,
                "interval": bybit_interval,
                "limit": min(limit, 1000)
            },
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0"}
        )

        r.raise_for_status()

        payload = r.json()

        if payload.get("retCode") != 0:
            raise RuntimeError(
                payload.get("retMsg", "Bybit data error")
            )

        rows = payload["result"]["list"]

        if not rows:
            raise RuntimeError("No market data returned")

        # Bybit format:
        # [startTime, open, high, low, close, volume, turnover]

        rows = list(reversed(rows))

        d = pd.DataFrame(
            rows,
            columns=[
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "quote_volume"
            ]
        )

        d["open_time"] = pd.to_numeric(
            d["open_time"], errors="coerce"
        )

        for c in [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_volume"
        ]:
            d[c] = pd.to_numeric(d[c], errors="coerce")

        d["close_time"] = d["open_time"]

        d["trades"] = 0
        d["taker_buy_base"] = 0
        d["taker_buy_quote"] = 0
        d["ignore"] = 0

        d["data_source"] = "Bybit Linear Futures (fallback)"

        return d

    except Exception as e:
        raise RuntimeError(
            f"Binance اور Bybit دونوں سے market data حاصل نہیں ہو سکا: {e}"
        )


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(d, n=14):
    p = d.close.shift(1)

    tr = pd.concat(
        [
            (d.high - d.low),
            (d.high - p).abs(),
            (d.low - p).abs()
        ],
        axis=1
    ).max(axis=1)

    return tr.ewm(alpha=1/n, adjust=False).mean()


def analyze(d):
    x = d.copy()

    x["ema9"] = ema(x.close, 9)
    x["ema21"] = ema(x.close, 21)
    x["rsi"] = rsi(x.close)
    x["atr"] = atr(x)
    x["vol_ma"] = x.volume.rolling(20).mean()

    x["high20"] = x.high.rolling(20).max()
    x["low20"] = x.low.rolling(20).min()

    a = x.iloc[-1]

    score = 0
    reasons = []

    if a.ema9 > a.ema21:
        score += 2
        reasons.append("EMA9، EMA21 سے اوپر ہے۔")
    else:
        score -= 2
        reasons.append("EMA9، EMA21 سے نیچے ہے۔")

    if a.close > a.ema21:
        score += 1
        reasons.append("قیمت EMA21 سے اوپر ہے۔")
    else:
        score -= 1
        reasons.append("قیمت EMA21 سے نیچے ہے۔")

    if 55 <= a.rsi < 70:
        score += 1
        reasons.append(
            f"RSI {a.rsi:.1f} مثبت مومینٹم دکھا رہا ہے۔"
        )

    elif 30 < a.rsi <= 45:
        score -= 1
        reasons.append(
            f"RSI {a.rsi:.1f} کمزور مومینٹم دکھا رہا ہے۔"
        )

    elif a.rsi >= 70:
        reasons.append(
            f"RSI {a.rsi:.1f} اووربوٹ زون میں ہے۔"
        )

    elif a.rsi <= 30:
        reasons.append(
            f"RSI {a.rsi:.1f} اوورسولڈ زون میں ہے۔"
        )

    if a.volume > a.vol_ma * 1.2:
        score += 1 if a.close > a.open else -1
        reasons.append(
            "حجم 20-candle اوسط سے کافی زیادہ ہے۔"
        )
    else:
        reasons.append(
            "حجم میں بڑا اضافہ نہیں ہے۔"
        )

    price = float(a.close)
    av = float(max(a.atr, price * 0.001))

    if score >= 3:
        sig = "LONG SETUP"
        entry = price
        sl = price - 1.25 * av
        tp1 = price + 1.5 * av
        tp2 = price + 2.5 * av

    elif score <= -3:
        sig = "SHORT SETUP"
        entry = price
        sl = price + 1.25 * av
        tp1 = price - 1.5 * av
        tp2 = price - 2.5 * av

    else:
        sig = "NO TRADE"
        entry = sl = tp1 = tp2 = price

    rr = (
        abs(tp2 - entry) / abs(entry - sl)
        if entry != sl else 0
    )

    return {
        "signal": sig,
        "score": score,
        "entry": entry,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "rr": rr,
        "rsi": float(a.rsi),
        "ema9": float(a.ema9),
        "ema21": float(a.ema21),
        "support": float(a.low20),
        "resistance": float(a.high20),
        "reasons": reasons,
        "data": x
    }


def fmt(x):
    return f"{x:,.6f}".rstrip("0").rstrip(".")


def risk_calc(entry, sl, balance, risk_pct):
    risk_money = balance * risk_pct / 100
    dist = abs(entry - sl)
    qty = risk_money / dist if dist else 0
    return risk_money, qty, qty * entry


def scan_symbols(symbols, interval):
    rows = []

    for s in symbols:
        try:
            a = analyze(klines(s, interval))

            rows.append({
                "Coin": s,
                "Signal": a["signal"],
                "Score": a["score"],
                "RSI": round(a["rsi"], 1),
                "Entry": a["entry"],
                "SL": a["sl"],
                "TP2": a["tp2"],
                "R:R": round(a["rr"], 2)
            })

        except Exception:
            pass

    return pd.DataFrame(rows)


def ask_ai(question, ctx):
    key = st.secrets.get(
        "OPENAI_API_KEY",
        os.getenv("OPENAI_API_KEY", "")
    )

    if not key or OpenAI is None:
        return None

    client = OpenAI(api_key=key)

    instructions = """
آپ Sheraz AI Futures Agent ہیں۔
صرف اردو رسم الخط میں جواب دیں۔

موجودہ فراہم کردہ مارکیٹ ڈیٹا سے باہر فرضی قیمت نہ بنائیں۔
واضح setup نہ ہو تو NO TRADE کہیں۔

جواب میں مارکیٹ کی حالت، Long/Short/No Trade،
وجہ، Entry، SL، TP1، TP2 اور invalidation دیں۔

منافع یا نقصان کی ضمانت نہ دیں۔
پوری رقم ایک trade میں لگانے یا بہت زیادہ leverage
لینے کی ترغیب نہ دیں۔

یہ تعلیمی/Paper Trading signal ہے،
مالی ضمانت نہیں۔
"""

    prompt = (
        f"صارف کا سوال:\n{question}\n\n"
        f"موجودہ مارکیٹ ڈیٹا:\n"
        f"{json.dumps(ctx, ensure_ascii=False, default=str)}"
    )

    try:
        response = client.responses.create(
            model="gpt-5.6-luna",
            instructions=instructions,
            input=prompt
        )

        return response.output_text

    except Exception as e:
        return (
            "AI جواب نہیں دے سکا۔ "
            "API Key/ماڈل کی سیٹنگ چیک کریں۔ "
            f"تکنیکی پیغام: {e}"
        )


st.title("🤖 Sheraz AI Futures Agent — V2")

st.caption(
    "اردو AI + Futures Scanner | Signal/Paper Mode | "
    "Live Auto-Trading بند ہے"
)


with st.sidebar:

    st.header("⚙️ سیٹنگز")

    symbol = st.text_input(
        "Coin Symbol",
        "BTCUSDT"
    ).upper().strip()

    interval = st.selectbox(
        "Timeframe",
        ["5m", "15m", "1m", "30m", "1h"],
        index=0
    )

    balance = st.number_input(
        "Paper Balance (USDT)",
        min_value=1.0,
        value=30.0,
        step=1.0
    )

    risk_pct = st.slider(
        "فی trade زیادہ سے زیادہ risk %",
        0.25,
        3.0,
        1.0,
        0.25
    )

    st.warning(
        "سگنل منافع کی ضمانت نہیں ہیں۔ "
        "Futures میں نقصان ہو سکتا ہے۔"
    )


try:

    d = klines(symbol, interval)

    a = analyze(d)

    price = float(d.close.iloc[-1])

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("قیمت", fmt(price))
    c2.metric("Signal", a["signal"])
    c3.metric("RSI", f"{a['rsi']:.1f}")
    c4.metric("Score", a["score"])

    source = d["data_source"].iloc[-1]

    if source == "Binance Futures":
        st.success(
            "📡 Market Data Source: Binance Futures"
        )
    else:
        st.info(
            "📡 Market Data Source: "
            "Bybit Linear Futures fallback"
        )

    if a["signal"] == "LONG SETUP":

        st.success(
            "🟢 LONG SETUP — پہلے Paper mode میں چیک کریں۔"
        )

    elif a["signal"] == "SHORT SETUP":

        st.error(
            "🔴 SHORT SETUP — پہلے Paper mode میں چیک کریں۔"
        )

    else:

        st.warning(
            "🟡 NO TRADE — واضح setup نہیں۔"
        )

    st.subheader("🎯 سگنل")

    z = st.columns(5)

    for col, label, val in zip(
        z,
        ["Entry", "Stop Loss", "TP1", "TP2", "Risk/Reward"],
        [
            a["entry"],
            a["sl"],
            a["tp1"],
            a["tp2"],
            f"1 : {a['rr']:.2f}"
        ]
    ):
        col.metric(
            label,
            fmt(val)
            if isinstance(val, (float, int))
            else val
        )

    st.subheader("🧠 تجزیے کی وجوہات")

    for r in a["reasons"]:
        st.write("•", r)

    st.write(
        f"Support: **{fmt(a['support'])}**  |  "
        f"Resistance: **{fmt(a['resistance'])}**"
    )

    st.subheader("💰 Risk Calculator")

    rm, qty, notional = risk_calc(
        a["entry"],
        a["sl"],
        balance,
        risk_pct
    )

    r1, r2, r3 = st.columns(3)

    r1.metric(
        "زیادہ سے زیادہ رسک",
        f"{rm:.2f} USDT"
    )

    r2.metric(
        "Calculated Qty",
        f"{qty:.6f}"
    )

    r3.metric(
        "Approx. Notional",
        f"{notional:.2f} USDT"
    )

    st.subheader("📊 قیمت + EMA")

    chart = a["data"].copy()

    chart["وقت"] = pd.to_datetime(
        chart.open_time,
        unit="ms"
    )

    chart = chart.set_index("وقت")[
        ["close", "ema9", "ema21"]
    ].tail(150)

    chart.columns = [
        "Price",
        "EMA9",
        "EMA21"
    ]

    st.line_chart(chart)

    st.subheader("🔍 Multi-Coin Scanner")

    default = (
        "BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,"
        "XRPUSDT,DOGEUSDT,ADAUSDT,LINKUSDT"
    )

    symbols = [
        s.strip().upper()
        for s in st.text_input(
            "Coins (comma separated)",
            default
        ).split(",")
        if s.strip()
    ]

    if st.button("Scan Coins"):

        result = scan_symbols(
            symbols,
            interval
        )

        if len(result):

            st.dataframe(
                result,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.warning(
                "کوئی coin scan نہیں ہو سکا۔"
            )

    st.subheader("🤖 اپنے AI Agent سے بات کریں")

    question = st.text_area(
        "مثال: ابھی مارکیٹ کی کیا صورتحال ہے؟ "
        "BTC میں Long/Short setup ہے؟",
        height=100
    )

    if st.button("AI سے پوچھیں"):

        ctx = {
            "symbol": symbol,
            "timeframe": interval,
            "price": price,
            "signal": a["signal"],
            "score": a["score"],
            "rsi": a["rsi"],
            "ema9": a["ema9"],
            "ema21": a["ema21"],
            "support": a["support"],
            "resistance": a["resistance"],
            "entry": a["entry"],
            "sl": a["sl"],
            "tp1": a["tp1"],
            "tp2": a["tp2"],
            "risk_reward": a["rr"],
            "reasons": a["reasons"]
        }

        ans = ask_ai(
            question or "موجودہ مارکیٹ کی صورتحال بتائیں۔",
            ctx
        )

        if ans:

            st.chat_message(
                "assistant"
            ).write(ans)

        else:

            st.info(
                "AI Chat فعال کرنے کے لیے "
                "Streamlit Secrets میں OPENAI_API_KEY "
                "شامل کریں۔ Structured signal ابھی دستیاب ہے۔"
            )

    st.subheader("📝 Signal Journal")

    if "journal" not in st.session_state:
        st.session_state.journal = []

    if st.button(
        "اس سگنل کو Journal میں محفوظ کریں"
    ):

        st.session_state.journal.append({
            "وقت": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "Coin": symbol,
            "TF": interval,
            "Signal": a["signal"],
            "Entry": a["entry"],
            "SL": a["sl"],
            "TP2": a["tp2"]
        })

        st.success(
            "Signal محفوظ ہو گیا۔"
        )

    if st.session_state.journal:

        st.dataframe(
            pd.DataFrame(
                st.session_state.journal
            ),
            use_container_width=True,
            hide_index=True
        )

    st.caption(
        "آخری اپڈیٹ: "
        + datetime.now(timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )
    )

except Exception as e:

    st.error(
        "Market data حاصل نہیں ہو سکا۔ "
        "Symbol مثلاً BTCUSDT درست لکھیں۔"
    )

    st.code(str(e))
