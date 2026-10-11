import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

# ==================== CONFIG ====================
st.set_page_config(page_title="Pro Crypto Dashboard v13", layout="wide", page_icon="📊")
st.markdown("""
<style>
    div[data-testid="stMetricValue"] {font-size: 1.4rem;}
    .stAlert {border-radius: 10px;}
</style>
""", unsafe_allow_html=True)

# ==================== AI CHATBOT SETUP ====================
try:
    from groq import Groq
    _groq_ok = True
except Exception:
    _groq_ok = False
try:
    _groq_key = st.secrets.get("GROQ_API_KEY", None)
except Exception:
    _groq_key = None

# ==================== STRATEGY CONSTANTS ====================
ADX_MIN = 20          # below this = chop, no trades
MIN_RISK_ATR = 1.0    # stop never tighter than 1 ATR (noise)
MAX_RISK_ATR = 3.0    # stop wider than 3 ATR -> trade skipped
MAX_COST_R = 0.20     # fees+slippage must be < 0.20R, else skip
MAX_HOLD = 48         # time stop (1h candles)
WARMUP = 250          # candles needed before indicators are valid
CHASE_ATR = 0.5       # live price > 0.5 ATR from signal close = missed
SWING_K = 3           # swing pivot strength
OVERLAY_BARS = 12     # lookback for OI / CVD checks
GRADE_ORDER = {"A": 0, "B": 1, "C": 2, "SKIP": 3}

# ==================== DATA LAYER ====================
# Ek hi venue se saara data (no mixing): Binance -> OKX -> MEXC
BINANCE = "https://fapi.binance.com"
OKX = "https://www.okx.com"
MEXC = "https://contract.mexc.com"
SPOT_BIN = "https://data-api.binance.vision"  # Binance spot, market-data-only host
HDRS = {"User-Agent": "Mozilla/5.0 (crypto-dashboard)"}


def _get(url, params=None, timeout=12):
    for attempt in range(3):
        r = requests.get(url, params=params, timeout=timeout, headers=HDRS)
        if r.status_code in (429, 418) and attempt < 2:  # rate limit -> thoda ruk kar retry
            time.sleep(1.0 + attempt)
            continue
        r.raise_for_status()
        return r.json()


def _f(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def _tickers_binance():
    rows = []
    for d in _get(f"{BINANCE}/fapi/v1/ticker/24hr"):
        s = d.get('symbol', '')
        if not s.endswith('USDT'):
            continue
        rows.append({'symbol': s, 'lastPrice': _f(d['lastPrice']),
                     'riseFallRate': _f(d['priceChangePercent']),
                     'volume24': _f(d['quoteVolume'])})
    return pd.DataFrame(rows)


def _tickers_okx():
    d = _get(f"{OKX}/api/v5/market/tickers", {"instType": "SWAP"})
    if d.get("code") != "0":
        raise ValueError("okx tickers")
    rows = []
    for x in d["data"]:
        iid = x.get("instId", "")
        if not iid.endswith("-USDT-SWAP"):
            continue
        last = _f(x.get("last")); op = _f(x.get("open24h"))
        if not np.isfinite(last) or last <= 0:
            continue
        chg = (last - op) / op * 100 if np.isfinite(op) and op > 0 else 0.0
        rows.append({'symbol': iid.replace("-USDT-SWAP", "USDT"), 'lastPrice': last,
                     'riseFallRate': chg, 'volume24': _f(x.get("volCcy24h"), 0.0) * last})
    return pd.DataFrame(rows)


def _tickers_mexc():
    d = _get(f"{MEXC}/api/v1/contract/ticker")
    if not d.get("success"):
        raise ValueError("mexc tickers")
    df = pd.DataFrame(d["data"])
    df = df[df['symbol'].str.endswith('_USDT')].copy()
    out = pd.DataFrame({'symbol': df['symbol'].str.replace('_USDT', 'USDT'),
                        'lastPrice': pd.to_numeric(df['lastPrice'], errors='coerce'),
                        'riseFallRate': pd.to_numeric(df['riseFallRate'], errors='coerce') * 100,
                        'volume24': pd.to_numeric(df['amount24'], errors='coerce')
                        if 'amount24' in df.columns else pd.to_numeric(df['volume24'], errors='coerce')})
    return out


@st.cache_data(ttl=30, show_spinner=False)
def fetch_tickers():
    """Returns (df, source). Pehla venue jo chal jaye."""
    for name, fn in (("binance", _tickers_binance), ("okx", _tickers_okx), ("mexc", _tickers_mexc)):
        try:
            df = fn()
            if df is not None and not df.empty:
                df = df.dropna(subset=['lastPrice', 'volume24'])
                return df.sort_values('volume24', ascending=False).reset_index(drop=True), name
        except Exception:
            continue
    return pd.DataFrame(), "none"


def _kl_binance(symbol, limit):
    need = limit + 1
    rows, end = [], None
    for _ in range(12):
        lim = min(1000, need - len(rows))
        p = {"symbol": symbol, "interval": "1h", "limit": lim}
        if end is not None:
            p["endTime"] = end
        data = _get(f"{BINANCE}/fapi/v1/klines", p)
        if not data:
            break
        rows = data + rows
        end = data[0][0] - 1
        if len(rows) >= need or len(data) < lim:
            break
    df = pd.DataFrame([k[:6] for k in rows], columns=['t', 'open', 'high', 'low', 'close', 'volume'])
    for c in ('open', 'high', 'low', 'close', 'volume'):
        df[c] = pd.to_numeric(df[c])
    df['time'] = pd.to_datetime(pd.to_numeric(df['t']), unit='ms')
    return df


def _kl_okx(symbol, limit):
    inst = symbol[:-4] + "-USDT-SWAP"
    need = min(limit, 1400) + 1
    rows, after = [], None
    for _ in range(6):
        p = {"instId": inst, "bar": "1H", "limit": 300}
        if after is not None:
            p["after"] = after
        d = _get(f"{OKX}/api/v5/market/candles", p)
        if d.get("code") != "0" or not d.get("data"):
            break
        batch = d["data"]
        rows.extend(batch)
        after = batch[-1][0]
        if len(rows) >= need or len(batch) < 300:
            break
        time.sleep(0.15)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([r[:9] for r in rows],
                      columns=['t', 'open', 'high', 'low', 'close', 'vol', 'volCcy', 'volCcyQuote', 'confirm'])
    df = df[df['confirm'] == '1'].copy()  # sirf closed candles
    for c in ('open', 'high', 'low', 'close'):
        df[c] = pd.to_numeric(df[c])
    df['volume'] = pd.to_numeric(df['volCcy'])
    df['time'] = pd.to_datetime(pd.to_numeric(df['t']), unit='ms')
    return df


def _kl_mexc(symbol, limit):
    d = _get(f"{MEXC}/api/v1/contract/kline/{symbol[:-4]}_USDT", {"interval": "Min60"})
    if not (d.get("success") and d.get("data")):
        return pd.DataFrame()
    k = d["data"]
    return pd.DataFrame({'time': pd.to_datetime(pd.Series(k['time']), unit='s'),
                         'open': pd.to_numeric(pd.Series(k['open'])),
                         'high': pd.to_numeric(pd.Series(k['high'])),
                         'low': pd.to_numeric(pd.Series(k['low'])),
                         'close': pd.to_numeric(pd.Series(k['close'])),
                         'volume': pd.to_numeric(pd.Series(k['vol']))})


@st.cache_data(ttl=45, show_spinner=False)
def fetch_klines(symbol, limit, src):
    """1h CLOSED candles only (forming candle hamesha drop -> no repaint)."""
    fn = {"binance": _kl_binance, "okx": _kl_okx, "mexc": _kl_mexc}.get(src)
    if fn is None:
        return pd.DataFrame()
    try:
        df = fn(symbol, limit)
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    df = df[['time', 'open', 'high', 'low', 'close', 'volume']].copy()
    df['time'] = pd.to_datetime(df['time']).astype('datetime64[ns]')
    now = pd.Timestamp.now(tz='UTC').tz_localize(None)
    df = df[df['time'] + pd.Timedelta(hours=1) <= now]
    df = df.sort_values('time').drop_duplicates('time').tail(limit).reset_index(drop=True)
    return df


@st.cache_data(ttl=60, show_spinner=False)
def fetch_spot_map(src):
    """symbol -> (spot price, spot 24h quote volume). Empty dict = unavailable."""
    out = {}
    try:
        if src == "binance":
            for d in _get(f"{SPOT_BIN}/api/v3/ticker/24hr"):
                sy = d.get('symbol', '')
                if sy.endswith('USDT'):
                    out[sy] = (_f(d.get('lastPrice')), _f(d.get('quoteVolume'), 0.0))
        elif src == "okx":
            d = _get(f"{OKX}/api/v5/market/tickers", {"instType": "SPOT"})
            if d.get("code") == "0":
                for x in d["data"]:
                    iid = x.get("instId", "")
                    if iid.endswith("-USDT"):
                        out[iid.replace("-USDT", "USDT")] = (_f(x.get("last")), _f(x.get("volCcy24h"), 0.0))
    except Exception:
        pass
    return out


@st.cache_data(ttl=120, show_spinner=False)
def fetch_perp_info(symbol, src):
    """Current funding (%), next settlement (ms). None = unavailable."""
    try:
        if src == "binance":
            d = _get(f"{BINANCE}/fapi/v1/premiumIndex", {"symbol": symbol})
            return {'funding': float(d["lastFundingRate"]) * 100, 'next': int(d.get("nextFundingTime", 0))}
        if src == "okx":
            d = _get(f"{OKX}/api/v5/public/funding-rate", {"instId": symbol[:-4] + "-USDT-SWAP"})
            if d.get("code") == "0" and d.get("data"):
                x = d["data"][0]
                return {'funding': float(x["fundingRate"]) * 100, 'next': int(_f(x.get("fundingTime"), 0))}
        if src == "mexc":
            d = _get(f"{MEXC}/api/v1/contract/funding_rate/{symbol[:-4]}_USDT")
            if d.get("success"):
                return {'funding': float(d["data"]["fundingRate"]) * 100, 'next': 0}
    except Exception:
        pass
    return None


@st.cache_data(ttl=300, show_spinner=False)
def fetch_funding_hist(symbol, src):
    """Last settlements (%, ascending). [] = unavailable."""
    try:
        if src == "binance":
            d = _get(f"{BINANCE}/fapi/v1/fundingRate", {"symbol": symbol, "limit": 30})
            return [float(x['fundingRate']) * 100 for x in sorted(d, key=lambda x: x['fundingTime'])]
        if src == "okx":
            d = _get(f"{OKX}/api/v5/public/funding-rate-history",
                     {"instId": symbol[:-4] + "-USDT-SWAP", "limit": 30})
            if d.get("code") == "0" and d.get("data"):
                rows = sorted(d["data"], key=lambda x: int(x['fundingTime']))
                return [float(x['fundingRate']) * 100 for x in rows]
    except Exception:
        pass
    return []


@st.cache_data(ttl=120, show_spinner=False)
def fetch_flow(symbol, src):
    """Taker buy/sell volume per hour (ascending). Empty df = data unavailable."""
    try:
        if src == "binance":
            d = _get(f"{BINANCE}/futures/data/takerlongshortRatio",
                     {"symbol": symbol, "period": "1h", "limit": 30})
            if d:
                df = pd.DataFrame(d)
                df['ts'] = pd.to_numeric(df['timestamp'])
                df['buy'] = pd.to_numeric(df['buyVol'])
                df['sell'] = pd.to_numeric(df['sellVol'])
                df['delta'] = df['buy'] - df['sell']
                return df[['ts', 'buy', 'sell', 'delta']].sort_values('ts').reset_index(drop=True)
        if src == "okx":
            d = _get(f"{OKX}/api/v5/rubik/stat/taker-volume",
                     {"ccy": symbol[:-4], "instType": "CONTRACTS", "period": "1H"})
            if d.get("code") == "0" and d.get("data"):
                df = pd.DataFrame([r[:3] for r in d["data"]], columns=['ts', 'sell', 'buy'])
                for c in ('ts', 'sell', 'buy'):
                    df[c] = pd.to_numeric(df[c])
                df['delta'] = df['buy'] - df['sell']
                return df[['ts', 'buy', 'sell', 'delta']].sort_values('ts').reset_index(drop=True)
    except Exception:
        pass
    return pd.DataFrame()


@st.cache_data(ttl=300, show_spinner=False)
def fetch_ls(symbol, src):
    """Retail (all accounts) and top-trader (positions) long/short ratio. None = unavailable."""
    out = {'retail': None, 'top': None}
    try:
        if src == "binance":
            g = _get(f"{BINANCE}/futures/data/globalLongShortAccountRatio",
                     {"symbol": symbol, "period": "1h", "limit": 1})
            if g:
                out['retail'] = float(g[-1]['longShortRatio'])
        elif src == "okx":
            d = _get(f"{OKX}/api/v5/rubik/stat/contracts/long-short-account-ratio",
                     {"ccy": symbol[:-4], "period": "1H"})
            if d.get("code") == "0" and d.get("data"):
                out['retail'] = float(d["data"][0][1])
    except Exception:
        pass
    try:
        if src == "binance":
            t = _get(f"{BINANCE}/futures/data/topLongShortPositionRatio",
                     {"symbol": symbol, "period": "1h", "limit": 1})
            if t:
                out['top'] = float(t[-1]['longShortRatio'])
    except Exception:
        pass
    return out


@st.cache_data(ttl=20, show_spinner=False)
def fetch_depth(symbol, src):
    """Spread % and order-book imbalance (symmetric price range). None = unavailable."""
    try:
        if src == "binance":
            d = _get(f"{BINANCE}/fapi/v1/depth", {"symbol": symbol, "limit": 100})
            bids = [(float(p), float(q)) for p, q in d['bids']]
            asks = [(float(p), float(q)) for p, q in d['asks']]
        elif src == "okx":
            d = _get(f"{OKX}/api/v5/market/books", {"instId": symbol[:-4] + "-USDT-SWAP", "sz": 100})
            if d.get("code") != "0" or not d.get("data"):
                return None
            bids = [(float(r[0]), float(r[1])) for r in d['data'][0]['bids']]
            asks = [(float(r[0]), float(r[1])) for r in d['data'][0]['asks']]
        else:
            return None
        if not bids or not asks:
            return None
        mid = (bids[0][0] + asks[0][0]) / 2
        rng = min(0.01 * mid, mid - bids[-1][0], asks[-1][0] - mid)  # dono taraf barabar range
        if rng <= 0:
            return None
        bd = sum(p * q for p, q in bids if p >= mid - rng)
        ad = sum(p * q for p, q in asks if p <= mid + rng)
        imb = (bd - ad) / (bd + ad) if (bd + ad) > 0 else 0.0
        return {'spread': (asks[0][0] - bids[0][0]) / mid * 100, 'imb': imb, 'range_pct': rng / mid * 100}
    except Exception:
        return None


@st.cache_data(ttl=120, show_spinner=False)
def fetch_oi(symbol, src):
    """Open interest history per hour (ascending). Empty df = data unavailable."""
    try:
        if src == "binance":
            d = _get(f"{BINANCE}/futures/data/openInterestHist",
                     {"symbol": symbol, "period": "1h", "limit": 30})
            if d:
                df = pd.DataFrame(d)
                df['ts'] = pd.to_numeric(df['timestamp'])
                df['oi'] = pd.to_numeric(df['sumOpenInterest'])
                return df[['ts', 'oi']].sort_values('ts').reset_index(drop=True)
        if src == "okx":
            d = _get(f"{OKX}/api/v5/rubik/stat/contracts/open-interest-volume",
                     {"ccy": symbol[:-4], "period": "1H"})
            if d.get("code") == "0" and d.get("data"):
                df = pd.DataFrame([r[:3] for r in d["data"]], columns=['ts', 'oi', 'vol'])
                df['ts'] = pd.to_numeric(df['ts'])
                df['oi'] = pd.to_numeric(df['oi'])
                return df[['ts', 'oi']].sort_values('ts').reset_index(drop=True)
    except Exception:
        pass
    return pd.DataFrame()


def data_health(df):
    """Candle freshness + gaps. Bad data = bad trade."""
    if df is None or df.empty:
        return None
    now = pd.Timestamp.now(tz='UTC').tz_localize(None)
    last = df['time'].iloc[-1]
    age_min = (now - (last + pd.Timedelta(hours=1))).total_seconds() / 60
    gaps = int((df['time'].diff() > pd.Timedelta(hours=1, minutes=30)).sum())
    return {'last': last, 'age_min': age_min, 'gaps': gaps, 'bars': len(df)}


# ==================== INDICATORS (own implementation, no pandas_ta) ====================
def ema(s, n):
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi(close, n=14):
    d = close.diff()
    au = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    ad = (-d).clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = au / ad.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.mask((ad == 0) & (au > 0), 100.0).mask((ad == 0) & (au == 0), 50.0)


def atr(df, n=14):
    pc = df['close'].shift(1)
    tr = pd.concat([df['high'] - df['low'], (df['high'] - pc).abs(), (df['low'] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def adx(df, n=14):
    up = df['high'].diff()
    dn = -df['low'].diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    mdm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    a = atr(df, n)
    pdi = 100 * pdm.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / a
    mdi = 100 * mdm.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / a
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def htf_trend(df):
    """4H trend 1h candles se banta hai. Har 4H candle sirf COMPLETE hone ke baad use hoti hai (no lookahead)."""
    zeros = np.zeros(len(df), dtype=int)
    g = df.set_index('time')['close'].resample(pd.Timedelta(hours=4))
    o4 = pd.DataFrame({'close': g.last(), 'n': g.count()})
    o4 = o4[o4['n'] == 4]
    if len(o4) < 60:
        return zeros
    e20, e50 = ema(o4['close'], 20), ema(o4['close'], 50)
    tr = pd.Series(0, index=o4.index)
    tr[(e20 > e50) & (o4['close'] > e50)] = 1
    tr[(e20 < e50) & (o4['close'] < e50)] = -1
    tr.index = tr.index + pd.Timedelta(hours=4)  # known only when the 4H candle closes
    close_times = pd.Index(df['time'] + pd.Timedelta(hours=1))
    return tr.reindex(close_times, method='ffill').fillna(0).astype(int).values


# ==================== SIGNAL ENGINE (price-only = backtestable) ====================
def build_signals(df):
    """Har row sirf apne se pehle ka data use karti hai -> live aur backtest bilkul same logic."""
    d = df.copy().reset_index(drop=True)
    c, o, h, l, v = d['close'], d['open'], d['high'], d['low'], d['volume']
    d['ema50'], d['ema200'] = ema(c, 50), ema(c, 200)
    d['atr'], d['adx'], d['rsi'] = atr(d), adx(d), rsi(c)
    d['t4'] = htf_trend(d)
    d['vsma'] = v.rolling(20).mean().shift(1)
    d['dh'] = h.rolling(20).max().shift(1)
    d['dl'] = l.rolling(20).min().shift(1)

    # swing points: confirmed only SWING_K candles after they form
    w = 2 * SWING_K + 1
    d['sw_low'] = l.where(l == l.rolling(w, center=True).min()).shift(SWING_K).ffill()
    d['sw_high'] = h.where(h == h.rolling(w, center=True).max()).shift(SWING_K).ffill()

    rng = (h - l).replace(0, np.nan)
    pos_up = (c - l) / rng
    pos_dn = (h - c) / rng
    up_ok = (d['t4'] == 1) & (c > d['ema200']) & (d['ema50'] > d['ema200']) & (d['adx'] > ADX_MIN)
    dn_ok = (d['t4'] == -1) & (c < d['ema200']) & (d['ema50'] < d['ema200']) & (d['adx'] > ADX_MIN)

    pb_long = (up_ok & (l.rolling(3).min() <= d['ema50']) & (c > d['ema50']) & (c > o)
               & (c > h.shift(1)) & d['rsi'].between(40, 65))
    pb_short = (dn_ok & (h.rolling(3).max() >= d['ema50']) & (c < d['ema50']) & (c < o)
                & (c < l.shift(1)) & d['rsi'].between(35, 60))
    bo_long = (up_ok & (c > d['dh']) & (v > 1.3 * d['vsma']) & (pos_up > 0.7)
               & ((c - d['dh']) < d['atr']) & (d['rsi'] < 75))
    bo_short = (dn_ok & (c < d['dl']) & (v > 1.3 * d['vsma']) & (pos_dn > 0.7)
                & ((d['dl'] - c) < d['atr']) & (d['rsi'] > 25))

    side = np.zeros(len(d), dtype=int)
    setup = np.array([""] * len(d), dtype=object)
    for cond, s, name in ((pb_long, 1, "Pullback"), (pb_short, -1, "Pullback"),
                          (bo_long, 1, "Breakout"), (bo_short, -1, "Breakout")):
        m = np.asarray(cond, dtype=bool) & (side == 0)
        side[m] = s
        setup[m] = name
    d['side'], d['setup'] = side, setup
    d['sl_raw'] = np.where(side == 1, d['sw_low'] - 0.2 * d['atr'],
                           np.where(side == -1, d['sw_high'] + 0.2 * d['atr'], np.nan))
    return d


def make_trade(side, entry, sl_raw, atr_v, rr, cost_pct):
    """Structure stop, 1-3 ATR ke andar. Warna trade reject."""
    if side == 0 or atr_v is None or not np.isfinite(atr_v) or atr_v <= 0:
        return None
    if sl_raw is None or not np.isfinite(sl_raw):
        dist = 1.5 * atr_v
    else:
        dist = (entry - sl_raw) if side == 1 else (sl_raw - entry)
        if dist <= 0:
            dist = 1.5 * atr_v
    if dist > MAX_RISK_ATR * atr_v:
        return None
    dist = max(dist, MIN_RISK_ATR * atr_v)
    cost_r = 2 * cost_pct / 100 * entry / dist
    if cost_r > MAX_COST_R:
        return None
    return {'side': side, 'entry': float(entry), 'sl': float(entry - side * dist),
            'tp': float(entry + side * rr * dist), 'risk': float(dist), 'rr': rr,
            'cost_r': float(cost_r), 'risk_pct': float(dist / entry * 100)}


def size_position(capital, risk_pct, tr):
    risk_amt = capital * risk_pct / 100
    per_unit = tr['risk'] * (1 + tr['cost_r'])  # stop loss + fees = total loss per unit
    units = risk_amt / per_unit
    value = units * tr['entry']
    lev = value / capital if capital > 0 else 0
    liq_pct = 100 / lev if lev > 0 else np.inf  # rough isolated-margin liquidation distance
    safe = tr['risk_pct'] <= 0.5 * liq_pct
    return {'risk_amt': risk_amt, 'units': units, 'value': value, 'lev': lev, 'safe': safe}


def live_plan(d, price, rr, cost_pct):
    last = d.iloc[-1]
    side = int(last['side'])
    if side == 0:
        return None
    atr_v, close, slr = float(last['atr']), float(last['close']), float(last['sl_raw'])
    plan = {'side': side, 'setup': last['setup'], 'close': close, 'atr': atr_v,
            'status': 'READY', 'note': '', 'trade': None}
    if (side == 1 and price <= slr) or (side == -1 and price >= slr):
        plan.update(status='INVALIDATED', note='Price structural stop ke paar chala gaya')
        return plan
    if side * (price - close) > CHASE_ATR * atr_v:
        plan.update(status='MISSED', note='Price signal se >0.5 ATR bhaag gaya — chase mat karo')
        return plan
    tr = make_trade(side, price, slr, atr_v, rr, cost_pct)
    if tr is None:
        plan.update(status='SKIP', note='Stop 3 ATR se wide hai ya fees move ka bohat hissa kha jayengi')
        return plan
    plan['trade'] = tr
    return plan


def why_no_trade(last):
    t4 = int(last['t4'])
    r = [f"4H trend: {'UP' if t4 == 1 else 'DOWN' if t4 == -1 else 'NEUTRAL'}",
         f"ADX {last['adx']:.1f} " + ("(trend ok)" if last['adx'] > ADX_MIN else f"(< {ADX_MIN} = chop, no trade)"),
         "Price " + ("above" if last['close'] > last['ema200'] else "below") + f" EMA200 (${last['ema200']:,.4f})",
         f"RSI {last['rsi']:.1f}"]
    r.append("Setup (pullback ya breakout) abhi complete nahi — intezar karo.")
    return r


# ==================== MARKET INTELLIGENCE (derivatives + spot + positioning) ====================
# Ye reads HEURISTICS hain (price-core ki tarah backtest nahi ho sakte: exchanges sirf ~30 din ka OI/flow/L-S deti hain).
# Isliye ye sirf grade/veto ke liye use hote hain, trade signal banane ke liye nahi.
STATE_ICON = {"bull": "🟢 Bullish", "bear": "🔴 Bearish", "neutral": "⚪ Neutral",
              "info": "ℹ️ Info", "warn": "⚠️ Warning", "na": "❔ No data"}


def _spot_lookup(sym, smap):
    for pref, mult in (("1000000", 1e6), ("1000", 1e3)):
        if sym.startswith(pref) and sym[len(pref):] in smap:
            return smap[sym[len(pref):]], mult
    return smap.get(sym), 1.0


def _ret(df, bars):
    if df is None or len(df) <= bars:
        return None
    return float(df['close'].iloc[-1] / df['close'].iloc[-1 - bars] - 1)


def market_intel(sym, src, df, price, vol_usd, spot_map, btc_df, btc_t4):
    """Saare parameters ek list mein: name, value, state(bull/bear/neutral/info/warn/na), note."""
    P = []

    def add(name, short, value, state, note="", counts=True):
        P.append({'name': name, 'short': short, 'value': value, 'state': state, 'note': note, 'counts': counts})

    ret12, ret24 = _ret(df, OVERLAY_BARS), _ret(df, 24)

    # 1) Funding (futures)
    info, hist = fetch_perp_info(sym, src), fetch_funding_hist(sym, src)
    if not info:
        add("Funding (now)", "Fund", "DATA UNAVAILABLE", "na")
    else:
        f = info['funding']
        s_ = "bear" if f > 0.05 else "bull" if f < -0.05 else "neutral"
        add("Funding (now)", "Fund", f"{f:+.4f}%", s_,
            {"bear": "longs crowded (contrarian bearish)", "bull": "shorts crowded (contrarian bullish)",
             "neutral": "normal range"}[s_])
        if info.get('next', 0) > 0:
            mins = (info['next'] - time.time() * 1000) / 60000
            add("Next funding", "Nxt", f"{mins:.0f} min mein" if mins > 0 else "ab", "info",
                "settlement se pehle entry = funding pay/receive hoti hai", counts=False)
    if len(hist) >= 6:
        a, b = float(np.mean(hist[-3:])), float(np.mean(hist[-6:-3]))
        tr_ = "rising" if a > b + 0.005 else "falling" if a < b - 0.005 else "flat"
        add("Funding trend", "FTr", f"{a:+.4f}% vs {b:+.4f}% ({tr_})", "info",
            "last 3 vs pichli 3 settlements", counts=False)

    # 2) Spot vs futures
    sp, mult = _spot_lookup(sym, spot_map or {})
    if sp and np.isfinite(sp[0]) and sp[0] > 0:
        prem = (price / mult - sp[0]) / sp[0] * 100
        if abs(prem) > 10:  # itna farq = alag token / galat match
            add("Spot data", "Spot", "spot symbol match nahi hua", "na", counts=False)
        else:
            s_ = "bear" if prem > 0.30 else "bull" if prem < -0.30 else "neutral"
            add("Perp vs spot premium", "Prem", f"{prem:+.3f}%", s_,
                "futures spot se mehenge = leverage demand; discount = leverage dar", counts=False)
            if sp[1] and sp[1] > 0 and vol_usd > 0:
                add("Perp / spot volume", "P/S", f"{vol_usd / sp[1]:.1f}x", "info",
                    "zyada = leverage-driven; kam = spot-led (healthier)", counts=False)
    else:
        add("Spot data", "Spot", "DATA UNAVAILABLE / spot market nahi", "na", counts=False)

    # 3) Open interest x price quadrant
    oi = fetch_oi(sym, src)
    if oi.empty or len(oi) <= OVERLAY_BARS or ret12 is None:
        add("OI vs price (12h)", "OI", "DATA UNAVAILABLE", "na")
    else:
        oc = float(oi['oi'].iloc[-1] / oi['oi'].iloc[-1 - OVERLAY_BARS] - 1)
        val = f"price {ret12*100:+.2f}%, OI {oc*100:+.2f}%"
        if abs(oc) < 0.005:
            add("OI vs price (12h)", "OI", val + " — OI flat", "neutral", "naye positions nahi ban rahe")
        elif ret12 > 0 and oc > 0:
            add("OI vs price (12h)", "OI", val + " — LONG BUILDUP", "bull", "naye longs, rally mein dum")
        elif ret12 < 0 and oc > 0:
            add("OI vs price (12h)", "OI", val + " — SHORT BUILDUP", "bear", "naye shorts, girawat mein dum")
        elif ret12 > 0:
            add("OI vs price (12h)", "OI", val + " — SHORT COVERING", "bear", "rally sirf shorts band hone se: kamzor, fade ho sakti hai")
        else:
            add("OI vs price (12h)", "OI", val + " — LONG UNWIND", "bull", "girawat sirf longs band hone se: kamzor, bounce ho sakta hai")

    # 4) Taker flow (CVD proxy)
    fl = fetch_flow(sym, src)
    if fl.empty or len(fl) < OVERLAY_BARS:
        add("Taker flow (12h)", "Flow", "DATA UNAVAILABLE", "na")
        add("CVD vs price", "CVD", "DATA UNAVAILABLE", "na", counts=False)
    else:
        t = fl.tail(OVERLAY_BARS)
        buy, sell = float(t['buy'].sum()), float(t['sell'].sum())
        ratio = buy / sell if sell > 0 else np.nan
        s_ = "bull" if ratio > 1.05 else "bear" if ratio < 0.95 else "neutral"
        add("Taker flow (12h)", "Flow", f"buy/sell {ratio:.2f} (net {buy - sell:,.0f})", s_,
            "aggressive buyers vs sellers")
        if len(fl) >= 24:
            t24 = fl.tail(24)
            r24 = float(t24['buy'].sum() / t24['sell'].sum()) if t24['sell'].sum() > 0 else np.nan
            add("Taker flow (24h)", "F24", f"buy/sell {r24:.2f}", "info", "12h se compare karo", counts=False)
        if ret12 is not None and ret12 > 0.005 and ratio < 0.95:
            add("CVD vs price", "CVD", "price upar, sellers dominate", "bear", "divergence: rally ko buyers support nahi kar rahe", counts=False)
        elif ret12 is not None and ret12 < -0.005 and ratio > 1.05:
            add("CVD vs price", "CVD", "price neeche, buyers dominate", "bull", "divergence: girawat mein buyers absorb kar rahe", counts=False)
        else:
            add("CVD vs price", "CVD", "price aur flow aligned", "neutral", counts=False)

    # 5) Positioning: retail vs top traders
    ls = fetch_ls(sym, src)
    r, tp = ls.get('retail'), ls.get('top')
    if r is None:
        add("Retail long/short", "Rtl", "DATA UNAVAILABLE", "na")
    else:
        s_ = "bear" if r > 2.0 else "bull" if r < 0.6 else "neutral"
        add("Retail long/short", "Rtl", f"{r:.2f}", s_, "contrarian: crowd extreme par ulta chalta hai")
    if tp is None:
        add("Top traders L/S (positions)", "Top", "DATA UNAVAILABLE (sirf Binance)", "na")
    else:
        s_ = "bull" if tp > 1.2 else "bear" if tp < 0.85 else "neutral"
        add("Top traders L/S (positions)", "Top", f"{tp:.2f}", s_, "bade accounts ke saath chalo (heuristic)")
    if r is not None and tp is not None:
        if r > 1.5 and tp < 1.0:
            add("Smart vs retail", "S/R", "retail long, top traders nahi", "bear", "retail trap ka risk", counts=False)
        elif r < 0.8 and tp > 1.1:
            add("Smart vs retail", "S/R", "retail short, top traders long", "bull", "squeeze ka imkaan", counts=False)
        else:
            add("Smart vs retail", "S/R", "aligned", "neutral", counts=False)

    # 6) Liquidity / order book
    dp = fetch_depth(sym, src)
    if dp is None:
        add("Spread", "Sprd", "DATA UNAVAILABLE", "na")
        add("Order-book imbalance", "Book", "DATA UNAVAILABLE", "na", counts=False)
    else:
        add("Spread", "Sprd", f"{dp['spread']:.3f}%", "warn" if dp['spread'] > 0.10 else "neutral",
            "0.10% se zyada = buri fills")
        s_ = "bull" if dp['imb'] > 0.25 else "bear" if dp['imb'] < -0.25 else "neutral"
        add("Order-book imbalance", "Book", f"{dp['imb']:+.2f} (±{dp['range_pct']:.2f}% range)", s_,
            "spoof ho sakta hai: sirf info", counts=False)

    # 7) Relative strength vs BTC
    if sym == "BTCUSDT" or btc_df is None or len(btc_df) < 80 or len(df) < 80:
        add("Relative strength vs BTC", "RS", "yehi BTC hai" if sym == "BTCUSDT" else "DATA UNAVAILABLE", "na")
    else:
        bs = btc_df.set_index('time')['close']
        t_now = df['time'].iloc[-1]
        try:
            b_now, b24, b72 = bs.loc[t_now], bs.loc[t_now - pd.Timedelta(hours=24)], bs.loc[t_now - pd.Timedelta(hours=72)]
            c_now, c24, c72 = df['close'].iloc[-1], df['close'].iloc[-25], df['close'].iloc[-73]
            rs24, rs72 = (c_now / c24 - b_now / b24) * 100, (c_now / c72 - b_now / b72) * 100
            s_ = "bull" if (rs24 > 1 and rs72 > 0) else "bear" if (rs24 < -1 and rs72 < 0) else "neutral"
            add("Relative strength vs BTC", "RS", f"24h {rs24:+.2f}%, 72h {rs72:+.2f}%", s_,
                "coin BTC se mazboot/kamzor (price-based)")
        except Exception:
            add("Relative strength vs BTC", "RS", "DATA UNAVAILABLE", "na")

    # 8) BTC market trend
    if sym == "BTCUSDT":
        add("BTC 4H trend", "BTC", "yehi BTC hai", "na")
    else:
        add("BTC 4H trend", "BTC", {1: "UP", -1: "DOWN", 0: "NEUTRAL"}[int(btc_t4)],
            "bull" if btc_t4 == 1 else "bear" if btc_t4 == -1 else "neutral", "alts aksar BTC ke saath chalte hain")

    # 9) Info-only context
    if len(df) > 25:
        vs = float(df['volume'].iloc[-1] / df['volume'].iloc[-21:-1].mean())
        add("Last-candle volume", "Vol", f"{vs:.2f}x avg20", "info", "breakout ke liye >1.3x", counts=False)
        a_ = atr(df)
        rank = float((a_.tail(100) <= a_.iloc[-1]).mean() * 100)
        add("Volatility (ATR %)", "ATR", f"{a_.iloc[-1] / df['close'].iloc[-1] * 100:.2f}% (pctile {rank:.0f})", "info",
            "pctile 80+ = volatile, size chhota rakho", counts=False)
        hi, lo = float(df['high'].tail(24).max()), float(df['low'].tail(24).min())
        pos = (float(df['close'].iloc[-1]) - lo) / (hi - lo) * 100 if hi > lo else 50.0
        add("24h range position", "Rng", f"{pos:.0f}%", "info", "0% = 24h low, 100% = 24h high", counts=False)
    return P


def intel_grade(P, side):
    """A/B/C/SKIP. Sirf counting parameters. Core signal ko sirf ghata sakta hai."""
    mine, opp = ("bull", "bear") if side == 1 else ("bear", "bull")
    cp = [p for p in P if p['counts']]
    good = sum(1 for p in cp if p['state'] == mine)
    bad = sum(1 for p in cp if p['state'] == opp or p['state'] == 'warn')
    known = sum(1 for p in cp if p['state'] != 'na')
    if known < 3:
        g = "C"
    elif bad >= 3:
        g = "SKIP"
    elif bad == 2:
        g = "C"
    elif bad == 0 and good >= 2:
        g = "A"
    else:
        g = "B"
    return g, good, bad, known


def intel_marks(P, side):
    mine, opp = ("bull", "bear") if side == 1 else ("bear", "bull")
    def m(p):
        if p['state'] == 'na':
            return "?"
        if p['state'] == opp or p['state'] == 'warn':
            return "✗"
        return "✓" if p['state'] == mine else "·"
    return " ".join(f"{p['short']}{m(p)}" for p in P if p['counts'])


def _pv(P, name):
    for p in P:
        if p['name'] == name:
            return p['value']
    return "-"


def btc_trend(src):
    df = fetch_klines("BTCUSDT", 498, src)
    if len(df) < 300:
        return 0
    return int(build_signals(df)['t4'].iloc[-1])


def analyze_symbol(sym, src, price, rr, cost_pct, btc_t4, vol_usd, spot_map, btc_df):
    """('signal', row) / ('watch', row) / None."""
    try:
        df = fetch_klines(sym, 498, src)
        if len(df) < 300:
            return None
        d = build_signals(df)
        last = d.iloc[-1]
        side = int(last['side'])
        if side == 0:  # setup ban raha hai? (trend + ADX ok, price EMA50 ke paas)
            t4 = int(last['t4'])
            if t4 != 0 and last['adx'] > ADX_MIN:
                dist = (last['close'] - last['ema50']) / last['atr']
                if (t4 == 1 and last['close'] > last['ema200'] and -0.3 <= dist <= 0.8) or \
                   (t4 == -1 and last['close'] < last['ema200'] and -0.8 <= dist <= 0.3):
                    return ('watch', {'Symbol': sym, 'Side': 'LONG' if t4 == 1 else 'SHORT',
                                      'Price': round(float(last['close']), 6), 'ADX': round(float(last['adx']), 1),
                                      'EMA50 dist (ATR)': round(float(dist), 2), 'Vol $M': round(vol_usd / 1e6, 1)})
            return None
        plan = live_plan(d, price, rr, cost_pct)
        if plan is None or plan['status'] != 'READY':
            return None
        P = market_intel(sym, src, df, price, vol_usd, spot_map, btc_df, btc_t4)
        grade, good, bad, known = intel_grade(P, plan['side'])
        tr = plan['trade']
        return ('signal', {'Symbol': sym, 'Side': 'LONG' if plan['side'] == 1 else 'SHORT', 'Setup': plan['setup'],
                           'Grade': grade, 'Entry': round(tr['entry'], 6), 'SL': round(tr['sl'], 6),
                           'TP': round(tr['tp'], 6), 'Risk %': round(tr['risk_pct'], 2),
                           'ADX': round(float(last['adx']), 1), 'Vol $M': round(vol_usd / 1e6, 1),
                           'Good': good, 'Bad': bad, 'Funding': _pv(P, "Funding (now)"),
                           'OI read': _pv(P, "OI vs price (12h)"), 'Spot prem': _pv(P, "Perp vs spot premium"),
                           'Checks': intel_marks(P, plan['side'])})
    except Exception:
        return None


# ==================== BACKTEST (walk-forward, fees included) ====================
def backtest(d, rr=2.0, cost_pct=0.07, max_hold=MAX_HOLD, warmup=WARMUP):
    n = len(d)
    o, h, l, c = d['open'].values, d['high'].values, d['low'].values, d['close'].values
    side, atrv, slr, tm = d['side'].values, d['atr'].values, d['sl_raw'].values, d['time'].values
    setups = d['setup'].values
    out = []
    i = warmup
    while i < n - 1:
        s = int(side[i])
        if s == 0:
            i += 1
            continue
        entry = o[i + 1]  # signal candle close ke baad agli candle ke open par entry
        tr = make_trade(s, entry, slr[i], atrv[i], rr, cost_pct)
        if tr is None:
            i += 1
            continue
        j_end = min(i + max_hold, n - 1)
        exit_px, reason, j = None, 'TIME', j_end
        for j in range(i + 1, j_end + 1):
            if s == 1:
                if l[j] <= tr['sl']:  # SL pehle check = conservative
                    exit_px, reason = min(tr['sl'], o[j]), 'SL'
                    break
                if h[j] >= tr['tp']:
                    exit_px, reason = tr['tp'], 'TP'
                    break
            else:
                if h[j] >= tr['sl']:
                    exit_px, reason = max(tr['sl'], o[j]), 'SL'
                    break
                if l[j] <= tr['tp']:
                    exit_px, reason = tr['tp'], 'TP'
                    break
        if exit_px is None:
            exit_px = c[j_end]
            j = j_end
        r = s * (exit_px - entry) / tr['risk'] - tr['cost_r']
        out.append({'time': pd.Timestamp(tm[i + 1]), 'side': 'LONG' if s == 1 else 'SHORT',
                    'setup': setups[i], 'entry': entry, 'exit': exit_px, 'reason': reason,
                    'bars': j - i, 'r': r})
        i = j
    return pd.DataFrame(out)


def bt_stats(tr):
    if tr is None or len(tr) == 0:
        return None
    r = tr['r'].values
    n = len(r)
    gp, gl = r[r > 0].sum(), -r[r < 0].sum()
    eq = np.cumsum(r)
    peak = np.maximum.accumulate(np.r_[0.0, eq])[1:]
    sd = r.std(ddof=1) if n > 1 else 0.0
    half = n // 2
    return {'n': n, 'win': float((r > 0).mean() * 100), 'exp': float(r.mean()), 'total': float(r.sum()),
            'pf': float(gp / gl) if gl > 0 else float('inf'), 'dd': float((peak - eq).max()),
            't': float(r.mean() / (sd / np.sqrt(n))) if sd > 0 else 0.0,
            'h1': float(r[:half].mean()) if half > 0 else float('nan'),
            'h2': float(r[half:].mean()), 'eq': eq}


def bt_verdict(s):
    if s['n'] < 30:
        return "warn", f"Sirf {s['n']} trades — statistically bekaar. Is result pe bharosa mat karo."
    if s['exp'] <= 0:
        return "bad", "NO EDGE: fees ke baad strategy loss mein hai. Real paisa mat lagao."
    if s['t'] < 2:
        return "warn", "Weak evidence (t-stat < 2): ye luck bhi ho sakti hai."
    if not (s['h1'] > 0 and s['h2'] > 0):
        return "warn", "Unstable: edge sirf data ke ek hisse mein hai."
    return "ok", "Edge possible — lekin pehle 2-4 hafte paper trade karo, phir chhota size."


def show_bt(trades, label):
    s = bt_stats(trades)
    if s is None:
        st.info(f"{label}: koi trade nahi bana.")
        return
    lvl, txt = bt_verdict(s)
    {"bad": st.error, "warn": st.warning, "ok": st.success}[lvl](txt)
    a, b = st.columns(2)
    a.metric("Trades", s['n'])
    b.metric("Win rate", f"{s['win']:.0f}%")
    a, b = st.columns(2)
    a.metric("Avg R / trade (net)", f"{s['exp']:+.3f}")
    b.metric("Profit factor", f"{s['pf']:.2f}" if np.isfinite(s['pf']) else "∞")
    a, b = st.columns(2)
    a.metric("Total R", f"{s['total']:+.1f}")
    b.metric("Max drawdown (R)", f"{s['dd']:.1f}")
    a, b = st.columns(2)
    a.metric("t-stat (>2 chahiye)", f"{s['t']:.2f}")
    b.metric("1st / 2nd half avg R", f"{s['h1']:+.2f} / {s['h2']:+.2f}")
    fig = go.Figure(go.Scatter(y=np.r_[0.0, s['eq']], mode='lines', name='Equity (R)'))
    fig.update_layout(height=260, template="plotly_dark", margin=dict(l=10, r=10, t=30, b=10),
                      title=f"{label} — equity curve (R)")
    st.plotly_chart(fig)
    st.caption("R = 1 unit risk. Sab numbers fees+slippage ke baad hain. Past result future ki guarantee nahi.")


def run_bt_symbol(sym, bars, src, rr, cost):
    df = fetch_klines(sym, bars, src)
    if len(df) < WARMUP + 100:
        return pd.DataFrame()
    t = backtest(build_signals(df), rr, cost)
    if not t.empty:
        t.insert(0, 'symbol', sym)
    return t


# ==================== MODE SELECTOR ====================
st.sidebar.markdown("## 🧭 Mode Selector")
MODE = st.sidebar.radio("Kaunsa mode?", ["📊 Crypto Dashboard", "🤖 AI Chatbot"], index=0)

if MODE == "🤖 AI Chatbot":
    st.title("🤖 AI Chatbot")
    if not _groq_ok or not _groq_key:
        st.error("groq library ya GROQ_API_KEY missing hai. Secrets configure karein.")
        st.stop()
    client = Groq(api_key=_groq_key)
    st.sidebar.header("⚙️ Chatbot Settings")
    model = st.sidebar.selectbox("Model", [
        "openai/gpt-oss-120b", "openai/gpt-oss-20b",
        "meta-llama/llama-4-maverick-17b-128e-instruct",
        "qwen/qwen3-32b", "moonshotai/kimi-k2-instruct"], index=0)
    temp = st.sidebar.slider("Temperature", 0.0, 1.5, 0.7, 0.1)
    sys_prompt = st.sidebar.text_area("System Prompt", "You are a helpful AI assistant.", height=100)
    if st.sidebar.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    if prompt := st.chat_input("Apna sawal likho..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            ph = st.empty()
            full = ""
            try:
                stream = client.chat.completions.create(
                    model=model, temperature=temp, stream=True,
                    messages=[{"role": "system", "content": sys_prompt}] + st.session_state.messages)
                for ch in stream:
                    if ch.choices[0].delta.content:
                        full += ch.choices[0].delta.content
                        ph.markdown(full + "▌")
                ph.markdown(full)
            except Exception as e:
                ph.error(f"Error: {e}")
        st.session_state.messages.append({"role": "assistant", "content": full})
    st.stop()

# ==================== DASHBOARD ====================
st.title("📊 Pro Crypto Dashboard v13")
st.caption(f"Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Closed candles only | Backtested price core + full market intelligence")

CAPITAL = st.sidebar.number_input("💰 Capital ($)", min_value=10.0, value=1000.0, step=100.0)
RISK_PCT = st.sidebar.number_input("⚠️ Risk per trade (%)", min_value=0.1, max_value=5.0, value=1.0, step=0.25)
RR = st.sidebar.slider("🎯 Reward : Risk", 1.5, 3.0, 2.0, 0.25)
COST = st.sidebar.number_input("💸 Cost per side (%) fee+slippage", min_value=0.0, max_value=0.5, value=0.07, step=0.01)
st.sidebar.caption("Risk 1% se zyada mat karo. Fees apne exchange tier ke hisaab se set karo.")

tickers, SRC = fetch_tickers()
if tickers.empty:
    st.error("Kisi bhi exchange se data nahi aaya (Binance/OKX/MEXC). Internet ya server region check karo.")
    st.stop()
price_map = dict(zip(tickers['symbol'], tickers['lastPrice']))
vol_map = dict(zip(tickers['symbol'], tickers['volume24'] / 1e6))
st.info(f"📡 Data source: **{SRC.upper()}** (saare candles + derivatives isi venue se)")
if SRC != "binance":
    st.warning("Binance se connect nahi hua (US-hosted server par Binance 451 se block karta hai). "
               f"{SRC.upper()} use ho raha hai. Funding/OI/flow ke kuch checks 'DATA UNAVAILABLE' ho sakte hain.")

# ---------- SCANNER ----------
st.caption(f"🌐 Universe: {len(tickers)} USDT perpetuals ({SRC.upper()})")
with st.expander("🎯 Scanner (backtested price core + full derivatives/spot intel)", expanded=True):
    a, b, c = st.columns(3)
    sc_top = a.selectbox("Kitne coins scan karein (volume order)", [20, 50, 100, 200, 300, "ALL"], index=2)
    sc_vol = b.number_input("Min 24h vol ($M)", 0.0, 1000.0, 5.0, 1.0)
    hide_skip = c.checkbox("SKIP grade chhupao", True)
    if st.button("🚀 Scan now", type="primary"):
        pool = tickers[tickers['volume24'] / 1e6 >= sc_vol]
        syms = pool['symbol'].tolist() if sc_top == "ALL" else pool['symbol'].head(int(sc_top)).tolist()
        if not syms:
            st.warning("Koi coin filter pass nahi hua.")
        else:
            btc_df_s = fetch_klines("BTCUSDT", 498, SRC)
            btc_t4_s = btc_trend(SRC)
            spot_s = fetch_spot_map(SRC)
            prog, stat = st.progress(0.0), st.empty()
            sigs, watch = [], []
            with ThreadPoolExecutor(max_workers=8) as ex:
                futs = [ex.submit(analyze_symbol, s_, SRC, price_map[s_], RR, COST, btc_t4_s,
                                  vol_map[s_] * 1e6, spot_s, btc_df_s) for s_ in syms]
                for k, f in enumerate(as_completed(futs), 1):
                    prog.progress(k / len(futs))
                    stat.text(f"Scanning... {k}/{len(futs)}")
                    r = f.result()
                    if r:
                        (sigs if r[0] == 'signal' else watch).append(r[1])
            stat.text(f"✅ {len(syms)} coins | BTC 4H: {['DOWN', 'NEUTRAL', 'UP'][btc_t4_s + 1]} | "
                      f"{len(sigs)} signals | {len(watch)} forming | spot data: {'ok' if spot_s else 'NA'}")
            st.session_state['scan'] = sigs
            st.session_state['watch'] = watch
    rows = st.session_state.get('scan')
    if rows is not None:
        if hide_skip:
            rows = [r for r in rows if r['Grade'] != 'SKIP']
        rows = sorted(rows, key=lambda r: (GRADE_ORDER[r['Grade']], -r['ADX']))
        if not rows:
            st.info("Abhi koi quality signal nahi. Ye normal hai — most of the time sahi jawab 'no trade' hota hai.")
        else:
            sdf = pd.DataFrame(rows)
            st.subheader(f"🟢 LONG ({int((sdf['Side'] == 'LONG').sum())})")
            st.dataframe(sdf[sdf['Side'] == 'LONG'], hide_index=True)
            st.subheader(f"🔴 SHORT ({int((sdf['Side'] == 'SHORT').sum())})")
            st.dataframe(sdf[sdf['Side'] == 'SHORT'], hide_index=True)
            st.download_button("⬇️ Scan CSV (forward-test log ke liye)", sdf.to_csv(index=False),
                               file_name=f"scan_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", mime="text/csv")
            st.caption("Grade A = 2+ reads saath, 0 khilaf | B = ok | C = 2 khilaf ya data kam | SKIP = 3+ khilaf. "
                       "Checks: ✓ saath, ✗ khilaf, · neutral, ? data nahi. Signal sirf closed candle par valid hai.")
    watch = st.session_state.get('watch')
    if watch:
        with st.expander(f"👀 Forming setups ({len(watch)}) — trigger candle ka intezar"):
            st.dataframe(pd.DataFrame(watch).sort_values('ADX', ascending=False), hide_index=True)
            st.caption("Ye signal NAHI hain. Trend + ADX ok hai aur price EMA50 ke paas pullback zone mein hai. "
                       "Trigger candle (EMA50 wapas cross + pichli candle ka high/low tootna) bane tab hi signal banta hai.")

st.markdown("---")

# ---------- COIN SELECTOR ----------
all_sym = tickers['symbol'].tolist()
c1, c2 = st.columns(2)
di = all_sym.index('BTCUSDT') if 'BTCUSDT' in all_sym else 0
sel_dd = c1.selectbox("Coin:", all_sym, index=di)
sel_txt = c2.text_input("Ya likho (e.g. ETHUSDT):")
SYM = sel_txt.strip().upper() if sel_txt.strip().upper() in all_sym else sel_dd
st.markdown(f"### ✅ `{SYM}`")

price = float(price_map[SYM])
m1, m2 = st.columns(2)
m1.metric("Price", f"${price:,.4f}", f"{float(tickers.loc[tickers['symbol'] == SYM, 'riseFallRate'].iloc[0]):.2f}%")
m2.metric("24h Volume", f"${vol_map[SYM]:,.1f}M")
if vol_map[SYM] < 20:
    st.warning("⚠️ LOW VOLUME (<$20M): slippage zyada, signals kam bharosemand. Avoid.")

df1 = fetch_klines(SYM, 498, SRC)
if len(df1) < 300:
    st.warning("Is coin ka kaafi data nahi mila.")
    st.stop()

health = data_health(df1)
with st.expander("🩺 Data health", expanded=health['age_min'] > 90 or health['gaps'] > 0):
    h1, h2, h3 = st.columns(3)
    h1.metric("Last closed candle (UTC)", health['last'].strftime("%m-%d %H:%M"))
    h2.metric("Closed ko guzre (min)", f"{health['age_min']:.0f}")
    h3.metric("Gaps", health['gaps'])
    if health['age_min'] > 90:
        st.error("Data stale hai — signal use mat karo.")

d = build_signals(df1)
last = d.iloc[-1]
plan = live_plan(d, price, RR, COST)
btc_df = fetch_klines("BTCUSDT", 498, SRC)
btc_t4 = btc_trend(SRC)
spot_map = fetch_spot_map(SRC)
P = market_intel(SYM, SRC, df1, price, vol_map[SYM] * 1e6, spot_map, btc_df, btc_t4)

st.header("🎯 Signal")
if plan is None:
    st.warning("## ⏸️ NO TRADE")
    for line in why_no_trade(last):
        st.write(f"• {line}")
    st.caption("Ye feature hai, bug nahi. Trade na lena bhi position hai.")
else:
    label = f"{'🟢 LONG' if plan['side'] == 1 else '🔴 SHORT'} — {plan['setup']}"
    if plan['status'] != 'READY':
        st.error(f"## {label} → {plan['status']}")
        st.write(plan['note'])
    else:
        tr = plan['trade']
        grade, good, bad, known = intel_grade(P, plan['side'])
        (st.success if grade == "A" else st.warning if grade in ("B", "C") else st.error)(f"## {label} | Grade {grade}")
        st.caption(f"Market intel: {good} saath | {bad} khilaf | {known} parameters ka data mila")
        if grade == "SKIP":
            st.write("3+ parameters trade ke khilaf — core signal hone ke bawajood SKIP.")
        a, b = st.columns(2)
        a.metric("Entry", f"${tr['entry']:,.4f}")
        b.metric("Stop Loss", f"${tr['sl']:,.4f}", f"-{tr['risk_pct']:.2f}%")
        a, b = st.columns(2)
        a.metric(f"Take Profit ({RR}R)", f"${tr['tp']:,.4f}")
        net_rr = (tr['rr'] - tr['cost_r']) / (1 + tr['cost_r'])
        b.metric("Net R:R (fees ke baad)", f"1 : {net_rr:.2f}")
        st.caption(f"Time stop: {MAX_HOLD}h. Breakeven win-rate ≈ {100 / (1 + net_rr):.0f}% — iske upar hi edge ban sakta hai.")

        sz = size_position(CAPITAL, RISK_PCT, tr)
        st.subheader("💼 Position Sizing")
        a, b = st.columns(2)
        a.metric("Risk amount", f"${sz['risk_amt']:.2f}")
        b.metric("Position value", f"${sz['value']:,.2f}")
        a, b = st.columns(2)
        a.metric("Size (units)", f"{sz['units']:,.4f} {SYM[:-4]}")
        b.metric("Leverage needed", f"{sz['lev']:.1f}x")
        if sz['lev'] > 10 or not sz['safe']:
            st.error("Leverage zyada / stop liquidation ke qareeb. Position chhoti karo ya zyada margin lagao.")

        opp = "bear" if plan['side'] == 1 else "bull"
        against = [p for p in P if p['counts'] and (p['state'] == opp or p['state'] == 'warn')]
        if against:
            st.subheader("❌ Khilaf parameters")
            for p in against:
                st.write(f"❌ **{p['name']}**: {p['value']} — {p['note']}")

st.header("📊 Context")
a, b = st.columns(2)
a.metric("4H trend", {1: "UP 📈", -1: "DOWN 📉", 0: "NEUTRAL"}[int(last['t4'])])
b.metric("ADX", f"{last['adx']:.1f}", "trending" if last['adx'] > ADX_MIN else "chop")
a, b = st.columns(2)
a.metric("RSI", f"{last['rsi']:.1f}")
b.metric("ATR", f"${last['atr']:,.4f}")

# ---------- MARKET INTELLIGENCE ----------
st.header("🧠 Market Intelligence")
bull_n = sum(1 for p in P if p['counts'] and p['state'] == 'bull')
bear_n = sum(1 for p in P if p['counts'] and p['state'] == 'bear')
na_n = sum(1 for p in P if p['state'] == 'na')
a, b = st.columns(2)
a.metric("🟢 Bullish reads", bull_n)
b.metric("🔴 Bearish reads", bear_n)
st.caption(f"{na_n} parameters ka data nahi mila (DATA UNAVAILABLE).")
st.dataframe(pd.DataFrame([{'Parameter': p['name'], 'Value': p['value'], 'Read': STATE_ICON[p['state']],
                            'Note': p['note']} for p in P]), hide_index=True)
st.caption("Ye reads heuristics hain (funding, OI, taker flow, L/S, spot premium, spread, RS vs BTC). "
           "Inka backtest nahi ho sakta, isliye ye sirf grade/veto karte hain — signal price-core se aata hai. "
           "Bullish reads zyada hone ka matlab BUY signal nahi hai.")

# ---------- CHART ----------
cd = d.tail(150)
fig = go.Figure(data=[go.Candlestick(x=cd['time'], open=cd['open'], high=cd['high'], low=cd['low'], close=cd['close'])])
fig.add_trace(go.Scatter(x=cd['time'], y=cd['ema50'], name='EMA50', line=dict(color='#00bfff', width=1)))
fig.add_trace(go.Scatter(x=cd['time'], y=cd['ema200'], name='EMA200', line=dict(color='orange', width=1)))
if plan and plan['status'] == 'READY':
    fig.add_hline(y=plan['trade']['sl'], line_color="red", annotation_text="SL")
    fig.add_hline(y=plan['trade']['tp'], line_color="#00ff88", annotation_text="TP")
    fig.add_hline(y=plan['trade']['entry'], line_dash="dot", line_color="white", annotation_text="Entry")
fig.update_layout(xaxis_rangeslider_visible=False, height=480, template="plotly_dark", margin=dict(l=10, r=10, t=30, b=10))
st.plotly_chart(fig)

# ---------- BACKTEST ----------
st.markdown("---")
st.header("🧪 Backtest — pehle saboot, phir paisa")
st.caption("Same engine, walk-forward, entry agli candle ke open par, SL pehle check, fees include.")
bt_bars = st.selectbox("History (1h candles)", [1000, 2000, 4000], index=1,
                       help="OKX par max ~1400 candles milti hain.")
if st.button(f"▶️ {SYM} par backtest"):
    with st.spinner("Backtest chal raha hai..."):
        t = run_bt_symbol(SYM, bt_bars, SRC, RR, COST)
    show_bt(t, SYM)
    if not t.empty:
        st.dataframe(t.tail(30).iloc[::-1], hide_index=True)

with st.expander("📦 Portfolio backtest (ek coin se zyada bharosemand)"):
    n_pf = st.selectbox("Top coins", [10, 20, 30, 50], index=1)
    if st.button("▶️ Portfolio backtest"):
        syms = tickers['symbol'].head(n_pf).tolist()
        prog = st.progress(0.0)
        frames = []
        with ThreadPoolExecutor(max_workers=3) as ex:
            futs = [ex.submit(run_bt_symbol, s, bt_bars, SRC, RR, COST) for s in syms]
            for k, f in enumerate(as_completed(futs), 1):
                prog.progress(k / len(futs))
                try:
                    r = f.result()
                    if r is not None and not r.empty:
                        frames.append(r)
                except Exception:
                    pass
        if frames:
            allt = pd.concat(frames).sort_values('time').reset_index(drop=True)
            show_bt(allt, f"Portfolio ({len(frames)} coins)")
            per = allt.groupby('symbol')['r'].agg(trades='count', avg_R='mean', total_R='sum').round(2)
            st.dataframe(per.sort_values('total_R', ascending=False))
            st.caption("Coins correlated hote hain, isliye t-stat thoda optimistic hai. Best coins chun kar trade mat karo — ye overfitting hai.")
        else:
            st.info("Koi trade nahi bana / data nahi aaya.")

st.sidebar.markdown("---")
st.sidebar.header("✅ v13 changes")
st.sidebar.markdown("""
1. **Saare coins scan** (ALL) + forming-setups watchlist
2. **Spot vs futures**: premium, volume ratio
3. **Funding** now/trend/next, **OI quadrant**, **taker flow + CVD divergence**
4. **Retail vs top-trader L/S**, spread, order-book imbalance
5. **Relative strength vs BTC**, volatility, range position
6. Grade A/B/C/SKIP + scan CSV (forward-test)
""")
if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()
