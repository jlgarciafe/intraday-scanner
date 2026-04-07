"""
═══════════════════════════════════════════════════════════════════════════════
  INTRADAY MOMENTUM SCANNER v3
  Fix: yfinance v0.2+ returns MultiIndex columns — now handled correctly.

  ⚠️  RISK DISCLAIMER:
  Consistently achieving >5% net daily returns is statistically rare and
  highly speculative. This scanner identifies momentum candidates only.
  Use exclusively for research. Never risk capital you cannot afford to lose.
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import logging
import time
import requests
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime, timezone
from typing import Optional
import yfinance as yf

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")
MARKET_CONTEXT     = os.getenv("MARKET_CONTEXT", "all").lower()
MIN_MOVE_PCT       = float(os.getenv("MIN_MOVE_PCT", "5.0"))
DRY_RUN            = os.getenv("DRY_RUN", "false").lower() == "true"

# ── Fee model ─────────────────────────────────────────────────────────────────
FEE_MODEL = {
    "us": {"commission_pct": 0.005, "spread_pct": 0.05,  "slippage_pct": 0.05},
    "uk": {"commission_pct": 0.100, "spread_pct": 0.10,  "slippage_pct": 0.10},
    "de": {"commission_pct": 0.100, "spread_pct": 0.10,  "slippage_pct": 0.10},
    "jp": {"commission_pct": 0.050, "spread_pct": 0.08,  "slippage_pct": 0.08},
    "es": {"commission_pct": 0.100, "spread_pct": 0.10,  "slippage_pct": 0.10},
    "hk": {"commission_pct": 0.080, "spread_pct": 0.10,  "slippage_pct": 0.10},
    "fr": {"commission_pct": 0.100, "spread_pct": 0.10,  "slippage_pct": 0.10},
}

# ── Ticker universe ────────────────────────────────────────────────────────────
TICKER_UNIVERSE = {
    "us": [
        "NVDA","AMD","TSLA","META","AMZN","GOOGL","MSFT","AAPL","NFLX","CRM",
        "PLTR","SNOW","RBLX","COIN","MSTR","SMCI","ARM","AVGO","MU","INTC",
        "MRNA","BNTX","REGN","BIIB","VRTX",
        "TQQQ","SQQQ","SPXL","UPRO","LABU","SOXL","TNA","FAS",
        "CRWD","DDOG","NET","ZS","PANW","OKTA",
        "XOM","CVX","OXY","HAL","SLB","FCX","NEM","GOLD",
    ],
    "uk": [
        "HSBA.L","BP.L","SHEL.L","AZN.L","ULVR.L","RIO.L","GSK.L",
        "BARC.L","LLOY.L","NWG.L","TSCO.L","VOD.L","BT-A.L","IAG.L",
        "FERG.L","CRH.L","EXPN.L","SGRO.L","LGEN.L","PHNX.L",
    ],
    "de": [
        "SAP.DE","SIE.DE","ALV.DE","MRK.DE","BAYN.DE","BMW.DE","MBG.DE",
        "VOW3.DE","DTE.DE","DBK.DE","CON.DE","ADS.DE","EOAN.DE","RWE.DE",
        "BAS.DE","HEI.DE","FRE.DE","MUV2.DE","DPW.DE","IFX.DE",
    ],
    "jp": [
        "7203.T","9984.T","6758.T","6861.T","8306.T","9432.T","6501.T",
        "7974.T","4519.T","8316.T","6902.T","4661.T","9433.T","8035.T",
        "6954.T","7267.T","4543.T","2914.T","9020.T","4568.T",
    ],
    "es": [
        # Ibex 35 — Spain (.MC)
        "SAN.MC","BBVA.MC","ITX.MC","IBE.MC","REP.MC","TEF.MC","AMS.MC",
        "ANA.MC","ELE.MC","CLNX.MC","FER.MC","GRF.MC","IAG.MC","MAP.MC",
        "MEL.MC","MTS.MC","NTGY.MC","RED.MC","ROVI.MC","SAB.MC",
    ],
    "hk": [
        # Hang Seng — Hong Kong (.HK)
        "0700.HK","0005.HK","0939.HK","1299.HK","0941.HK","2318.HK","0388.HK",
        "1398.HK","2628.HK","0003.HK","0011.HK","0002.HK","0016.HK","0027.HK",
        "1810.HK","9988.HK","0175.HK","1177.HK","2020.HK","6862.HK",
    ],
    "fr": [
        # CAC 40 — France (.PA)
        "MC.PA","OR.PA","TTE.PA","SAN.PA","AIR.PA","BNP.PA","SU.PA","RI.PA",
        "CAP.PA","ACA.PA","BN.PA","KER.PA","DG.PA","GLE.PA","HO.PA","LR.PA",
        "ML.PA","ORA.PA","PUB.PA","RMS.PA","SAF.PA","SGO.PA","STLA.PA",
        "STM.PA","SW.PA","VIE.PA","VIV.PA","WLN.PA","CS.PA","EDF.PA",
    ],
}


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA LAYER
# ═══════════════════════════════════════════════════════════════════════════════

def flatten_yf_df(raw: pd.DataFrame) -> pd.DataFrame:
    """
    yfinance v0.2+ returns MultiIndex columns like ('Close', 'HSBA.L').
    This flattens them to plain column names: 'Close', 'Open', etc.
    """
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    # Remove duplicate columns if any
    raw = raw.loc[:, ~raw.columns.duplicated()]
    return raw


def fetch_snapshot(ticker: str) -> Optional[pd.DataFrame]:
    """
    Fetch daily OHLCV data. Tries multiple periods for resilience.
    Flattens yfinance MultiIndex columns automatically.
    """
    for period in ["30d", "5d", "1mo"]:
        for attempt in range(2):
            try:
                raw = yf.download(
                    ticker,
                    period=period,
                    interval="1d",
                    auto_adjust=True,
                    progress=False,
                    timeout=15,
                )
                if raw is None or len(raw) < 3:
                    rows = len(raw) if raw is not None else 0
                    logger.info(f"    {ticker}: {rows} rows (period={period}, attempt={attempt+1})")
                    time.sleep(0.5)
                    continue

                df = flatten_yf_df(raw)

                # Verify required columns exist
                required = {"Open", "High", "Low", "Close", "Volume"}
                if not required.issubset(set(df.columns)):
                    logger.info(f"    {ticker}: missing columns — got {list(df.columns)}")
                    time.sleep(0.5)
                    continue

                logger.info(f"    {ticker}: {len(df)} days OK (period={period})")
                return df

            except Exception as e:
                logger.info(f"    {ticker}: error — {type(e).__name__}: {str(e)[:100]}")
            time.sleep(0.5)

    logger.info(f"    {ticker}: ❌ all attempts failed")
    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  MATHEMATICAL FILTERS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_atr(df: pd.DataFrame, period: int = 10) -> float:
    """ATR as % of latest close."""
    if len(df) < 3:
        return 0.0
    try:
        period   = min(period, len(df) - 1)
        high     = df["High"].astype(float).values
        low      = df["Low"].astype(float).values
        close    = df["Close"].astype(float).values
        tr_list  = []
        for i in range(1, len(close)):
            tr = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i]  - close[i-1]),
            )
            tr_list.append(tr)
        atr = sum(tr_list[-period:]) / period
        return (atr / close[-1]) * 100 if close[-1] > 0 else 0.0
    except Exception as e:
        logger.debug(f"ATR error: {e}")
        return 0.0


def compute_rvol(df: pd.DataFrame) -> float:
    """Relative volume vs recent average."""
    if len(df) < 3:
        return 1.0
    try:
        vols     = df["Volume"].astype(float).values
        lookback = min(20, len(vols) - 1)
        avg_vol  = sum(vols[-lookback-1:-1]) / lookback
        return float(vols[-1]) / avg_vol if avg_vol > 0 else 1.0
    except Exception:
        return 1.0


def compute_rsi(closes, period: int = 10) -> float:
    """RSI momentum indicator."""
    period = min(period, len(closes) - 2)
    if period < 2:
        return 50.0
    gains, losses = [], []
    for i in range(1, len(closes)):
        d = float(closes[i]) - float(closes[i-1])
        gains.append(max(d, 0))
        losses.append(max(-d, 0))
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    if avg_loss == 0:
        return 100.0
    return 100 - (100 / (1 + avg_gain / avg_loss))


def analyse_ticker(ticker: str, df: pd.DataFrame) -> dict:
    """Compute all metrics for a ticker with verbose logging."""
    try:
        close  = df["Close"].astype(float).values
        open_  = df["Open"].astype(float).values
        high   = df["High"].astype(float).values
        low    = df["Low"].astype(float).values
        volume = df["Volume"].astype(float).values

        atr_pct    = compute_atr(df)
        rvol       = compute_rvol(df)
        rsi        = compute_rsi(close)

        day_return = ((close[-1] - close[-2]) / close[-2] * 100) if len(close) >= 2 and close[-2] > 0 else 0.0
        day_range  = ((high[-1] - low[-1]) / low[-1] * 100) if low[-1] > 0 else 0.0
        gap_pct    = ((open_[-1] - close[-2]) / close[-2] * 100) if len(close) >= 2 and close[-2] > 0 else 0.0

        # Composite score
        score = (
            min(atr_pct / 5.0, 1.0) * 30 +
            min(rvol / 3.0, 1.0) * 25 +
            min(abs(day_return) / 3.0, 1.0) * 25 +
            (1.0 if 60 <= rsi <= 80 else 0.6 if 40 <= rsi < 60 else 0.4) * 20
        )

        logger.info(
            f"    {ticker}: price={close[-1]:.2f} | ATR={atr_pct:.2f}% | "
            f"RVOL={rvol:.2f}x | RSI={rsi:.0f} | day={day_return:+.2f}% | "
            f"range={day_range:.2f}% | vol={int(volume[-1]):,} | score={score:.0f}"
        )

        return {
            "ticker":     ticker,
            "price":      round(float(close[-1]), 4),
            "atr_pct":    round(atr_pct, 2),
            "rvol":       round(rvol, 2),
            "rsi":        round(rsi, 1),
            "gap_pct":    round(gap_pct, 2),
            "day_return": round(day_return, 2),
            "day_range":  round(day_range, 2),
            "volume":     int(volume[-1]),
            "score":      round(score, 1),
        }
    except Exception as e:
        logger.info(f"    {ticker}: analysis error — {e}")
        return {}


def passes_filters(m: dict, market: str) -> tuple:
    """
    Return (passed, reason).
    Volume and RVOL filters are ALWAYS enforced — low volume = illiquid = untradeable.
    ATR and score filters only apply when MIN_MOVE_PCT > 0.
    """
    vol   = m.get("volume", 0)
    score = m.get("score", 0)
    atr   = m.get("atr_pct", 0)
    rvol  = m.get("rvol", 0)

    # ── Always enforced — liquidity gates ─────────────────────────────────────
    min_vol = 50_000 if market == "uk" else 100_000
    if vol < min_vol:
        return False, f"vol {vol:,} < {min_vol:,}"

    if rvol < 1.3:
        return False, f"RVOL {rvol:.2f}x < 1.3x"

    # ── Move threshold gates (only when target > 0) ───────────────────────────
    if MIN_MOVE_PCT > 0:
        if atr < MIN_MOVE_PCT * 0.7:
            return False, f"ATR {atr:.2f}% < {MIN_MOVE_PCT*0.7:.2f}%"
        if score < 35:
            return False, f"score {score:.0f} < 35"
    else:
        # No move threshold — just need minimum score sanity check
        if score < 20:
            return False, f"score {score:.0f} < 20"

    return True, "passed"


# ═══════════════════════════════════════════════════════════════════════════════
#  FEE MODEL
# ═══════════════════════════════════════════════════════════════════════════════

def net_yield(gross_pct: float, market: str) -> float:
    fees = FEE_MODEL.get(market, FEE_MODEL["us"])
    cost = (fees["commission_pct"] + fees["spread_pct"] + fees["slippage_pct"]) * 2
    return round(gross_pct - cost, 2)


# ═══════════════════════════════════════════════════════════════════════════════
#  SCANNER
# ═══════════════════════════════════════════════════════════════════════════════

def scan_market(market_key: str) -> list:
    tickers = TICKER_UNIVERSE.get(market_key, [])
    if not tickers:
        return []

    logger.info(f"\nScanning {market_key.upper()}: {len(tickers)} tickers | target >{MIN_MOVE_PCT}%")
    candidates, data_fail, filtered = [], 0, 0

    for i, ticker in enumerate(tickers, 1):
        logger.info(f"  [{i}/{len(tickers)}] {ticker}")
        df = fetch_snapshot(ticker)
        if df is None:
            data_fail += 1
            continue

        m = analyse_ticker(ticker, df)
        if not m:
            data_fail += 1
            continue

        passed, reason = passes_filters(m, market_key)
        if not passed:
            logger.info(f"    {ticker}: ❌ {reason}")
            filtered += 1
            continue

        logger.info(f"    {ticker}: ✅ CANDIDATE")
        candidates.append({**m, "market": market_key.upper(),
                           "net_yield": net_yield(m["atr_pct"], market_key)})
        time.sleep(0.2)

    logger.info(f"\n{market_key.upper()}: scanned={len(tickers)} | "
                f"data_fail={data_fail} | filtered={filtered} | candidates={len(candidates)}")
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates


def run_all_markets() -> list:
    if MARKET_CONTEXT == "all":
        markets = ["us", "uk", "de", "jp", "es", "hk", "fr"]
    elif MARKET_CONTEXT == "eu":
        markets = ["uk", "de", "es", "fr"]
    elif MARKET_CONTEXT == "asia":
        markets = ["jp", "hk"]
    elif MARKET_CONTEXT in TICKER_UNIVERSE:
        markets = [MARKET_CONTEXT]
    else:
        markets = ["us", "uk", "de", "jp", "es", "hk", "fr"]

    all_c = []
    for m in markets:
        all_c.extend(scan_market(m))
    all_c.sort(key=lambda x: x["score"], reverse=True)
    return all_c


# ═══════════════════════════════════════════════════════════════════════════════
#  OUTPUT
# ═══════════════════════════════════════════════════════════════════════════════

def format_markdown(candidates: list) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    if not candidates:
        return (
            f"## 📊 Intraday Scanner — {now}\n\n"
            f"**Market:** {MARKET_CONTEXT.upper()}  |  **Target:** >{MIN_MOVE_PCT}%\n\n"
            "**No candidates met the filter criteria.**\n\n"
            "> ⚠️ Research only. Not financial advice.\n"
        )

    lines = [
        f"## 📊 Intraday Scanner — {now}",
        f"**Market:** {MARKET_CONTEXT.upper()}  |  **Target:** >{MIN_MOVE_PCT}%  |  **Candidates:** {len(candidates)}",
        "",
        "> ⚠️ Research only. Not financial advice. Past ATR ≠ future moves.",
        "",
        "| # | Ticker | Mkt | Price | Day% | Range% | ATR% | RVOL | RSI | Score | Net |",
        "|---|--------|-----|-------|------|--------|------|------|-----|-------|-----|",
    ]
    for i, c in enumerate(candidates[:25], 1):
        e = "🟢" if c["day_return"] > 0 else "🔴"
        n = "✅" if c["net_yield"] > 0 else "❌"
        lines.append(
            f"| {i} | **{c['ticker']}** | {c['market']} | {c['price']:.2f} | "
            f"{e}{c['day_return']:+.1f}% | {c['day_range']:.1f}% | {c['atr_pct']:.1f}% | "
            f"{c['rvol']:.1f}x | {c['rsi']:.0f} | **{c['score']:.0f}** | {n}{c['net_yield']:+.1f}% |"
        )
    return "\n".join(lines)


def format_telegram(candidates: list) -> str:
    now  = datetime.now(timezone.utc).strftime("%H:%M UTC")
    top  = candidates[:10]
    lines = [
        f"📊 <b>Intraday Scanner — {now}</b>",
        f"Market: {MARKET_CONTEXT.upper()} | Target: >{MIN_MOVE_PCT}% | Found: {len(candidates)}",
        "",
    ]
    for i, c in enumerate(top, 1):
        e = "🟢" if c["day_return"] > 0 else "🔴"
        lines.append(
            f"{i}. <b>{c['ticker']}</b> {e}{c['day_return']:+.1f}% | "
            f"ATR {c['atr_pct']:.1f}% | RVOL {c['rvol']:.1f}x | Score <b>{c['score']:.0f}</b>"
        )
    lines += ["", "⚠️ Research only. Not financial advice."]
    return "\n".join(lines)


def send_telegram(text: str) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
        r.raise_for_status()
        return True
    except Exception as e:
        logger.error(f"Telegram error: {e}")
        return False


def send_telegram_doc(filepath: str, caption: str = "") -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    try:
        with open(filepath, "rb") as f:
            r = requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendDocument",
                data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption},
                files={"document": f},
                timeout=30,
            )
        r.raise_for_status()
        return True
    except Exception as e:
        logger.error(f"Telegram doc error: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    logger.info("=" * 60)
    logger.info("INTRADAY MOMENTUM SCANNER v3")
    logger.info(f"Market:  {MARKET_CONTEXT.upper()}")
    logger.info(f"Target:  >{MIN_MOVE_PCT}%")
    logger.info(f"Mode:    {'DRY RUN' if DRY_RUN else 'LIVE'}")
    logger.info("=" * 60)

    candidates = run_all_markets()
    report     = format_markdown(candidates)
    timestamp  = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    filename   = f"scan_{MARKET_CONTEXT}_{timestamp}.md"

    with open(filename, "w") as f:
        f.write(report)
    logger.info(f"Report saved: {filename}")
    print("\n" + report)

    if not DRY_RUN:
        if candidates:
            send_telegram(format_telegram(candidates))
            send_telegram_doc(filename, f"Scan — {MARKET_CONTEXT.upper()} {timestamp}")
        else:
            send_telegram(
                f"📊 <b>Scanner — {datetime.now(timezone.utc).strftime('%H:%M UTC')}</b>\n"
                f"Market: {MARKET_CONTEXT.upper()} | Target: >{MIN_MOVE_PCT}%\n"
                "No candidates this scan."
            )

    logger.info(f"Complete — {len(candidates)} candidates found")


if __name__ == "__main__":
    main()
