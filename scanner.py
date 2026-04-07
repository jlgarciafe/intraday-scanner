"""
═══════════════════════════════════════════════════════════════════════════════
  INTRADAY MOMENTUM SCANNER v2
  Target: >N% daily move candidates across US, UK, DE, JP markets
  Method: Relative volume + ATR momentum + price structure filters

  ⚠️  RISK DISCLAIMER:
  Consistently achieving >5% net daily returns is statistically rare and
  highly speculative. Studies show >95% of intraday traders lose money over
  any sustained period. This scanner identifies CANDIDATES with momentum
  characteristics — it does not predict outcomes. Use exclusively for
  research. Never risk capital you cannot afford to lose entirely.
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import logging
import time
import random
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
}

# ── Ticker universe ────────────────────────────────────────────────────────────
TICKER_UNIVERSE = {
    "us": [
        "NVDA","AMD","TSLA","META","AMZN","GOOGL","MSFT","AAPL","NFLX","CRM",
        "PLTR","SNOW","RBLX","COIN","MSTR","SMCI","ARM","AVGO","MU","INTC",
        "MRNA","BNTX","REGN","BIIB","VRTX",
        "TQQQ","SQQQ","SPXL","UPRO","LABU","SOXL","TNA","FAS",
        "CRWD","DDOG","NET","ZS","PANW","OKTA","GTLB","S",
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
}


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA LAYER — with verbose diagnostics
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_snapshot(ticker: str) -> Optional[pd.DataFrame]:
    """
    Fetch 30 days of daily OHLCV. Falls back to 5d if 30d fails.
    Logs exactly what happened for every ticker.
    """
    for period in ["30d", "5d", "1mo"]:
        for attempt in range(2):
            try:
                df = yf.download(
                    ticker,
                    period=period,
                    interval="1d",
                    auto_adjust=True,
                    progress=False,
                    timeout=15,
                )
                if df is not None and len(df) >= 3:
                    logger.info(f"    {ticker}: {len(df)} days data fetched (period={period})")
                    return df
                else:
                    rows = len(df) if df is not None else 0
                    logger.info(f"    {ticker}: only {rows} rows returned (period={period}, attempt={attempt+1})")
            except Exception as e:
                logger.info(f"    {ticker}: fetch error — {type(e).__name__}: {str(e)[:80]}")
            time.sleep(0.5)
    logger.info(f"    {ticker}: ❌ all fetch attempts failed — skipping")
    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  MATHEMATICAL FILTERS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_atr(df: pd.DataFrame, period: int = 10) -> float:
    """ATR% using last N days (reduced to 10 for shorter data sets)."""
    if len(df) < 3:
        return 0.0
    try:
        period = min(period, len(df) - 1)
        high  = df["High"].values
        low   = df["Low"].values
        close = df["Close"].values
        tr_list = []
        for i in range(1, len(close)):
            tr = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i]  - close[i-1]),
            )
            tr_list.append(tr)
        atr = sum(tr_list[-period:]) / period
        latest_close = float(close[-1])
        return (atr / latest_close) * 100 if latest_close > 0 else 0.0
    except Exception as e:
        logger.debug(f"ATR error: {e}")
        return 0.0


def compute_rvol(df: pd.DataFrame) -> float:
    """RVOL vs available history (min 3 days)."""
    if len(df) < 3:
        return 1.0
    try:
        vols = df["Volume"].values
        lookback = min(20, len(vols) - 1)
        avg_vol   = sum(vols[-lookback-1:-1]) / lookback
        today_vol = float(vols[-1])
        rvol = today_vol / avg_vol if avg_vol > 0 else 1.0
        return round(rvol, 2)
    except Exception:
        return 1.0


def compute_rsi(closes, period: int = 10) -> float:
    """RSI with reduced period for shorter data."""
    period = min(period, len(closes) - 2)
    if period < 2:
        return 50.0
    gains, losses = [], []
    for i in range(1, len(closes)):
        delta = float(closes[i]) - float(closes[i-1])
        gains.append(max(delta, 0))
        losses.append(max(-delta, 0))
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def analyse_ticker(ticker: str, df: pd.DataFrame) -> dict:
    """
    Full analysis for a single ticker. Returns dict with all metrics.
    Logs verbose per-ticker breakdown so we can see exactly what's happening.
    """
    try:
        close  = df["Close"].values
        open_  = df["Open"].values
        high   = df["High"].values
        low    = df["Low"].values

        atr_pct      = compute_atr(df)
        rvol         = compute_rvol(df)
        rsi          = compute_rsi(close)

        latest_close  = float(close[-1])
        prev_close    = float(close[-2]) if len(close) >= 2 else latest_close
        today_open    = float(open_[-1])

        gap_pct       = ((today_open - prev_close) / prev_close * 100) if prev_close > 0 else 0.0
        day_return    = ((latest_close - prev_close) / prev_close * 100) if prev_close > 0 else 0.0
        day_range     = ((float(high[-1]) - float(low[-1])) / float(low[-1]) * 100) if float(low[-1]) > 0 else 0.0
        latest_volume = int(df["Volume"].values[-1])

        # Composite score
        atr_score  = min(atr_pct / 5.0, 1.0) * 30
        rvol_score = min(rvol / 3.0, 1.0) * 25
        mom_score  = min(abs(day_return) / 3.0, 1.0) * 25
        rsi_score  = (1.0 if 60 <= rsi <= 80 else 0.7 if 50 <= rsi < 60 else 0.5) * 20
        score      = atr_score + rvol_score + mom_score + rsi_score

        result = {
            "ticker":      ticker,
            "price":       round(latest_close, 4),
            "atr_pct":     round(atr_pct, 2),
            "rvol":        round(rvol, 2),
            "rsi":         round(rsi, 1),
            "gap_pct":     round(gap_pct, 2),
            "day_return":  round(day_return, 2),
            "day_range":   round(day_range, 2),
            "volume":      latest_volume,
            "score":       round(score, 1),
            "data_days":   len(df),
        }

        # Verbose per-ticker log — shows exactly what values were computed
        logger.info(
            f"    {ticker}: price={latest_close:.2f} | "
            f"ATR={atr_pct:.2f}% | RVOL={rvol:.2f}x | RSI={rsi:.0f} | "
            f"day={day_return:+.2f}% | range={day_range:.2f}% | "
            f"vol={latest_volume:,} | score={score:.0f}"
        )
        return result

    except Exception as e:
        logger.info(f"    {ticker}: analysis error — {e}")
        return {}


def passes_filters(metrics: dict, market: str) -> tuple:
    """
    Apply filters and return (passed: bool, reason: str).
    When MIN_MOVE_PCT=0, only apply score > 20 as minimum sanity check.
    """
    atr   = metrics.get("atr_pct", 0)
    rvol  = metrics.get("rvol", 0)
    score = metrics.get("score", 0)
    vol   = metrics.get("volume", 0)

    # Minimum volume — UK stocks trade in pence, volume thresholds differ
    min_vol = 50_000 if market == "uk" else 100_000
    if vol < min_vol:
        return False, f"volume {vol:,} < {min_vol:,}"

    # When target = 0, just need minimal score
    if MIN_MOVE_PCT == 0:
        if score < 20:
            return False, f"score {score:.0f} < 20"
        return True, "passed (no move threshold)"

    # ATR must support the target move (at 70% of target)
    if atr < MIN_MOVE_PCT * 0.7:
        return False, f"ATR {atr:.2f}% < {MIN_MOVE_PCT*0.7:.2f}% needed"

    # RVOL minimum
    if rvol < 1.3:
        return False, f"RVOL {rvol:.2f}x < 1.3x"

    # Score
    if score < 35:
        return False, f"score {score:.0f} < 35"

    return True, "passed"


# ═══════════════════════════════════════════════════════════════════════════════
#  FEE MODEL
# ═══════════════════════════════════════════════════════════════════════════════

def compute_net_yield(gross_pct: float, market: str) -> dict:
    fees = FEE_MODEL.get(market, FEE_MODEL["us"])
    cost = (fees["commission_pct"] + fees["spread_pct"] + fees["slippage_pct"]) * 2
    return {
        "gross_pct":    round(gross_pct, 2),
        "cost_pct":     round(cost, 2),
        "net_pct":      round(gross_pct - cost, 2),
        "breakeven":    round(cost, 2),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN SCANNER
# ═══════════════════════════════════════════════════════════════════════════════

def scan_market(market_key: str) -> list:
    tickers = TICKER_UNIVERSE.get(market_key, [])
    if not tickers:
        logger.warning(f"No tickers defined for market: {market_key}")
        return []

    logger.info(f"\nScanning {market_key.upper()}: {len(tickers)} tickers")
    logger.info(f"Min move threshold: {MIN_MOVE_PCT}%")
    candidates = []
    skipped_data = 0
    skipped_filter = 0

    for i, ticker in enumerate(tickers, 1):
        logger.info(f"  [{i}/{len(tickers)}] {ticker}")

        df = fetch_snapshot(ticker)
        if df is None or len(df) < 3:
            skipped_data += 1
            continue

        metrics = analyse_ticker(ticker, df)
        if not metrics:
            skipped_data += 1
            continue

        passed, reason = passes_filters(metrics, market_key)
        if not passed:
            logger.info(f"    {ticker}: ❌ filtered — {reason}")
            skipped_filter += 1
            continue

        logger.info(f"    {ticker}: ✅ CANDIDATE — {reason}")

        net = compute_net_yield(metrics["atr_pct"], market_key)
        candidates.append({
            **metrics,
            "market":    market_key.upper(),
            "net_yield": net["net_pct"],
            "breakeven": net["breakeven"],
        })

        time.sleep(0.2)

    logger.info(f"\n{market_key.upper()} Summary:")
    logger.info(f"  Total scanned:      {len(tickers)}")
    logger.info(f"  Data failures:      {skipped_data}")
    logger.info(f"  Filtered out:       {skipped_filter}")
    logger.info(f"  Candidates:         {len(candidates)}")

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates


def run_all_markets() -> list:
    if MARKET_CONTEXT in ("all",):
        markets = ["us", "uk", "de", "jp"]
    elif MARKET_CONTEXT == "eu":
        markets = ["uk", "de"]
    elif MARKET_CONTEXT in TICKER_UNIVERSE:
        markets = [MARKET_CONTEXT]
    else:
        logger.warning(f"Unknown market: {MARKET_CONTEXT} — defaulting to all")
        markets = ["us", "uk", "de", "jp"]

    all_candidates = []
    for market in markets:
        all_candidates.extend(scan_market(market))
    all_candidates.sort(key=lambda x: x["score"], reverse=True)
    return all_candidates


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
        "> ⚠️ **Risk Disclaimer:** Consistently achieving >5% net daily returns is statistically rare",
        "> and highly speculative. This is for research only. Not financial advice.",
        "",
        "| # | Ticker | Mkt | Price | Day% | Range% | ATR% | RVOL | RSI | Score | Net |",
        "|---|--------|-----|-------|------|--------|------|------|-----|-------|-----|",
    ]

    for i, c in enumerate(candidates[:25], 1):
        day_e = "🟢" if c["day_return"] > 0 else "🔴"
        net_e = "✅" if c["net_yield"] > 0 else "❌"
        lines.append(
            f"| {i} | **{c['ticker']}** | {c['market']} | "
            f"{c['price']:.2f} | {day_e}{c['day_return']:+.1f}% | "
            f"{c['day_range']:.1f}% | {c['atr_pct']:.1f}% | "
            f"{c['rvol']:.1f}x | {c['rsi']:.0f} | **{c['score']:.0f}** | "
            f"{net_e}{c['net_yield']:+.1f}% |"
        )

    return "\n".join(lines)


def format_telegram(candidates: list) -> str:
    now = datetime.now(timezone.utc).strftime("%H:%M UTC")
    top = candidates[:10]
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
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    logger.info("=" * 60)
    logger.info("INTRADAY MOMENTUM SCANNER v2 — starting")
    logger.info(f"Market:  {MARKET_CONTEXT.upper()}")
    logger.info(f"Target:  >{MIN_MOVE_PCT}%")
    logger.info(f"Mode:    {'DRY RUN' if DRY_RUN else 'LIVE'}")
    logger.info("=" * 60)

    candidates = run_all_markets()

    report    = format_markdown(candidates)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    filename  = f"scan_{MARKET_CONTEXT}_{timestamp}.md"

    with open(filename, "w") as f:
        f.write(report)
    logger.info(f"\nReport saved: {filename}")
    print("\n" + report)

    if not DRY_RUN:
        if candidates:
            send_telegram(format_telegram(candidates))
            send_telegram_doc(filename, f"Scan report — {MARKET_CONTEXT.upper()}")
            logger.info("Telegram sent")
        else:
            send_telegram(
                f"📊 <b>Intraday Scanner — {datetime.now(timezone.utc).strftime('%H:%M UTC')}</b>\n"
                f"Market: {MARKET_CONTEXT.upper()} | Target: >{MIN_MOVE_PCT}%\n\n"
                "No candidates found this scan."
            )
    else:
        logger.info("DRY RUN — Telegram suppressed")

    logger.info("=" * 60)
    logger.info(f"Complete — {len(candidates)} candidates")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
