"""
================================================================================
  INTRADAY MOMENTUM SCANNER v3
  Target: >N% daily move candidates across US, UK, DE, JP, ES, HK, FR markets
  Method: Relative volume + ATR momentum + price structure filters

  RISK DISCLAIMER:
  Consistently achieving >5% net daily returns is statistically rare and
  highly speculative. Studies show >95% of intraday traders lose money over
  any sustained period. This scanner identifies CANDIDATES with momentum
  characteristics -- it does not predict outcomes. Use exclusively for
  research. Never risk capital you cannot afford to lose entirely.
================================================================================
"""

import os
import sys
import json
import logging
import time
import requests
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime, timezone
from typing import Optional

import yfinance as yf

# ── Logging ───────────────────────────────────────────────────────────────────
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
MIN_MOVE_PCT       = float(os.getenv("MIN_MOVE_PCT", "3.0"))
DRY_RUN            = os.getenv("DRY_RUN", "false").lower() == "true"
OUTPUT_JSON        = os.getenv("OUTPUT_JSON", "scan_results.json")

# ── Fee model ─────────────────────────────────────────────────────────────────
FEE_MODEL = {
    "us": {"commission_pct": 0.005, "spread_pct": 0.05, "slippage_pct": 0.05},
    "uk": {"commission_pct": 0.100, "spread_pct": 0.10, "slippage_pct": 0.10},
    "de": {"commission_pct": 0.100, "spread_pct": 0.10, "slippage_pct": 0.10},
    "jp": {"commission_pct": 0.050, "spread_pct": 0.08, "slippage_pct": 0.08},
    "es": {"commission_pct": 0.100, "spread_pct": 0.10, "slippage_pct": 0.10},
    "hk": {"commission_pct": 0.080, "spread_pct": 0.10, "slippage_pct": 0.10},
    "fr": {"commission_pct": 0.100, "spread_pct": 0.10, "slippage_pct": 0.10},
}

# ── Ticker universe ───────────────────────────────────────────────────────────
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
    "es": [
        "SAN.MC","BBVA.MC","ITX.MC","TEF.MC","IBE.MC","REP.MC","AMS.MC",
        "CABK.MC","BKT.MC","MAP.MC","SAB.MC","ANA.MC","FER.MC","ENG.MC",
        "RED.MC","ACX.MC","CIE.MC","ELE.MC","GRF.MC","MTS.MC",
    ],
    "hk": [
        "0700.HK","0939.HK","1299.HK","0005.HK","3690.HK","2318.HK",
        "0941.HK","1810.HK","0175.HK","2020.HK","0388.HK","1211.HK",
        "0883.HK","0016.HK","2382.HK","0011.HK","0012.HK","0267.HK",
        "0002.HK","1177.HK",
    ],
    "fr": [
        "MC.PA","OR.PA","TTE.PA","SAN.PA","AIR.PA","BNP.PA","ACA.PA",
        "CS.PA","GLE.PA","VIE.PA","EL.PA","DG.PA","RI.PA","SGO.PA",
        "VIV.PA","STM.PA","CAP.PA","ML.PA","URW.PA","ORA.PA",
    ],
}


# =============================================================================
#  DATA LAYER
# =============================================================================

def fetch_snapshot(ticker: str) -> Optional[pd.DataFrame]:
    """
    Fetch 30 days of daily OHLCV. Falls back to 5d if 30d fails.
    Flattens MultiIndex columns from yfinance.
    """
    for period in ["30d", "5d", "1mo"]:
        try:
            df = yf.download(ticker, period=period, interval="1d",
                             progress=False, auto_adjust=True)
            if df is None or df.empty:
                continue

            # Flatten MultiIndex columns (yfinance v0.2+)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Drop rows with no close price
            df = df.dropna(subset=["Close"])

            if len(df) >= 3:
                logger.info(f"    {ticker}: fetched {len(df)} bars (period={period})")
                return df

        except Exception as e:
            logger.warning(f"    {ticker}: fetch error ({period}) — {e}")
            time.sleep(0.5)

    logger.warning(f"    {ticker}: all fetch attempts failed")
    return None


# =============================================================================
#  ANALYSIS
# =============================================================================

def analyse_ticker(ticker: str, df: pd.DataFrame) -> Optional[dict]:
    """
    Compute ATR%, RVOL, day return and score from OHLCV dataframe.
    Returns None if data is insufficient or malformed.
    """
    try:
        close  = df["Close"].squeeze()
        high   = df["High"].squeeze()
        low    = df["Low"].squeeze()
        volume = df["Volume"].squeeze()

        if len(close) < 3:
            return None

        # Current price and day return
        price     = float(close.iloc[-1])
        prev_close = float(close.iloc[-2])
        if prev_close == 0:
            return None

        day_return = (price - prev_close) / prev_close * 100

        # ATR (10-day average true range as % of price)
        period = min(10, len(df) - 1)
        tr_list = []
        for i in range(-period, 0):
            h = float(high.iloc[i])
            l = float(low.iloc[i])
            c_prev = float(close.iloc[i - 1])
            tr_list.append(max(h - l, abs(h - c_prev), abs(l - c_prev)))

        atr = sum(tr_list) / len(tr_list) if tr_list else 0
        atr_pct = (atr / price * 100) if price > 0 else 0

        # Relative volume
        vol_today = float(volume.iloc[-1])
        vol_avg   = float(volume.iloc[-6:-1].mean()) if len(volume) >= 6 else float(volume.mean())
        rvol      = (vol_today / vol_avg) if vol_avg > 0 else 0

        # Score: weight ATR and RVOL
        score = (atr_pct * 10) + (min(rvol, 5) * 10) + min(abs(day_return), 10)

        logger.info(
            f"    {ticker}: price={price:.2f} | day={day_return:+.1f}% | "
            f"ATR={atr_pct:.1f}% | RVOL={rvol:.1f}x | score={score:.0f}"
        )

        return {
            "ticker":     ticker,
            "price":      round(price, 4),
            "prev_close": round(prev_close, 4),
            "day_return": round(day_return, 2),
            "atr_pct":    round(atr_pct, 2),
            "rvol":       round(rvol, 2),
            "score":      round(score, 1),
            "vol_today":  int(vol_today),
        }

    except Exception as e:
        logger.warning(f"    {ticker}: analysis error — {e}")
        return None


def passes_filters(metrics: dict, market_key: str) -> tuple:
    """
    Returns (passed: bool, reason: str).
    Filters: ATR%, RVOL, minimum move, minimum score.
    """
    atr_pct    = metrics["atr_pct"]
    rvol       = metrics["rvol"]
    day_return = abs(metrics["day_return"])
    score      = metrics["score"]

    # Market-specific thresholds
    min_atr = {"us": 2.0, "uk": 1.0, "de": 1.0, "jp": 1.0,
               "es": 1.0, "hk": 1.5, "fr": 1.0}.get(market_key, 1.5)
    min_rvol = 1.3

    if atr_pct < min_atr:
        return False, f"ATR {atr_pct:.1f}% < {min_atr}% min"
    if rvol < min_rvol:
        return False, f"RVOL {rvol:.1f}x < {min_rvol}x min"
    if MIN_MOVE_PCT > 0 and day_return < MIN_MOVE_PCT:
        return False, f"move {day_return:.1f}% < {MIN_MOVE_PCT}% target"
    if score < 20:
        return False, f"score {score:.0f} < 20 min"

    return True, f"ATR={atr_pct:.1f}% RVOL={rvol:.1f}x move={day_return:.1f}%"


def compute_net_yield(atr_pct: float, market_key: str) -> dict:
    fees  = FEE_MODEL.get(market_key, FEE_MODEL["us"])
    total_cost = fees["commission_pct"] + fees["spread_pct"] + fees["slippage_pct"]
    net   = atr_pct - total_cost
    return {"net_pct": round(net, 2), "breakeven": round(total_cost, 2)}


# =============================================================================
#  MARKET RUNNER
# =============================================================================

def scan_market(market_key: str) -> list:
    tickers = TICKER_UNIVERSE.get(market_key, [])
    if not tickers:
        logger.warning(f"No tickers defined for market: {market_key}")
        return []

    logger.info(f"\nScanning {market_key.upper()}: {len(tickers)} tickers")
    candidates    = []
    skipped_data  = 0
    skipped_filter = 0

    for i, ticker in enumerate(tickers, 1):
        logger.info(f"  [{i}/{len(tickers)}] {ticker}")

        df = fetch_snapshot(ticker)
        if df is None or len(df) < 3:
            logger.warning(f"    {ticker}: skipped — no data")
            skipped_data += 1
            continue

        metrics = analyse_ticker(ticker, df)
        if not metrics:
            skipped_data += 1
            continue

        passed, reason = passes_filters(metrics, market_key)
        if not passed:
            logger.info(f"    {ticker}: FILTERED -- {reason}")
            skipped_filter += 1
            continue

        logger.info(f"    {ticker}: CANDIDATE -- {reason}")

        net = compute_net_yield(metrics["atr_pct"], market_key)
        candidates.append({
            **metrics,
            "market":    market_key.upper(),
            "net_yield": net["net_pct"],
            "breakeven": net["breakeven"],
            "scan_time_utc": datetime.now(timezone.utc).isoformat(),
        })

        time.sleep(0.2)

    logger.info(f"\n{market_key.upper()} Summary:")
    logger.info(f"  Total scanned:  {len(tickers)}")
    logger.info(f"  Data failures:  {skipped_data}")
    logger.info(f"  Filtered out:   {skipped_filter}")
    logger.info(f"  Candidates:     {len(candidates)}")

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates


def run_all_markets() -> list:
    market_map = {
        "all":  ["us", "uk", "de", "jp", "es", "hk", "fr"],
        "eu":   ["uk", "de", "es", "fr"],
        "asia": ["jp", "hk"],
    }

    if MARKET_CONTEXT in market_map:
        markets = market_map[MARKET_CONTEXT]
    elif MARKET_CONTEXT in TICKER_UNIVERSE:
        markets = [MARKET_CONTEXT]
    else:
        logger.warning(f"Unknown market '{MARKET_CONTEXT}' — defaulting to all")
        markets = ["us", "uk", "de", "jp", "es", "hk", "fr"]

    all_candidates = []
    for market in markets:
        all_candidates.extend(scan_market(market))

    all_candidates.sort(key=lambda x: x["score"], reverse=True)
    return all_candidates


# =============================================================================
#  OUTPUT FORMATTERS
# =============================================================================

def format_markdown(candidates: list) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    if not candidates:
        return (
            f"## Intraday Scanner -- {now}\n\n"
            f"**Market:** {MARKET_CONTEXT.upper()}  |  **Target:** >{MIN_MOVE_PCT}%\n\n"
            "**No candidates met the filter criteria.**\n\n"
            "> Research only. Not financial advice."
        )

    lines = [
        f"## Intraday Scanner -- {now}",
        f"**Market:** {MARKET_CONTEXT.upper()}  |  **Target:** >{MIN_MOVE_PCT}%  |  "
        f"**Found:** {len(candidates)}",
        "",
        "| # | Ticker | Market | Price | Day % | ATR % | RVOL | Score | Net Yield |",
        "|---|--------|--------|-------|-------|-------|------|-------|-----------|",
    ]
    for i, c in enumerate(candidates, 1):
        arrow = "+" if c["day_return"] >= 0 else ""
        lines.append(
            f"| {i} | **{c['ticker']}** | {c['market']} | {c['price']:.2f} | "
            f"{arrow}{c['day_return']:.1f}% | {c['atr_pct']:.1f}% | "
            f"{c['rvol']:.1f}x | {c['score']:.0f} | {c['net_yield']:.1f}% |"
        )
    lines += ["", "> Research only. Not financial advice."]
    return "\n".join(lines)


def format_telegram(candidates: list) -> str:
    now = datetime.now(timezone.utc).strftime("%H:%M UTC")
    lines = [
        f"<b>Intraday Scanner -- {now}</b>",
        f"Market: {MARKET_CONTEXT.upper()} | Target: >{MIN_MOVE_PCT}% | Found: {len(candidates)}",
        "",
    ]
    for c in candidates[:10]:
        e = "+" if c["day_return"] >= 0 else ""
        lines.append(
            f"<b>{c['ticker']}</b> {e}{c['day_return']:+.1f}% | "
            f"ATR {c['atr_pct']:.1f}% | RVOL {c['rvol']:.1f}x | Score <b>{c['score']:.0f}</b>"
        )
    lines += ["", "Research only. Not financial advice."]
    return "\n".join(lines)


def send_telegram(text: str) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram credentials not set -- skipping")
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


# =============================================================================
#  MAIN
# =============================================================================

def main():
    logger.info("=" * 60)
    logger.info("INTRADAY MOMENTUM SCANNER v3")
    logger.info(f"Market:  {MARKET_CONTEXT.upper()}")
    logger.info(f"Target:  >{MIN_MOVE_PCT}%")
    logger.info(f"Mode:    {'DRY RUN' if DRY_RUN else 'LIVE'}")
    logger.info("=" * 60)

    candidates = run_all_markets()

    # ── Write scan_results.json for ta_runner.py ──────────────────────────────
    with open(OUTPUT_JSON, "w") as f:
        json.dump(candidates, f, indent=2)
    logger.info(f"scan_results.json written -- {len(candidates)} candidate(s)")

    # ── Write markdown report ─────────────────────────────────────────────────
    report    = format_markdown(candidates)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    filename  = f"scan_{MARKET_CONTEXT}_{timestamp}.md"

    with open(filename, "w") as f:
        f.write(report)
    logger.info(f"Report saved: {filename}")
    print("\n" + report)

    # ── Telegram ──────────────────────────────────────────────────────────────
    if not DRY_RUN:
        if candidates:
            send_telegram(format_telegram(candidates))
            send_telegram_doc(filename, f"Scan -- {MARKET_CONTEXT.upper()} {timestamp}")
            logger.info("Telegram sent")
        else:
            send_telegram(
                f"<b>Scanner -- {datetime.now(timezone.utc).strftime('%H:%M UTC')}</b>\n"
                f"Market: {MARKET_CONTEXT.upper()} | Target: >{MIN_MOVE_PCT}%\n\n"
                "No candidates found this scan."
            )
    else:
        logger.info("DRY RUN -- Telegram suppressed")

    logger.info("=" * 60)
    logger.info(f"Complete -- {len(candidates)} candidates")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
