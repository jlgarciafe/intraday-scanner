"""
═══════════════════════════════════════════════════════════════════════════════
  INTRADAY MOMENTUM SCANNER
  Target: >5% daily move candidates across US, UK, DE, JP markets
  Method: Relative volume + ATR momentum + price structure filters
  Output: Ranked Markdown table + Telegram alert

  ⚠️  RISK DISCLAIMER (embedded per design requirement):
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
import math
from datetime import datetime, timezone
from typing import Optional
import warnings
warnings.filterwarnings("ignore")

import requests
import pandas as pd
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
MARKET_CONTEXT     = os.getenv("MARKET_CONTEXT", "all")
MIN_MOVE_PCT       = float(os.getenv("MIN_MOVE_PCT", "5.0"))
DRY_RUN            = os.getenv("DRY_RUN", "false").lower() == "true"

# ── Fee model ─────────────────────────────────────────────────────────────────
# Realistic round-trip cost assumptions per market
FEE_MODEL = {
    "us": {"commission_pct": 0.005, "spread_pct": 0.05,  "slippage_pct": 0.05},
    "uk": {"commission_pct": 0.100, "spread_pct": 0.10,  "slippage_pct": 0.10},
    "de": {"commission_pct": 0.100, "spread_pct": 0.10,  "slippage_pct": 0.10},
    "jp": {"commission_pct": 0.050, "spread_pct": 0.08,  "slippage_pct": 0.08},
}

# ── Ticker universe ────────────────────────────────────────────────────────────
# Curated high-liquidity, high-beta universe per market
# These are screened for sufficient volume to trade meaningfully
TICKER_UNIVERSE = {
    "us": [
        # Mega-cap tech (high beta / options active)
        "NVDA","AMD","TSLA","META","AMZN","GOOGL","MSFT","AAPL","NFLX","CRM",
        # High-beta growth / AI
        "PLTR","SNOW","RBLX","COIN","MSTR","SMCI","ARM","AVGO","MU","INTC",
        # Biotech / high volatility
        "MRNA","BNTX","SGEN","REGN","BIIB","VRTX","ILMN",
        # Leveraged ETFs (very high vol — explicitly speculative)
        "TQQQ","SQQQ","SPXL","UPRO","LABU","SOXL","TNA","FAS",
        # High-momentum mid-caps
        "CRWD","DDOG","NET","ZS","PANW","OKTA","GTLB","S","CYBR",
        # Energy / commodities
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
#  DATA LAYER
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_snapshot(ticker: str, period: str = "5d", interval: str = "1d") -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV data from Yahoo Finance with retry logic.
    Returns DataFrame or None on failure.
    """
    for attempt in range(3):
        try:
            df = yf.download(
                ticker, period=period, interval=interval,
                auto_adjust=True, progress=False, timeout=10,
            )
            if df is not None and len(df) >= 2:
                return df
        except Exception as e:
            logger.debug(f"{ticker} attempt {attempt+1}: {e}")
            time.sleep(1.5 * (attempt + 1))
    return None


def fetch_intraday(ticker: str) -> Optional[pd.DataFrame]:
    """
    Fetch intraday 5-minute bars for today.
    Used to compute live session momentum.
    """
    for attempt in range(3):
        try:
            df = yf.download(
                ticker, period="1d", interval="5m",
                auto_adjust=True, progress=False, timeout=10,
            )
            if df is not None and len(df) >= 5:
                return df
        except Exception as e:
            logger.debug(f"{ticker} intraday attempt {attempt+1}: {e}")
            time.sleep(1.0)
    return None


def get_ticker_info(ticker: str) -> dict:
    """Fetch metadata: market cap, sector, shortName."""
    try:
        t = yf.Ticker(ticker)
        info = t.info
        return {
            "name":       info.get("shortName", ticker),
            "sector":     info.get("sector", "—"),
            "market_cap": info.get("marketCap", 0),
            "currency":   info.get("currency", "USD"),
        }
    except Exception:
        return {"name": ticker, "sector": "—", "market_cap": 0, "currency": "USD"}


# ═══════════════════════════════════════════════════════════════════════════════
#  MATHEMATICAL FILTERS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    """
    Average True Range (ATR) — measures volatility as average of:
      TR = max(High-Low, |High-PrevClose|, |Low-PrevClose|)
    Returns ATR as percentage of latest close price.
    """
    if len(df) < period + 1:
        return 0.0
    try:
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
    except Exception:
        return 0.0


def compute_relative_volume(df: pd.DataFrame, lookback: int = 20) -> float:
    """
    Relative Volume (RVOL) = Today's volume / Average volume over lookback days.
    RVOL > 2.0 = meaningfully elevated interest.
    RVOL > 5.0 = high-conviction momentum signal.
    """
    if len(df) < lookback + 1:
        return 1.0
    try:
        vols = df["Volume"].values
        avg_vol  = sum(vols[-lookback-1:-1]) / lookback
        today_vol = float(vols[-1])
        return today_vol / avg_vol if avg_vol > 0 else 1.0
    except Exception:
        return 1.0


def compute_momentum_score(df: pd.DataFrame) -> dict:
    """
    Composite momentum score combining:
      1. ATR%         — historical daily range capability
      2. RVOL         — relative volume (conviction signal)
      3. Gap%         — today's open vs yesterday's close
      4. Session%     — today's intraday move so far
      5. RSI (14)     — momentum direction filter

    Score ranges 0–100. Candidates above threshold are surfaced.
    """
    if len(df) < 15:
        return {}

    try:
        close  = df["Close"].values
        open_  = df["Open"].values
        high   = df["High"].values
        low    = df["Low"].values

        # ATR% (14-day)
        atr_pct = compute_atr(df, 14)

        # RVOL (20-day)
        rvol = compute_relative_volume(df, 20)

        # Gap % = (today open - yesterday close) / yesterday close * 100
        gap_pct = ((open_[-1] - close[-2]) / close[-2]) * 100 if close[-2] > 0 else 0.0

        # Session % = (today close - today open) / today open * 100
        session_pct = ((close[-1] - open_[-1]) / open_[-1]) * 100 if open_[-1] > 0 else 0.0

        # Total day range % = (today high - today low) / today low * 100
        day_range_pct = ((high[-1] - low[-1]) / low[-1]) * 100 if low[-1] > 0 else 0.0

        # 1-day return %
        day_return_pct = ((close[-1] - close[-2]) / close[-2]) * 100 if close[-2] > 0 else 0.0

        # RSI (14)
        rsi = compute_rsi(close, 14)

        # 5-day momentum (price change over 5 days)
        momentum_5d = ((close[-1] - close[-6]) / close[-6]) * 100 if len(close) >= 6 and close[-6] > 0 else 0.0

        # ── Composite score (weighted) ────────────────────────────────────────
        # ATR >= 3% needed to have capability of >5% move
        atr_score     = min(atr_pct / 5.0, 1.0) * 30        # 30 pts max
        rvol_score    = min(rvol / 5.0, 1.0) * 25            # 25 pts max
        momentum_score = min(abs(day_return_pct) / 5.0, 1.0) * 25  # 25 pts max
        rsi_score     = _rsi_score(rsi) * 20                 # 20 pts max

        total_score = atr_score + rvol_score + momentum_score + rsi_score

        return {
            "atr_pct":       round(atr_pct, 2),
            "rvol":          round(rvol, 2),
            "gap_pct":       round(gap_pct, 2),
            "session_pct":   round(session_pct, 2),
            "day_range_pct": round(day_range_pct, 2),
            "day_return_pct":round(day_return_pct, 2),
            "momentum_5d":   round(momentum_5d, 2),
            "rsi":           round(rsi, 1),
            "score":         round(total_score, 1),
            "latest_close":  round(float(close[-1]), 4),
            "latest_volume": int(df["Volume"].values[-1]),
        }
    except Exception as e:
        logger.debug(f"Momentum score error: {e}")
        return {}


def compute_rsi(closes, period: int = 14) -> float:
    """RSI (Wilder smoothing method)."""
    if len(closes) < period + 1:
        return 50.0
    gains, losses = [], []
    for i in range(1, len(closes)):
        delta = closes[i] - closes[i-1]
        gains.append(max(delta, 0))
        losses.append(max(-delta, 0))
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _rsi_score(rsi: float) -> float:
    """
    Score RSI: reward extreme readings (momentum continuation zones).
    RSI 60–80 (bullish momentum) or 20–40 (oversold bounce) = high score.
    RSI ~50 = neutral = low score.
    """
    if 65 <= rsi <= 80:  return 1.0   # Strong bullish momentum
    if 55 <= rsi < 65:   return 0.75
    if 80 < rsi <= 90:   return 0.6   # Overbought — caution
    if 20 <= rsi <= 40:  return 0.8   # Oversold bounce potential
    if 40 < rsi < 55:    return 0.3   # Neutral
    return 0.2


# ═══════════════════════════════════════════════════════════════════════════════
#  FEE & NET YIELD MODEL
# ═══════════════════════════════════════════════════════════════════════════════

def compute_net_yield(
    gross_move_pct: float,
    market: str,
    leverage: float = 1.0,
) -> dict:
    """
    Net yield after round-trip costs:
      Gross move × leverage
      − commission (entry + exit)
      − spread cost (entry + exit)
      − slippage (entry + exit)

    ⚠️  This is a simplified model. Real costs vary by broker, order size,
    and market conditions. Leverage amplifies BOTH gains and losses equally.
    A 5% gross move with 2x leverage = 10% gross but also 2x the loss if wrong.
    """
    fees = FEE_MODEL.get(market, FEE_MODEL["us"])
    total_cost_pct = (
        fees["commission_pct"] * 2 +  # entry + exit
        fees["spread_pct"] * 2 +       # entry + exit spread
        fees["slippage_pct"] * 2       # entry + exit slippage
    )
    gross_leveraged = gross_move_pct * leverage
    net = gross_leveraged - total_cost_pct
    breakeven = total_cost_pct / leverage  # gross move needed to break even

    return {
        "gross_pct":    round(gross_move_pct, 2),
        "leverage":     leverage,
        "gross_lev_pct":round(gross_leveraged, 2),
        "total_cost_pct":round(total_cost_pct, 2),
        "net_yield_pct":round(net, 2),
        "breakeven_pct":round(breakeven, 2),
        "is_positive":  net > 0,
    }


def assess_move_probability(
    atr_pct: float,
    rvol: float,
    target_pct: float = 5.0,
) -> str:
    """
    Qualitative probability assessment for achieving target_pct move.
    Based on ATR capability and volume conviction.
    This is NOT a statistical probability — it is a heuristic tier.
    """
    # Capability: can the ATR physically support this move?
    capable = atr_pct >= target_pct * 0.8

    if not capable:
        return "LOW — ATR too small"
    if rvol >= 5.0 and atr_pct >= target_pct:
        return "HIGH — Vol + ATR aligned"
    if rvol >= 3.0 and atr_pct >= target_pct * 0.9:
        return "MEDIUM-HIGH"
    if rvol >= 2.0:
        return "MEDIUM"
    return "LOW-MEDIUM — Volume weak"


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN SCANNER
# ═══════════════════════════════════════════════════════════════════════════════

def scan_market(market_key: str) -> list:
    """
    Scan all tickers in a market, apply momentum filters, return ranked candidates.
    """
    tickers = TICKER_UNIVERSE.get(market_key, [])
    if not tickers:
        return []

    logger.info(f"Scanning {market_key.upper()}: {len(tickers)} tickers")
    candidates = []

    for i, ticker in enumerate(tickers, 1):
        logger.info(f"  [{i}/{len(tickers)}] {ticker}")

        df = fetch_snapshot(ticker, period="30d", interval="1d")
        if df is None or len(df) < 10:
            logger.debug(f"  {ticker}: insufficient data")
            continue

        metrics = compute_momentum_score(df)
        if not metrics:
            continue

        # ── Filter 1: Minimum ATR capability ──────────────────────────────────
        # Stock must have demonstrated ability to move >= MIN_MOVE_PCT * 0.7
        # (we don't require it's moving exactly that much today)
        if metrics["atr_pct"] < MIN_MOVE_PCT * 0.7:
            logger.debug(f"  {ticker}: ATR {metrics['atr_pct']}% below threshold")
            continue

        # ── Filter 2: Relative volume > 1.5 ──────────────────────────────────
        # Elevated volume = institutional/retail interest = follow-through
        if metrics["rvol"] < 1.5:
            logger.debug(f"  {ticker}: RVOL {metrics['rvol']} too low")
            continue

        # ── Filter 3: Minimum momentum score ─────────────────────────────────
        if metrics["score"] < 40:
            logger.debug(f"  {ticker}: score {metrics['score']} too low")
            continue

        # ── Net yield model ───────────────────────────────────────────────────
        # Project net yield assuming the stock achieves its ATR move today
        net = compute_net_yield(metrics["atr_pct"], market_key, leverage=1.0)
        prob = assess_move_probability(metrics["atr_pct"], metrics["rvol"], MIN_MOVE_PCT)

        # ── Volume filter: minimum liquidity ─────────────────────────────────
        if metrics["latest_volume"] < 100_000:
            logger.debug(f"  {ticker}: volume too low for liquid execution")
            continue

        candidates.append({
            "ticker":        ticker,
            "market":        market_key.upper(),
            "price":         metrics["latest_close"],
            "day_return_pct":metrics["day_return_pct"],
            "day_range_pct": metrics["day_range_pct"],
            "atr_pct":       metrics["atr_pct"],
            "rvol":          metrics["rvol"],
            "rsi":           metrics["rsi"],
            "momentum_5d":   metrics["momentum_5d"],
            "score":         metrics["score"],
            "net_yield_1x":  net["net_yield_pct"],
            "breakeven_pct": net["breakeven_pct"],
            "probability":   prob,
            "volume":        metrics["latest_volume"],
        })

        time.sleep(0.3)  # polite rate limit on Yahoo Finance

    # Sort by composite score descending
    candidates.sort(key=lambda x: x["score"], reverse=True)
    logger.info(f"  {market_key.upper()}: {len(candidates)} candidates passed filters")
    return candidates


def run_all_markets() -> list:
    """Run scanner across markets based on MARKET_CONTEXT."""
    if MARKET_CONTEXT == "all":
        markets = ["us", "uk", "de", "jp"]
    elif MARKET_CONTEXT == "eu":
        markets = ["uk", "de"]
    else:
        markets = [MARKET_CONTEXT]

    all_candidates = []
    for market in markets:
        if market in TICKER_UNIVERSE:
            all_candidates.extend(scan_market(market))

    # Final global sort by score
    all_candidates.sort(key=lambda x: x["score"], reverse=True)
    return all_candidates


# ═══════════════════════════════════════════════════════════════════════════════
#  OUTPUT FORMATTERS
# ═══════════════════════════════════════════════════════════════════════════════

def format_markdown_table(candidates: list, top_n: int = 20) -> str:
    """
    Generate a clean Markdown table of top candidates.
    Includes all key metrics and net yield projection.
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    top = candidates[:top_n]

    if not top:
        return f"## Intraday Scanner — {now}\n\n**No candidates met the filter criteria.**\n"

    lines = [
        f"## 📊 Intraday Momentum Scanner — {now}",
        f"**Market:** {MARKET_CONTEXT.upper()}  |  **Target move:** >{MIN_MOVE_PCT}%  |  **Candidates:** {len(candidates)}",
        "",
        "> ⚠️ **Risk Disclaimer:** Consistently achieving >5% net daily returns is statistically rare",
        "> and highly speculative. This output is for research only. Past ATR does not guarantee",
        "> future moves. Never risk capital you cannot afford to lose entirely.",
        "",
        "| # | Ticker | Mkt | Price | Day% | Range% | ATR% | RVOL | RSI | Score | Net 1x | Prob |",
        "|---|--------|-----|-------|------|--------|------|------|-----|-------|--------|------|",
    ]

    for i, c in enumerate(top, 1):
        day_emoji  = "🟢" if c["day_return_pct"] > 0 else "🔴"
        net_emoji  = "✅" if c["net_yield_1x"] > 0 else "❌"
        prob_emoji = "🔥" if "HIGH" in c["probability"] else "⚡" if "MEDIUM" in c["probability"] else "❄️"

        lines.append(
            f"| {i} | **{c['ticker']}** | {c['market']} | "
            f"{c['price']:.2f} | "
            f"{day_emoji}{c['day_return_pct']:+.1f}% | "
            f"{c['day_range_pct']:.1f}% | "
            f"{c['atr_pct']:.1f}% | "
            f"{c['rvol']:.1f}x | "
            f"{c['rsi']:.0f} | "
            f"**{c['score']:.0f}** | "
            f"{net_emoji}{c['net_yield_1x']:+.1f}% | "
            f"{prob_emoji} {c['probability'].split('—')[0].strip()} |"
        )

    lines += [
        "",
        "### Column Guide",
        "- **Day%** = today's return so far  |  **Range%** = high-low range today",
        "- **ATR%** = 14-day avg daily range capability  |  **RVOL** = relative volume vs 20-day avg",
        "- **RSI** = 14-period RSI  |  **Score** = composite momentum score (0-100)",
        f"- **Net 1x** = projected net yield at 1x leverage after fees/spread/slippage (breakeven ~{candidates[0]['breakeven_pct'] if candidates else 0:.2f}%)",
        "",
        "### Top 3 Deep Dive",
    ]

    for c in top[:3]:
        lines += [
            f"**{c['ticker']}** ({c['market']}) — Score: {c['score']:.0f}",
            f"- ATR capability: {c['atr_pct']:.2f}% daily | RVOL: {c['rvol']:.1f}x | RSI: {c['rsi']:.0f}",
            f"- 5-day momentum: {c['momentum_5d']:+.1f}% | Today: {c['day_return_pct']:+.1f}%",
            f"- Net yield (1x leverage): {c['net_yield_1x']:+.2f}% | Probability tier: {c['probability']}",
            f"- Volume today: {c['volume']:,}",
            "",
        ]

    return "\n".join(lines)


def format_telegram_alert(candidates: list) -> str:
    """Compact Telegram message for top candidates."""
    now   = datetime.now(timezone.utc).strftime("%H:%M UTC")
    top   = candidates[:8]
    lines = [
        f"📊 <b>Intraday Scanner — {now}</b>",
        f"Market: {MARKET_CONTEXT.upper()} | Target: >{MIN_MOVE_PCT}% | Found: {len(candidates)}",
        "",
    ]
    for i, c in enumerate(top, 1):
        day_emoji = "🟢" if c["day_return_pct"] > 0 else "🔴"
        lines.append(
            f"{i}. <b>{c['ticker']}</b> {day_emoji}{c['day_return_pct']:+.1f}% | "
            f"ATR {c['atr_pct']:.1f}% | RVOL {c['rvol']:.1f}x | "
            f"Score <b>{c['score']:.0f}</b>"
        )
    lines += [
        "",
        "⚠️ Research only. Not financial advice.",
    ]
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
#  NOTIFICATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def send_telegram(text: str) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram credentials not set")
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


def send_telegram_document(filepath: str, caption: str = "") -> bool:
    """Send Markdown file as document to Telegram."""
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
        logger.error(f"Telegram document error: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    logger.info("=" * 60)
    logger.info("INTRADAY MOMENTUM SCANNER — starting")
    logger.info(f"Market context: {MARKET_CONTEXT.upper()}")
    logger.info(f"Target move:    >{MIN_MOVE_PCT}%")
    logger.info(f"Mode:           {'DRY RUN' if DRY_RUN else 'LIVE'}")
    logger.info("=" * 60)

    # Run scanner
    candidates = run_all_markets()
    logger.info(f"Total candidates: {len(candidates)}")

    # Generate Markdown report
    md_report = format_markdown_table(candidates, top_n=25)
    report_file = f"scan_{MARKET_CONTEXT}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}.md"

    with open(report_file, "w") as f:
        f.write(md_report)
    logger.info(f"Report saved: {report_file}")

    # Print to stdout (visible in GitHub Actions logs)
    print("\n" + md_report)

    # Send Telegram alerts
    if not DRY_RUN and candidates:
        alert_text = format_telegram_alert(candidates)
        send_telegram(alert_text)
        send_telegram_document(report_file, f"Full scan report — {MARKET_CONTEXT.upper()}")
        logger.info("Telegram alerts sent")
    elif DRY_RUN:
        logger.info("DRY RUN — Telegram suppressed")
    else:
        logger.info("No candidates — no alert sent")

    logger.info("=" * 60)
    logger.info(f"Run complete — {len(candidates)} candidates found")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
