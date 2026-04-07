"""
ta_runner.py
────────────────────────────────────────────────────────────
Automated TA Entry Framework — runs after intraday scanner.
For each qualifying ticker, computes:
  - Trend classification
  - Key support / resistance levels
  - RSI, MACD, ADX
  - Entry zone, Stop Loss, T1, T2, Recommended Exit
  - R/R ratios
  - Verdict: ENTER NOW / WAIT FOR DIP / WAIT FOR BREAKOUT / PASS

Outputs:
  - orders.json   -> machine-readable, broker-ready (top 10 ranked)
  - ta_report.md  -> full audit trail (all tickers)
  - Telegram      -> top 10 only, sorted highest return probability first

Usage:
  python ta_runner.py --tickers AAPL MSFT ARHS
  python ta_runner.py --from-file scan_results.json
"""

import os
import json
import argparse
import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

LOOKBACK_DAYS       = 252
MIN_RR_T1           = 1.2
MIN_RR_T2           = 2.0
MAX_STOP_PCT        = 0.25
MIN_STOP_PCT        = 0.05
ADX_TREND_THRESHOLD = 25
TOP_N               = 10     # Top N ranked setups sent to Telegram

TELEGRAM_BOT_TOKEN  = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID    = os.getenv("TELEGRAM_CHAT_ID", "")
DRY_RUN             = os.getenv("DRY_RUN", "false").lower() == "true"


# ─────────────────────────────────────────────────────────────────────────────
# TECHNICAL INDICATORS
# ─────────────────────────────────────────────────────────────────────────────

def compute_rsi(close, period=14):
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    rsi      = 100 - (100 / (1 + rs))
    return round(float(rsi.iloc[-1]), 2)


def compute_macd(close):
    ema12  = close.ewm(span=12, adjust=False).mean()
    ema26  = close.ewm(span=26, adjust=False).mean()
    macd   = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist   = macd - signal
    return (
        round(float(macd.iloc[-1]), 4),
        round(float(signal.iloc[-1]), 4),
        round(float(hist.iloc[-1]), 4),
    )


def compute_adx(high, low, close, period=14):
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    dm_pos = high.diff().clip(lower=0)
    dm_neg = (-low.diff()).clip(lower=0)
    dm_pos = dm_pos.where(dm_pos > dm_neg, 0)
    dm_neg = dm_neg.where(dm_neg > dm_pos, 0)
    tr_s = tr.ewm(com=period - 1, min_periods=period).mean()
    dip  = 100 * dm_pos.ewm(com=period - 1, min_periods=period).mean() / tr_s
    din  = 100 * dm_neg.ewm(com=period - 1, min_periods=period).mean() / tr_s
    dx   = 100 * (dip - din).abs() / (dip + din).replace(0, np.nan)
    adx  = dx.ewm(com=period - 1, min_periods=period).mean()
    return round(float(adx.iloc[-1]), 2)


def compute_atr(high, low, close, period=14):
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    return round(float(tr.ewm(com=period - 1, min_periods=period).mean().iloc[-1]), 4)


def find_support_resistance(df, current_price):
    levels = []
    for w in [10, 20, 50]:
        subset = df.tail(max(w * 2, 30))
        for i in range(w, len(subset) - w):
            if subset["Low"].iloc[i] == subset["Low"].iloc[i - w: i + w + 1].min():
                levels.append(round(float(subset["Low"].iloc[i]), 4))
            if subset["High"].iloc[i] == subset["High"].iloc[i - w: i + w + 1].max():
                levels.append(round(float(subset["High"].iloc[i]), 4))

    for ma_col in ["MA20", "MA50", "MA100", "MA200"]:
        if ma_col in df.columns:
            val = float(df[ma_col].iloc[-1])
            if not math.isnan(val):
                levels.append(round(val, 4))

    levels = sorted(set(levels))
    clustered, used = [], set()
    for lv in levels:
        if lv in used:
            continue
        cluster  = [x for x in levels if abs(x - lv) / lv < 0.015]
        clustered.append(round(sum(cluster) / len(cluster), 4))
        for x in cluster:
            used.add(x)

    supports    = sorted([l for l in clustered if l < current_price * 0.99], reverse=True)
    resistances = sorted([l for l in clustered if l > current_price * 1.01])
    return {"supports": supports[:5], "resistances": resistances[:5]}


# ─────────────────────────────────────────────────────────────────────────────
# CORE TA ENTRY ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def run_ta_entry(ticker, scanner_data=None):
    result = {
        "ticker": ticker, "status": "error", "error": None,
        "current_price": None, "currency": "USD",
        "entry_low": None, "entry_high": None,
        "stop_loss": None, "stop_pct": None,
        "target_1": None, "rr_t1": None,
        "target_2": None, "rr_t2": None,
        "recommended_exit": None, "rr_exit": None,
        "verdict": None, "rsi": None, "macd": None, "adx": None,
        "trend_primary": None, "trend_secondary": None, "momentum_st": None,
        "ma50": None, "ma200": None,
        "week52_high": None, "week52_low": None,
        "atr": None, "supports": [], "resistances": [],
        "catalyst_note": None, "pass_reason": None,
    }

    try:
        end_dt   = datetime.utcnow()
        start_dt = end_dt - timedelta(days=LOOKBACK_DAYS + 50)
        df = yf.download(
            ticker,
            start=start_dt.strftime("%Y-%m-%d"),
            end=end_dt.strftime("%Y-%m-%d"),
            progress=False
        )
        if df.empty or len(df) < 60:
            result["error"] = "Insufficient price history"
            return result

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna(subset=["Close", "High", "Low", "Volume"])

        current_price         = float(df["Close"].iloc[-1])
        result["current_price"] = round(current_price, 4)

        try:
            info = yf.Ticker(ticker).info
            result["currency"] = info.get("currency", "USD")
            ed = info.get("earningsDate") or info.get("earningsTimestamp")
            if ed:
                if isinstance(ed, (int, float)):
                    ed = datetime.fromtimestamp(ed).strftime("%Y-%m-%d")
                result["catalyst_note"] = f"Earnings: {ed}"
        except Exception:
            pass

        df["MA20"]  = df["Close"].rolling(20).mean()
        df["MA50"]  = df["Close"].rolling(50).mean()
        df["MA100"] = df["Close"].rolling(100).mean()
        df["MA200"] = df["Close"].rolling(200).mean()

        ma20  = float(df["MA20"].iloc[-1])
        ma50  = float(df["MA50"].iloc[-1])
        ma200 = float(df["MA200"].iloc[-1])
        result["ma50"]  = round(ma50, 4)
        result["ma200"] = round(ma200, 4)

        last_252 = df.tail(252)
        result["week52_high"] = round(float(last_252["High"].max()), 4)
        result["week52_low"]  = round(float(last_252["Low"].min()), 4)

        rsi             = compute_rsi(df["Close"])
        macd, sig, hist = compute_macd(df["Close"])
        adx             = compute_adx(df["High"], df["Low"], df["Close"])
        atr             = compute_atr(df["High"], df["Low"], df["Close"])
        result["rsi"]   = rsi
        result["macd"]  = macd
        result["adx"]   = adx
        result["atr"]   = round(atr, 4)

        if current_price > ma50 and ma50 > ma200:
            primary = "UPTREND"
        elif current_price < ma50 and ma50 < ma200:
            primary = "DOWNTREND"
        else:
            primary = "MIXED"

        secondary   = "UPTREND" if current_price > ma20 else "DOWNTREND"
        momentum_st = "BULLISH" if (rsi > 55 and hist > 0) else ("BEARISH" if (rsi < 45 and hist < 0) else "NEUTRAL")

        result["trend_primary"]   = primary
        result["trend_secondary"] = secondary
        result["momentum_st"]     = momentum_st

        levels      = find_support_resistance(df, current_price)
        supports    = levels["supports"]
        resistances = levels["resistances"]
        result["supports"]    = supports
        result["resistances"] = resistances

        nearest_support = supports[0] if supports else current_price * 0.95
        entry_low  = round(nearest_support * 1.005, 4)
        entry_high = round(current_price * 1.005, 4)
        entry_mid  = round((entry_low + entry_high) / 2, 4)

        stop_loss = round(nearest_support - 1.5 * atr, 4)
        stop_pct  = round((entry_mid - stop_loss) / entry_mid, 4)

        if stop_pct < MIN_STOP_PCT:
            stop_loss = round(entry_mid * (1 - MIN_STOP_PCT), 4)
            stop_pct  = MIN_STOP_PCT

        if stop_pct > MAX_STOP_PCT:
            result["status"]      = "pass"
            result["pass_reason"] = f"Stop {stop_pct:.1%} exceeds 25% max -- no clean stop"
            result["verdict"]     = "PASS"
            return result

        result["entry_low"]  = entry_low
        result["entry_high"] = entry_high
        result["stop_loss"]  = stop_loss
        result["stop_pct"]   = round(stop_pct * 100, 2)

        risk = entry_mid - stop_loss

        t1    = next((r for r in resistances if r > entry_high), None)
        if t1 is None:
            t1 = round(entry_mid + 1.5 * risk, 4)
        rr_t1 = round((t1 - entry_mid) / risk, 2)
        if rr_t1 < MIN_RR_T1:
            result["status"]      = "pass"
            result["pass_reason"] = f"T1 R/R {rr_t1:.2f} below minimum {MIN_RR_T1}"
            result["verdict"]     = "PASS"
            return result

        t2    = next((r for r in resistances if r > t1 * 1.01), None)
        if t2 is None:
            t2 = round(entry_mid + 2.5 * risk, 4)
        rr_t2 = round((t2 - entry_mid) / risk, 2)
        if rr_t2 < MIN_RR_T2:
            t2    = round(entry_mid + 2.5 * risk, 4)
            rr_t2 = round((t2 - entry_mid) / risk, 2)

        exit_price = next((r for r in resistances if r > t2 * 1.02), None)
        if exit_price is None:
            exit_price = round(entry_mid + 4 * risk, 4)
        exit_price = min(exit_price, round(result["week52_high"] * 0.95, 4))
        rr_exit    = round((exit_price - entry_mid) / risk, 2)

        result["target_1"]         = round(t1, 4)
        result["rr_t1"]            = rr_t1
        result["target_2"]         = round(t2, 4)
        result["rr_t2"]            = rr_t2
        result["recommended_exit"] = round(exit_price, 4)
        result["rr_exit"]          = rr_exit

        at_support      = current_price <= nearest_support * 1.02
        near_resistance = bool(resistances and current_price >= resistances[0] * 0.97)
        at_52w_low      = current_price <= result["week52_low"] * 1.05

        if at_support and momentum_st != "BEARISH" and primary != "DOWNTREND":
            verdict = "ENTER NOW"
        elif current_price <= entry_high and primary == "UPTREND":
            verdict = "ENTER NOW"
        elif near_resistance:
            verdict = f"WAIT FOR BREAKOUT ABOVE {resistances[0]:.2f}"
        elif current_price > entry_high:
            verdict = f"WAIT FOR DIP TO {entry_low:.2f}"
        else:
            verdict = "ENTER NOW"

        if primary == "DOWNTREND" and not at_52w_low:
            verdict = f"WAIT FOR BREAKOUT ABOVE {ma50:.2f} (50-day MA)"

        result["verdict"] = verdict
        result["status"]  = "ok"

    except Exception as e:
        result["error"] = str(e)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# RANKING — highest return probability first
# ─────────────────────────────────────────────────────────────────────────────

def rank_by_return_probability(actionable, top_n=TOP_N):
    """
    Score each actionable setup and return top_n sorted best-first.

    Score (max 100):
      Verdict urgency  — ENTER NOW=30, WAIT FOR DIP=20, WAIT FOR BREAKOUT=10
      R/R quality      — rr_exit up to 6:1 mapped to 0-30 pts
      Trend alignment  — UPTREND=20, MIXED=10, DOWNTREND=5
      RSI room to run  — 40-65=20 (optimal), 65-75=15, else=5
    """
    def score(r):
        v     = r.get("verdict", "")
        vs    = 30 if v.startswith("ENTER") else (20 if v.startswith("WAIT FOR DIP") else 10)
        rr    = min(r.get("rr_exit", 0) / 6.0, 1.0) * 30
        trend = r.get("trend_primary", "")
        ts    = 20 if trend == "UPTREND" else (10 if trend == "MIXED" else 5)
        rsi   = r.get("rsi") or 50
        rs    = 20 if 40 <= rsi <= 65 else (15 if 65 < rsi <= 75 else 5)
        return vs + rr + ts + rs

    return sorted(actionable, key=score, reverse=True)[:top_n]


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT FORMATTERS
# ─────────────────────────────────────────────────────────────────────────────

def format_telegram_card(r, rank):
    ccy     = r.get("currency", "USD")
    sym     = r.get("ticker", "?")
    price   = r.get("current_price", 0)
    mid     = (r["entry_low"] + r["entry_high"]) / 2
    entry   = f"{r['entry_low']:.2f}-{r['entry_high']:.2f}"
    stop    = f"{r['stop_loss']:.2f} (-{r['stop_pct']:.1f}%)"
    t1_pct  = (r["target_1"] - mid) / mid * 100
    t2_pct  = (r["target_2"] - mid) / mid * 100
    ex_pct  = (r["recommended_exit"] - mid) / mid * 100
    t1      = f"{r['target_1']:.2f} (+{t1_pct:.1f}%) R/R {r['rr_t1']:.1f}:1"
    t2      = f"{r['target_2']:.2f} (+{t2_pct:.1f}%) R/R {r['rr_t2']:.1f}:1"
    ex      = f"{r['recommended_exit']:.2f} (+{ex_pct:.1f}%) R/R {r['rr_exit']:.1f}:1"
    verdict  = r.get("verdict", "-")
    catalyst = r.get("catalyst_note") or "No near-term catalyst"

    if verdict.startswith("ENTER"):
        tag = "[ENTER NOW]"
    elif verdict.startswith("WAIT FOR DIP"):
        tag = "[WAIT - DIP]"
    else:
        tag = "[WAIT - BRKOUT]"

    return "\n".join([
        f"#{rank} {sym} | {ccy} {price:.2f} | {r.get('trend_primary','-')} | RSI {r.get('rsi','-')}",
        f"",
        f"Entry:  {ccy} {entry}",
        f"Stop:   {ccy} {stop}",
        f"T1:     {ccy} {t1}",
        f"T2:     {ccy} {t2}",
        f"Exit:   {ccy} {ex}",
        f"",
        f"{tag} {verdict}",
        f"Catalyst: {catalyst}",
        f"-----------------------------",
    ])


def format_markdown_row(r):
    if r["status"] == "pass":
        return f"| {r['ticker']} | PASS | - | - | - | - | {r.get('pass_reason','-')} |"
    if r["status"] == "error":
        return f"| {r['ticker']} | ERROR | - | - | - | - | {r.get('error','-')} |"
    return (
        f"| {r['ticker']} "
        f"| {r['current_price']:.2f} "
        f"| {r['entry_low']:.2f}-{r['entry_high']:.2f} "
        f"| {r['stop_loss']:.2f} (-{r['stop_pct']:.1f}%) "
        f"| {r['target_1']:.2f} ({r['rr_t1']:.1f}:1) "
        f"| {r['recommended_exit']:.2f} ({r['rr_exit']:.1f}:1) "
        f"| {r['verdict']} |"
    )


# ─────────────────────────────────────────────────────────────────────────────
# TELEGRAM SENDER
# ─────────────────────────────────────────────────────────────────────────────

def send_telegram(text):
    if DRY_RUN:
        print(f"[DRY RUN] Telegram:\n{text}\n")
        return True
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("WARN: Telegram credentials not set -- skipping")
        return False
    import urllib.request
    url     = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = json.dumps({
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True
    }).encode()
    try:
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"}
        )
        urllib.request.urlopen(req, timeout=15)
        return True
    except Exception as e:
        print(f"ERROR sending Telegram: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TA Entry Runner")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--tickers",   nargs="+", help="Space-separated tickers")
    group.add_argument("--from-file", help="Path to scan_results.json")
    args = parser.parse_args()

    if args.tickers:
        tickers      = args.tickers
        scanner_data = {}
    else:
        with open(args.from_file) as f:
            scan_data = json.load(f)
        tickers      = [d["ticker"] for d in scan_data]
        scanner_data = {d["ticker"]: d for d in scan_data}

    now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    print(f"\n{'='*60}")
    print(f"TA ENTRY RUNNER -- {now_str}")
    print(f"Processing {len(tickers)} ticker(s): {', '.join(tickers)}")
    print(f"{'='*60}\n")

    results    = []
    actionable = []

    for ticker in tickers:
        print(f"  -> {ticker}...", end=" ", flush=True)
        r = run_ta_entry(ticker, scanner_data.get(ticker, {}))
        results.append(r)

        if r["status"] == "ok":
            print(f"OK | {r['verdict'][:50]}")
            actionable.append(r)
        elif r["status"] == "pass":
            print(f"PASS -- {r['pass_reason']}")
        else:
            print(f"ERROR -- {r['error']}")

    # Rank by return probability, take top N
    top10 = rank_by_return_probability(actionable, top_n=TOP_N)

    print(f"\n  Ranked top {len(top10)} from {len(actionable)} actionable:")
    for i, r in enumerate(top10, 1):
        print(f"    {i}. {r['ticker']} | {r.get('trend_primary','-')} | {r.get('verdict','-')[:50]}")

    # Write orders.json — top 10 only, ranked
    orders = [
        {
            "rank":             i + 1,
            "ticker":           r["ticker"],
            "currency":         r["currency"],
            "timestamp_utc":    datetime.utcnow().isoformat(),
            "current_price":    r["current_price"],
            "entry_limit_low":  r["entry_low"],
            "entry_limit_high": r["entry_high"],
            "stop_loss":        r["stop_loss"],
            "stop_pct":         r["stop_pct"],
            "target_1":         r["target_1"],
            "rr_t1":            r["rr_t1"],
            "target_2":         r["target_2"],
            "rr_t2":            r["rr_t2"],
            "recommended_exit": r["recommended_exit"],
            "rr_exit":          r["rr_exit"],
            "verdict":          r["verdict"],
            "catalyst":         r.get("catalyst_note"),
            "rsi":              r["rsi"],
            "ma50":             r["ma50"],
            "ma200":            r["ma200"],
            "week52_low":       r["week52_low"],
            "week52_high":      r["week52_high"],
        }
        for i, r in enumerate(top10)
    ]
    with open("orders.json", "w") as f:
        json.dump(orders, f, indent=2)
    print(f"\norders.json written -- {len(orders)} ranked setup(s)")

    # Write ta_report.md — full audit trail of all tickers
    md = [
        f"# TA Entry Report -- {now_str}",
        f"",
        f"**{len(actionable)} actionable** from {len(tickers)} scanned | "
        f"**Top {len(top10)} ranked** by return probability",
        f"",
        f"| Ticker | Price | Entry Zone | Stop | T1 (R/R) | Exit (R/R) | Verdict |",
        f"|--------|-------|------------|------|----------|-----------|---------|",
    ]
    for r in results:
        md.append(format_markdown_row(r))

    with open("ta_report.md", "w") as f:
        f.write("\n".join(md))
    print("ta_report.md written")

    # Send Telegram — header + top 10 cards ranked best-first
    if top10:
        send_telegram(
            f"TA Entry -- {now_str}\n"
            f"Top {len(top10)} setups ranked by return probability\n"
            f"({len(actionable)} actionable from {len(tickers)} scanned)\n"
            f"------------------------------"
        )
        for i, r in enumerate(top10, 1):
            send_telegram(format_telegram_card(r, rank=i))
    else:
        send_telegram(
            f"TA Runner -- {now_str}\n"
            f"0 actionable setups from {len(tickers)} scanned."
        )

    print(f"\n{'='*60}")
    print(f"Done. {len(actionable)}/{len(tickers)} actionable. Top {len(top10)} sent to Telegram.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
