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
  - orders.json   -> machine-readable, broker-ready
  - ta_report.md  -> human-readable summary
  - Telegram alert per actionable ticker (R/R >= 1.2:1 on T1)

Usage:
  python ta_runner.py --tickers AAPL MSFT ARHS
  python ta_runner.py --from-file scan_results.json

Risk disclaimer: This module generates technical entry parameters for
informational purposes only. It is not financial advice. All entry,
stop and target levels are algorithmic estimates. Always verify with
your broker and apply your own risk management before placing orders.
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
    windows = [10, 20, 50]

    for w in windows:
        subset = df.tail(max(w * 2, 30))
        for i in range(w, len(subset) - w):
            low_slice = subset["Low"].iloc[i - w: i + w + 1]
            if subset["Low"].iloc[i] == low_slice.min():
                levels.append(round(float(subset["Low"].iloc[i]), 4))
        for i in range(w, len(subset) - w):
            high_slice = subset["High"].iloc[i - w: i + w + 1]
            if subset["High"].iloc[i] == high_slice.max():
                levels.append(round(float(subset["High"].iloc[i]), 4))

    for ma_col in ["MA20", "MA50", "MA100", "MA200"]:
        if ma_col in df.columns:
            val = float(df[ma_col].iloc[-1])
            if not math.isnan(val):
                levels.append(round(val, 4))

    levels = sorted(set(levels))

    # Cluster levels within 1.5% of each other
    clustered = []
    used = set()
    for lv in levels:
        if lv in used:
            continue
        cluster  = [x for x in levels if abs(x - lv) / lv < 0.015]
        centroid = round(sum(cluster) / len(cluster), 4)
        clustered.append(centroid)
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
        # 1. Fetch price history
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

        # 2. Current price
        current_price         = float(df["Close"].iloc[-1])
        result["current_price"] = round(current_price, 4)

        # Earnings date
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

        # 3. Moving averages
        df["MA20"]  = df["Close"].rolling(20).mean()
        df["MA50"]  = df["Close"].rolling(50).mean()
        df["MA100"] = df["Close"].rolling(100).mean()
        df["MA200"] = df["Close"].rolling(200).mean()

        ma20  = float(df["MA20"].iloc[-1])
        ma50  = float(df["MA50"].iloc[-1])
        ma200 = float(df["MA200"].iloc[-1])
        result["ma50"]  = round(ma50, 4)
        result["ma200"] = round(ma200, 4)

        # 4. 52-week range
        last_252         = df.tail(252)
        w52_high         = float(last_252["High"].max())
        w52_low          = float(last_252["Low"].min())
        result["week52_high"] = round(w52_high, 4)
        result["week52_low"]  = round(w52_low, 4)

        # 5. Indicators
        rsi             = compute_rsi(df["Close"])
        macd, sig, hist = compute_macd(df["Close"])
        adx             = compute_adx(df["High"], df["Low"], df["Close"])
        atr             = compute_atr(df["High"], df["Low"], df["Close"])
        result["rsi"]   = rsi
        result["macd"]  = macd
        result["adx"]   = adx
        result["atr"]   = round(atr, 4)

        # 6. Trend classification
        if current_price > ma50 and ma50 > ma200:
            primary = "UPTREND"
        elif current_price < ma50 and ma50 < ma200:
            primary = "DOWNTREND"
        else:
            primary = "MIXED"

        secondary = "UPTREND" if current_price > ma20 else "DOWNTREND"

        if rsi > 55 and hist > 0:
            momentum_st = "BULLISH"
        elif rsi < 45 and hist < 0:
            momentum_st = "BEARISH"
        else:
            momentum_st = "NEUTRAL"

        result["trend_primary"]   = primary
        result["trend_secondary"] = secondary
        result["momentum_st"]     = momentum_st

        # 7. Support / resistance
        levels      = find_support_resistance(df, current_price)
        supports    = levels["supports"]
        resistances = levels["resistances"]
        result["supports"]    = supports
        result["resistances"] = resistances

        # 8. Entry zone
        nearest_support = supports[0] if supports else current_price * 0.95
        entry_low  = round(nearest_support * 1.005, 4)
        entry_high = round(current_price * 1.005, 4)
        entry_mid  = round((entry_low + entry_high) / 2, 4)

        # 9. Stop loss — below nearest support by 1.5x ATR
        stop_loss = round(nearest_support - 1.5 * atr, 4)
        stop_pct  = round((entry_mid - stop_loss) / entry_mid, 4)

        if stop_pct < MIN_STOP_PCT:
            stop_loss = round(entry_mid * (1 - MIN_STOP_PCT), 4)
            stop_pct  = MIN_STOP_PCT

        if stop_pct > MAX_STOP_PCT:
            result["status"]      = "pass"
            result["pass_reason"] = f"Stop {stop_pct:.1%} exceeds 25% max — no clean stop"
            result["verdict"]     = "PASS"
            return result

        result["entry_low"]  = entry_low
        result["entry_high"] = entry_high
        result["stop_loss"]  = stop_loss
        result["stop_pct"]   = round(stop_pct * 100, 2)

        # 10. Targets
        risk = entry_mid - stop_loss

        t1 = next((r for r in resistances if r > entry_high), None)
        if t1 is None:
            t1 = round(entry_mid + 1.5 * risk, 4)

        rr_t1 = round((t1 - entry_mid) / risk, 2)
        if rr_t1 < MIN_RR_T1:
            result["status"]      = "pass"
            result["pass_reason"] = f"T1 R/R {rr_t1:.2f} below minimum {MIN_RR_T1}"
            result["verdict"]     = "PASS"
            return result

        t2 = next((r for r in resistances if r > t1 * 1.01), None)
        if t2 is None:
            t2 = round(entry_mid + 2.5 * risk, 4)

        rr_t2 = round((t2 - entry_mid) / risk, 2)
        if rr_t2 < MIN_RR_T2:
            t2    = round(entry_mid + 2.5 * risk, 4)
            rr_t2 = round((t2 - entry_mid) / risk, 2)

        exit_price = next((r for r in resistances if r > t2 * 1.02), None)
        if exit_price is None:
            exit_price = round(entry_mid + 4 * risk, 4)
        exit_price = min(exit_price, round(w52_high * 0.95, 4))
        rr_exit    = round((exit_price - entry_mid) / risk, 2)

        result["target_1"]         = round(t1, 4)
        result["rr_t1"]            = rr_t1
        result["target_2"]         = round(t2, 4)
        result["rr_t2"]            = rr_t2
        result["recommended_exit"] = round(exit_price, 4)
        result["rr_exit"]          = rr_exit

        # 11. Verdict
        at_support      = current_price <= nearest_support * 1.02
        near_resistance = bool(resistances and current_price >= resistances[0] * 0.97)
        at_52w_low      = current_price <= w52_low * 1.05

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

        # Override: downtrend only valid entry if at 52-week low
        if primary == "DOWNTREND" and not at_52w_low:
            verdict = f"WAIT FOR BREAKOUT ABOVE {ma50:.2f} (50-day MA)"

        result["verdict"] = verdict
        result["status"]  = "ok"

    except Exception as e:
        result["error"] = str(e)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT FORMATTERS
# ─────────────────────────────────────────────────────────────────────────────

def format_telegram_card(r):
    ccy     = r.get("currency", "USD")
    sym     = r.get("ticker", "?")
    price   = r.get("current_price", 0)
    mid     = (r["entry_low"] + r["entry_high"]) / 2
    entry   = f"{r['entry_low']:.2f}-{r['entry_high']:.2f}"
    stop    = f"{r['stop_loss']:.2f} (-{r['stop_pct']:.1f}%)"
    t1_pct  = (r["target_1"] - mid) / mid * 100
    t2_pct  = (r["target_2"] - mid) / mid * 100
    ex_pct  = (r["recommended_exit"] - mid) / mid * 100
    t1      = f"{r['target_1']:.2f} (+{t1_pct:.1f}%) | R/R {r['rr_t1']:.1f}:1"
    t2      = f"{r['target_2']:.2f} (+{t2_pct:.1f}%) | R/R {r['rr_t2']:.1f}:1"
    ex      = f"{r['recommended_exit']:.2f} (+{ex_pct:.1f}%) | R/R {r['rr_exit']:.1f}:1"
    verdict  = r.get("verdict", "-")
    catalyst = r.get("catalyst_note") or "No near-term catalyst identified"

    if verdict.startswith("ENTER"):
        emoji = "GREEN"
    elif verdict.startswith("WAIT FOR DIP"):
        emoji = "YELLOW"
    elif verdict.startswith("WAIT FOR BREAKOUT"):
        emoji = "ORANGE"
    else:
        emoji = "WHITE"

    return "\n".join([
        f"TA Entry -- {sym} | {ccy} {price:.2f}",
        f"",
        f"Trend: {r.get('trend_primary','?')} | RSI: {r.get('rsi','?')} | MACD: {r.get('macd','?')}",
        f"",
        f"Entry Zone:   {ccy} {entry}",
        f"Stop Loss:    {ccy} {stop}",
        f"Target 1:     {ccy} {t1}",
        f"Target 2:     {ccy} {t2}",
        f"Exit:         {ccy} {ex}",
        f"",
        f"[{emoji}] VERDICT: {verdict}",
        f"",
        f"Catalyst: {catalyst}",
        f"-----------------------------",
        f"Not investment advice.",
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
# SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────

def format_summary_table(actionable, now_str, total_scanned):
    """
    Clean summary table of all actionable setups.
    Columns: Ticker | Trend | Entry Zone | Stop Loss | T1 | T2 | Exit | Verdict | Catalyst
    """
    if not actionable:
        return ""

    lines = [
        f"",
        f"## TA Entry Summary -- {now_str}",
        f"**{len(actionable)} actionable setup(s) from {total_scanned} scanned**",
        f"",
        f"| # | Ticker | Trend | Entry Zone | Stop Loss | Target 1 | Target 2 | Exit | Verdict | Catalyst |",
        f"|---|--------|-------|------------|-----------|----------|----------|------|---------|----------|",
    ]

    for i, r in enumerate(actionable, 1):
        ccy      = r.get("currency", "USD")
        ticker   = r["ticker"]
        trend    = r.get("trend_primary", "-")
        entry    = f"{ccy} {r['entry_low']:.2f}-{r['entry_high']:.2f}"
        stop     = f"{ccy} {r['stop_loss']:.2f} (-{r['stop_pct']:.1f}%)"
        t1       = f"{ccy} {r['target_1']:.2f} ({r['rr_t1']:.1f}:1)"
        t2       = f"{ccy} {r['target_2']:.2f} ({r['rr_t2']:.1f}:1)"
        exit_p   = f"{ccy} {r['recommended_exit']:.2f} ({r['rr_exit']:.1f}:1)"
        verdict  = r.get("verdict", "-")
        catalyst = r.get("catalyst_note") or "None"
        lines.append(
            f"| {i} | **{ticker}** | {trend} | {entry} | {stop} "
            f"| {t1} | {t2} | {exit_p} | {verdict} | {catalyst} |"
        )

    return "\n".join(lines)


def format_summary_telegram(actionable, now_str, total_scanned):
    """Compact Telegram version of the summary table."""
    lines = [
        f"ACTIONABLE SETUPS SUMMARY -- {now_str}",
        f"{len(actionable)} selected from {total_scanned} scanned",
        f"",
    ]
    for i, r in enumerate(actionable, 1):
        ccy     = r.get("currency", "USD")
        verdict = r.get("verdict", "-")
        # Short verdict label
        if verdict.startswith("ENTER"):
            tag = "ENTER"
        elif verdict.startswith("WAIT FOR DIP"):
            tag = "DIP"
        elif verdict.startswith("WAIT FOR BREAKOUT"):
            tag = "BRKOUT"
        else:
            tag = verdict[:8]

        catalyst = r.get("catalyst_note") or "-"
        lines.append(
            f"{i}. {r['ticker']} | {r.get('trend_primary','-')} | "
            f"Entry {ccy}{r['entry_low']:.2f}-{r['entry_high']:.2f} | "
            f"Stop {ccy}{r['stop_loss']:.2f} | "
            f"T1 {ccy}{r['target_1']:.2f} | "
            f"T2 {ccy}{r['target_2']:.2f} | "
            f"Exit {ccy}{r['recommended_exit']:.2f} | "
            f"{tag} | {catalyst}"
        )
    lines.append("")
    lines.append("Not investment advice.")
    return "\n".join(lines)


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

    # Write orders.json
    orders = [
        {
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
        for r in actionable
    ]
    with open("orders.json", "w") as f:
        json.dump(orders, f, indent=2)
    print(f"\norders.json written -- {len(orders)} actionable setup(s)")

    # Write ta_report.md — full detail rows + summary table at the end
    md = [
        f"# TA Entry Report -- {now_str}",
        "",
        "## Full Analysis",
        "| Ticker | Price | Entry Zone | Stop | T1 (R/R) | Exit (R/R) | Verdict |",
        "|--------|-------|------------|------|----------|-----------|---------|",
    ]
    for r in results:
        md.append(format_markdown_row(r))

    # Append summary table of actionable setups only
    summary_md = format_summary_table(actionable, now_str, len(tickers))
    if summary_md:
        md.append(summary_md)

    with open("ta_report.md", "w") as f:
        f.write("\n".join(md))
    print("ta_report.md written")

    # Send Telegram — individual cards + final summary table
    if actionable:
        send_telegram(
            f"TA Entry Runner -- {now_str}\n"
            f"{len(actionable)} actionable setup(s) from {len(tickers)} scanned\n"
            f"------------------------------"
        )
        for r in actionable:
            send_telegram(format_telegram_card(r))
        # Final summary message
        send_telegram(format_summary_telegram(actionable, now_str, len(tickers)))
    else:
        send_telegram(
            f"TA Runner -- {now_str}\n"
            f"0 actionable setups from {len(tickers)} scanned."
        )

    print(f"\n{'='*60}")
    print(f"Done. {len(actionable)}/{len(tickers)} actionable.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
