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
from concurrent.futures import ThreadPoolExecutor, as_completed
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

# ── Position sizing ───────────────────────────────────────────────────────────
# Capital and risk % are read from env so they can be adjusted without code changes.
# Default: €250k capital, 2.5% risk per trade (middle of the 2-3% range).
CAPITAL_EUR = float(os.getenv("CAPITAL_EUR", "250000"))
RISK_PCT    = float(os.getenv("RISK_PCT",    "0.025"))


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
# POSITION SIZING
# ─────────────────────────────────────────────────────────────────────────────

def compute_position_sizing(entry_mid: float, stop_loss: float,
                            capital: float = CAPITAL_EUR,
                            risk_pct: float = RISK_PCT) -> dict:
    """
    Standard fixed-risk position sizing:
        risk_eur      = capital × risk_pct
        num_shares    = risk_eur / (entry - stop)
        position_eur  = num_shares × entry

    Returns a sizing dict, or None if stop_distance is invalid.
    Works for both LONG (entry > stop) — SHORT handled by abs().
    """
    risk_eur      = capital * risk_pct
    stop_distance = abs(entry_mid - stop_loss)
    if stop_distance <= 0:
        return None
    num_shares   = risk_eur / stop_distance
    position_eur = num_shares * entry_mid
    return {
        "risk_eur":      round(risk_eur),
        "position_eur":  round(position_eur),
        "num_shares":    int(round(num_shares)),
        "risk_pct_used": round(risk_pct * 100, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CORE TA ENTRY ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def run_ta_entry(ticker, scanner_data=None):
    result = {
        "ticker": ticker, "name": ticker, "status": "error", "error": None,
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
        # Position sizing (added — compute_position_sizing)
        "position_size_eur": None, "risk_eur": None,
        "num_shares": None, "risk_pct_used": None,
        # Bias + catalyst tier inherited from scanner
        "bias": "LONG", "catalyst_tier": "B",
        # Scanner context carried forward for display
        "rs_vs_bench": None, "day_return": None, "rvol": None,
    }

    # ── Inherit scanner context early — bias is needed before level calculations
    if scanner_data:
        result["bias"]          = scanner_data.get("bias", "LONG")
        result["catalyst_tier"] = scanner_data.get("catalyst_tier", "B")
        result["rs_vs_bench"]   = scanner_data.get("rs_vs_bench")
        result["day_return"]    = scanner_data.get("day_return")
        result["rvol"]          = scanner_data.get("rvol")
    bias = result["bias"]

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
            result["name"] = info.get("longName") or info.get("shortName") or ticker
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
        if bias == "SHORT":
            # For SHORT: bearish signals confirm direction (desired)
            momentum_st = "BEARISH" if (rsi < 45 and hist < 0) else ("BULLISH" if (rsi > 55 and hist > 0) else "NEUTRAL")
        else:
            momentum_st = "BULLISH" if (rsi > 55 and hist > 0) else ("BEARISH" if (rsi < 45 and hist < 0) else "NEUTRAL")

        result["trend_primary"]   = primary
        result["trend_secondary"] = secondary
        result["momentum_st"]     = momentum_st

        levels      = find_support_resistance(df, current_price)
        supports    = levels["supports"]
        resistances = levels["resistances"]
        result["supports"]    = supports
        result["resistances"] = resistances

        if bias == "SHORT":
            # ── SHORT trade: sell near resistance, stop above, targets below ──
            nearest_resistance = resistances[0] if resistances else current_price * 1.05
            entry_high = round(nearest_resistance * 0.995, 4)
            entry_low  = round(current_price * 0.995, 4)
            if entry_low > entry_high:
                entry_low, entry_high = entry_high, entry_low
            entry_mid  = round((entry_low + entry_high) / 2, 4)

            stop_loss = round(nearest_resistance + 1.5 * atr, 4)
            stop_pct  = round((stop_loss - entry_mid) / entry_mid, 4)

            if stop_pct < MIN_STOP_PCT:
                stop_loss = round(entry_mid * (1 + MIN_STOP_PCT), 4)
                stop_pct  = MIN_STOP_PCT
            if stop_pct > MAX_STOP_PCT:
                result["status"]      = "pass"
                result["pass_reason"] = f"Short stop {stop_pct:.1%} exceeds 25% max"
                result["verdict"]     = "PASS"
                return result

            result["entry_low"]  = entry_low
            result["entry_high"] = entry_high
            result["stop_loss"]  = stop_loss
            result["stop_pct"]   = round(stop_pct * 100, 2)

            risk = stop_loss - entry_mid   # positive distance from entry to stop

            # Downside targets — nearest support levels below entry (supports sorted descending)
            t1 = next((s for s in supports if s < entry_low * 0.99), None)
            if t1 is None:
                t1 = round(entry_mid - 1.5 * risk, 4)
            rr_t1 = round((entry_mid - t1) / risk, 2)
            if rr_t1 < MIN_RR_T1:
                result["status"]      = "pass"
                result["pass_reason"] = f"Short T1 R/R {rr_t1:.2f} below minimum {MIN_RR_T1}"
                result["verdict"]     = "PASS"
                return result

            t2 = next((s for s in supports if s < t1 * 0.99), None)
            if t2 is None:
                t2 = round(entry_mid - 2.5 * risk, 4)
            rr_t2 = round((entry_mid - t2) / risk, 2)
            if rr_t2 < MIN_RR_T2:
                t2    = round(entry_mid - 2.5 * risk, 4)
                rr_t2 = round((entry_mid - t2) / risk, 2)

            exit_price = next((s for s in supports if s < t2 * 0.98), None)
            if exit_price is None:
                exit_price = round(entry_mid - 4 * risk, 4)
            # Apply 52w low floor only when it still leaves room below T2
            w52_floor = round(result["week52_low"] * 1.05, 4)
            if w52_floor < t2:
                exit_price = max(exit_price, w52_floor)
            exit_price = min(exit_price, round(t2 * 0.98, 4))  # ceiling: always below T2
            rr_exit    = round((entry_mid - exit_price) / risk, 2)

            result["target_1"]         = round(t1, 4)
            result["rr_t1"]            = rr_t1
            result["target_2"]         = round(t2, 4)
            result["rr_t2"]            = rr_t2
            result["recommended_exit"] = round(exit_price, 4)
            result["rr_exit"]          = rr_exit

            sizing = compute_position_sizing(entry_mid, stop_loss)
            if sizing:
                result["position_size_eur"] = sizing["position_eur"]
                result["risk_eur"]          = sizing["risk_eur"]
                result["num_shares"]        = sizing["num_shares"]
                result["risk_pct_used"]     = sizing["risk_pct_used"]

            # Verdict for SHORT
            at_support_sh = bool(supports and current_price <= supports[0] * 1.03)
            if at_support_sh:
                verdict = f"WAIT FOR BREAKDOWN BELOW {supports[0]:.2f}"
            elif momentum_st == "BULLISH":
                verdict = f"WAIT — Counter-trend momentum (RSI {rsi:.0f})"
            else:
                verdict = "ENTER SHORT NOW"

        else:
            # ── LONG trade (original logic — unchanged) ───────────────────────
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
            # Apply 52w high cap only when it still leaves room above T2
            w52_cap = round(result["week52_high"] * 0.95, 4)
            if w52_cap > t2:
                exit_price = min(exit_price, w52_cap)
            exit_price = max(exit_price, round(t2 * 1.02, 4))  # floor: always above T2
            rr_exit    = round((exit_price - entry_mid) / risk, 2)

            result["target_1"]         = round(t1, 4)
            result["rr_t1"]            = rr_t1
            result["target_2"]         = round(t2, 4)
            result["rr_t2"]            = rr_t2
            result["recommended_exit"] = round(exit_price, 4)
            result["rr_exit"]          = rr_exit

            sizing = compute_position_sizing(entry_mid, stop_loss)
            if sizing:
                result["position_size_eur"] = sizing["position_eur"]
                result["risk_eur"]          = sizing["risk_eur"]
                result["num_shares"]        = sizing["num_shares"]
                result["risk_pct_used"]     = sizing["risk_pct_used"]

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
    Rank: tier first (A+ > A > B), then entry quality within each tier.
    Direction-aware: DOWNTREND is good alignment for SHORT, UPTREND for LONG.

    Score components (within-tier, max ~100):
      Verdict urgency  — ENTER NOW=30, WAIT FOR DIP/BREAKDOWN=20, else=10
      R/R quality      — rr_exit up to 6:1 mapped to 0-30 pts
      Trend alignment  — direction-aware: 20 / 10 / 5
      RSI              — direction-aware: optimal zone=20, edge=15, poor=5
    """
    tier_priority = {"A+": 300, "A": 200, "B": 100}

    def score(r):
        tier  = tier_priority.get(r.get("catalyst_tier", "B"), 100)
        bias  = r.get("bias", "LONG")
        v     = r.get("verdict", "")
        vs    = 30 if v.startswith("ENTER") else (20 if ("DIP" in v or "BREAKDOWN" in v) else 10)
        rr    = min(r.get("rr_exit", 0) / 6.0, 1.0) * 30
        trend = r.get("trend_primary", "")
        if bias == "SHORT":
            ts = 20 if trend == "DOWNTREND" else (10 if trend == "MIXED" else 5)
        else:
            ts = 20 if trend == "UPTREND" else (10 if trend == "MIXED" else 5)
        rsi = r.get("rsi") or 50
        if bias == "SHORT":
            rs = 20 if rsi <= 45 else (15 if rsi <= 55 else 5)
        else:
            rs = 20 if 40 <= rsi <= 65 else (15 if 65 < rsi <= 75 else 5)
        return tier + vs + rr + ts + rs

    return sorted(actionable, key=score, reverse=True)[:top_n]


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT FORMATTERS
# ─────────────────────────────────────────────────────────────────────────────

def format_best_opportunities_summary(top10: list, total_actionable: int, total_scanned: int) -> str:
    now  = datetime.utcnow().strftime("%H:%M UTC")
    lines = [
        f"🎯 *BEST OPPORTUNITIES — {now}*",
        f"Ranked by entry quality | {total_actionable} actionable from {total_scanned} scanned",
        f"",
    ]
    trend_tag   = lambda t: "MA↑" if t == "UPTREND" else ("MA→" if t == "MIXED" else "MA↓")
    verdict_tag = lambda v: ("🔴 SHORT" if v.startswith("ENTER SHORT") else
                             ("✅ ENTER"  if v.startswith("ENTER") else
                              ("⏳ BRKDWN" if "BREAKDOWN" in v else
                               ("⏳ DIP"    if "DIP" in v else "⏳ WAIT"))))
    ct_tag  = lambda c: {"A+": "🔥A+", "A": "🔵A", "B": "🟡B"}.get(c, "🟡B")
    bias_tag = lambda b: "📉 SHORT" if b == "SHORT" else "📈 LONG"
    for i, r in enumerate(top10, 1):
        sym    = r.get("ticker", "?")
        trend  = trend_tag(r.get("trend_primary", ""))
        rsi    = r.get("rsi", "-")
        tag    = verdict_tag(r.get("verdict", ""))
        ct     = ct_tag(r.get("catalyst_tier", "B"))
        bias   = bias_tag(r.get("bias", "LONG"))
        day_r  = r.get("day_return")
        rvol_v = r.get("rvol")
        move   = f" {'🟢' if day_r and day_r > 0 else '🔴'}{day_r:+.1f}%" if day_r is not None else ""
        vol    = f" RVOL {rvol_v:.1f}x" if rvol_v is not None else ""
        lines.append(f"{i}. {ct} *{sym}*{move}{vol} | {bias} | {trend} | RSI {rsi} | {tag}")
    lines += ["", "_Detail cards follow_ ↓"]
    return "\n".join(lines)


def format_telegram_card(r, rank):
    ccy      = r.get("currency", "USD")
    sym      = r.get("ticker", "?")
    name     = r.get("name", sym)
    price    = r.get("current_price", 0)
    mid      = (r["entry_low"] + r["entry_high"]) / 2
    entry    = f"{r['entry_low']:.2f} - {r['entry_high']:.2f}"
    verdict  = r.get("verdict", "-")
    bias     = r.get("bias", "LONG")

    # ── Direction-aware stop and target display ────────────────────────────────
    if bias == "SHORT":
        # Stop is ABOVE entry — loss if price rises past it
        stop   = f"{r['stop_loss']:.2f}  (+{r['stop_pct']:.1f}% ▲ above entry)"
        # Targets are BELOW entry — gain when price falls
        t1_pct = (mid - r["target_1"]) / mid * 100
        t2_pct = (mid - r["target_2"]) / mid * 100
        ex_pct = (mid - r["recommended_exit"]) / mid * 100
        t1     = f"{r['target_1']:.2f}  (-{t1_pct:.1f}% ▼)  R/R {r['rr_t1']:.1f}:1"
        t2     = f"{r['target_2']:.2f}  (-{t2_pct:.1f}% ▼)  R/R {r['rr_t2']:.1f}:1"
        ex     = f"{r['recommended_exit']:.2f}  (-{ex_pct:.1f}% ▼)  R/R {r['rr_exit']:.1f}:1"
    else:
        # Stop is BELOW entry — loss if price falls past it
        stop   = f"{r['stop_loss']:.2f}  (-{r['stop_pct']:.1f}%)"
        t1_pct = (r["target_1"] - mid) / mid * 100
        t2_pct = (r["target_2"] - mid) / mid * 100
        ex_pct = (r["recommended_exit"] - mid) / mid * 100
        t1     = f"{r['target_1']:.2f}  (+{t1_pct:.1f}%)  R/R {r['rr_t1']:.1f}:1"
        t2     = f"{r['target_2']:.2f}  (+{t2_pct:.1f}%)  R/R {r['rr_t2']:.1f}:1"
        ex     = f"{r['recommended_exit']:.2f}  (+{ex_pct:.1f}%)  R/R {r['rr_exit']:.1f}:1"

    # ── Direction-aware verdict tag ────────────────────────────────────────────
    if verdict.startswith("ENTER SHORT"):
        tag = "🔴 ENTER SHORT NOW"
    elif verdict.startswith("ENTER"):
        tag = "✅ ENTER NOW"
    elif "BREAKDOWN" in verdict:
        tag = f"⏳ {verdict}"
    elif "DIP" in verdict:
        tag = f"⏳ WAIT — {verdict}"
    else:
        tag = f"⏳ {verdict}"

    # ── Direction-aware momentum label ─────────────────────────────────────────
    mom = r.get("momentum_st", "-")
    if bias == "SHORT":
        if mom == "BEARISH":
            momentum_display = "BEARISH ✅ (short confirmed — enter)"
        elif mom == "BULLISH":
            momentum_display = "BULLISH ⚠️ (counter-trend — wait for turn)"
        else:
            momentum_display = "NEUTRAL  (acceptable — proceed with caution)"
    else:
        if mom == "BULLISH":
            momentum_display = "BULLISH ✅ (enter now)"
        elif mom == "BEARISH":
            momentum_display = "BEARISH ⚠️ (wait for reversal)"
        else:
            momentum_display = "NEUTRAL  (acceptable — watch for confirmation)"

    trend_tag  = {"UPTREND": "MA↑", "MIXED": "MA→", "DOWNTREND": "MA↓"}.get(r.get("trend_primary", ""), "MA→")
    ct         = r.get("catalyst_tier", "B")
    ct_label   = {"A+": "🔥A+", "A": "🔵A", "B": "🟡B"}.get(ct, "🟡B")
    bias_label = "📉 SHORT" if bias == "SHORT" else "📈 LONG"
    catalyst   = r.get("catalyst_note") or "No near-term catalyst"

    # ── Scanner context line (today's move, RVOL, RS vs market) ───────────────
    day_r  = r.get("day_return")
    rvol_v = r.get("rvol")
    rs_v   = r.get("rs_vs_bench")
    ctx    = []
    if day_r is not None:
        ctx.append(f"{'🟢' if day_r > 0 else '🔴'}{day_r:+.1f}% today")
    if rvol_v is not None:
        ctx.append(f"RVOL {rvol_v:.1f}x")
    if rs_v is not None:
        ctx.append(f"RS {rs_v:+.1f}% vs mkt")
    context_line = "  |  ".join(ctx) if ctx else None

    # ── Position sizing ────────────────────────────────────────────────────────
    pos_eur  = r.get("position_size_eur")
    risk_eur = r.get("risk_eur")
    n_shares = r.get("num_shares")
    risk_pct = r.get("risk_pct_used", RISK_PCT * 100)
    sizing_line = (f"Size:     €{pos_eur:,.0f}  ({n_shares:,} units)  Risk: €{risk_eur:,.0f} ({risk_pct}%)"
                   if pos_eur else "Size:     —")

    card_lines = [
        f"*#{rank} {ct_label} {sym}* ({name}) — {bias_label}",
    ]
    if context_line:
        card_lines.append(context_line)
    card_lines += [
        f"{ccy} {price:.2f} | {trend_tag} | RSI {r.get('rsi', '-')}",
        f"",
        f"Entry:    {ccy} {entry}",
        f"Stop:     {ccy} {stop}",
        f"T1:       {ccy} {t1}",
        f"T2:       {ccy} {t2}",
        f"Exit:     {ccy} {ex}",
        f"",
        f"{sizing_line}",
        f"Momentum: {momentum_display}",
        f"Catalyst: {catalyst}",
        f"{tag}",
        f"─────────────────────────────",
    ]
    return "\n".join(card_lines)


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

    print(f"  Running parallel TA analysis (workers=10)...")
    with ThreadPoolExecutor(max_workers=10) as ex:
        fut_map = {ex.submit(run_ta_entry, t, scanner_data.get(t, {})): t for t in tickers}
        for fut in as_completed(fut_map):
            r = fut.result()
            results.append(r)
            ticker = r["ticker"]
            if r["status"] == "ok":
                print(f"  -> {ticker}: OK | {r['verdict'][:50]}")
                actionable.append(r)
            elif r["status"] == "pass":
                print(f"  -> {ticker}: PASS -- {r['pass_reason']}")
            else:
                print(f"  -> {ticker}: ERROR -- {r['error']}")

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

    # Send Telegram — summary list first, then individual cards
    if top10:
        send_telegram(format_best_opportunities_summary(top10, len(actionable), len(tickers)))
        for i, r in enumerate(top10, 1):
            send_telegram(format_telegram_card(r, rank=i))
    else:
        send_telegram(
            f"🎯 *BEST OPPORTUNITIES — {now_str}*\n"
            f"0 actionable setups from {len(tickers)} scanned."
        )

    print(f"\n{'='*60}")
    print(f"Done. {len(actionable)}/{len(tickers)} actionable. Top {len(top10)} sent to Telegram.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
