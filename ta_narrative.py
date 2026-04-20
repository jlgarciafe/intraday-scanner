"""
ta_narrative.py
────────────────────────────────────────────────────────────
Calls the Claude API on the top-10 recommendations from orders.json.
Produces:
  - ta_ratings.json  : per-ticker rating + strength (machine-readable)
  - ta_narrative.md  : flag notes only (appended to email body)

Usage:
  python ta_narrative.py [--orders orders.json]
"""

import os
import json
import argparse
from datetime import datetime

import anthropic

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are a disciplined technical analyst applying the ta-entry batch skill.

Given a list of ranked opportunities, return ONLY valid JSON — no prose before or after.

Schema:
{
  "ratings": [
    {
      "ticker":   "IMVT",
      "rating":   "BUY",
      "strength": "↑↑"
    }
  ],
  "flag_notes": "Markdown text of flag notes (3–6 sentences max)."
}

RATING VALUES (pick exactly one per ticker):
  BUY      — Strong setup. Trend aligned with bias, R/R ≥ 2:1, momentum confirms. Full-size.
  SPEC BUY — Valid but elevated risk: binary catalyst ≤14 days, price extended,
             counter-trend SHORT in uptrend, or thin liquidity. Half-size.
  HOLD     — No compelling entry now. Thesis intact but no trigger. Do not initiate.
  EXIT     — Broken structure, insufficient R/R, or structural decay instrument.

STRENGTH VALUES (pick exactly one):
  ↑↑  strong momentum aligned with trade direction
  ↑   recovering / acceptable
  —   neutral
  ↓   weakening or counter-trend

HARD RULES:
  - Leveraged inverse ETFs (SQQQ, SPXS, SDS, UVXY, etc.) → always EXIT.
  - Earnings within 7 days → SPEC BUY minimum.
  - SHORT in primary UPTREND → SPEC BUY minimum.
  - The ratings array must contain exactly one entry per ticker in the input.
  - Return valid JSON only. No markdown fences, no commentary outside the JSON.

FLAG NOTES (the "flag_notes" field):
  Write 3–6 sentences covering:
  1. Cleanest setup: name the single best candidate and one sentence why.
  2. Catalyst cluster: if ≥2 positions have events within 14 days, name them.
  3. Counter-trend warning: name any SHORT in a primary UPTREND.
  4. Structural decay: explain EXIT call for any inverse ETFs.
  5. Sizing note: half-size for SPEC BUY, standard for BUY.
  Omit any point that does not apply.
"""

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--orders", default="orders.json")
    parser.add_argument("--out-ratings",  default="ta_ratings.json")
    parser.add_argument("--out-narrative", default="ta_narrative.md")
    args = parser.parse_args()

    if not os.path.exists(args.orders):
        print(f"ERROR: {args.orders} not found")
        raise SystemExit(1)

    with open(args.orders) as f:
        orders = json.load(f)

    run_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    if not orders:
        _write_empty(args.out_ratings, args.out_narrative, run_time)
        return

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("WARN: ANTHROPIC_API_KEY not set — writing placeholder")
        _write_empty(args.out_ratings, args.out_narrative, run_time)
        return

    user_msg = (
        f"Scanner run: {run_time}\n"
        f"Rank these {len(orders)} setups and return the JSON:\n\n"
        f"```json\n{json.dumps(orders, indent=2)}\n```"
    )

    print(f"Calling Claude API for {len(orders)} setup(s)...")
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )

    raw = message.content[0].text.strip()

    # Strip markdown fences if Claude wrapped the JSON anyway
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:])
    if raw.endswith("```"):
        raw = raw.rsplit("```", 1)[0]
    raw = raw.strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"ERROR: Claude returned invalid JSON: {e}\nRaw:\n{raw[:500]}")
        _write_empty(args.out_ratings, args.out_narrative, run_time)
        return

    ratings = {r["ticker"]: r for r in data.get("ratings", [])}
    flag_notes = data.get("flag_notes", "")

    # Save ratings JSON
    with open(args.out_ratings, "w", encoding="utf-8") as f:
        json.dump(ratings, f, indent=2, ensure_ascii=False)
    print(f"{args.out_ratings} written ({len(ratings)} tickers)")

    # Save narrative (flag notes only)
    with open(args.out_narrative, "w", encoding="utf-8") as f:
        f.write(flag_notes + "\n")
    print(f"{args.out_narrative} written")


def _write_empty(ratings_path, narrative_path, run_time):
    with open(ratings_path, "w") as f:
        json.dump({}, f)
    with open(narrative_path, "w") as f:
        f.write(f"*TA narrative unavailable — {run_time}*\n")


if __name__ == "__main__":
    main()
