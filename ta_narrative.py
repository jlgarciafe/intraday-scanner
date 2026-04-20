"""
ta_narrative.py
────────────────────────────────────────────────────────────
Calls the Claude API on the top-10 recommendations from orders.json.
Applies the ta-entry batch-skill rules to produce:
  - A ranked table (Rating × Verdict)
  - A short narrative flag section (3–6 sentences)

Output: ta_narrative.md

Usage:
  python ta_narrative.py [--orders orders.json] [--out ta_narrative.md]
"""

import os
import json
import argparse
from datetime import datetime

import anthropic

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT — condensed batch skill rules
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are a disciplined technical analyst applying the ta-entry batch skill.
Your job is to evaluate the provided ranked opportunities and produce:
  1. A compact batch table summarising all positions.
  2. A short flag section (3–6 sentences) covering clusters and the cleanest setup.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BATCH TABLE FORMAT (Markdown)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Produce a Markdown table with these exact columns in order:
| # | Stock | Rating | Price | Entry | Stop | T1 | T2 | Exit | R/R | Momentum | Catalyst | Verdict |

Column specs:
- #        : Rank integer from the input
- Stock    : TICKER — Short Name (e.g. FRO — Frontline)
- Rating   : 🟢 BUY / 🟡 SPEC BUY / 🟠 HOLD / 🔴 EXIT (see rubric below)
- Price    : Current price with currency symbol
- Entry    : Entry zone low–high (e.g. 14.50–15.20) — show "—" if PASS/EXIT
- Stop     : Stop loss level — show "—" if PASS/EXIT
- T1       : Target 1 price — show "—" if PASS/EXIT
- T2       : Target 2 price — show "—" if PASS/EXIT
- Exit     : Recommended exit price — show "—" if PASS/EXIT
- R/R      : rr_exit to one decimal (e.g. 3.2:1) — show "—" if PASS/EXIT
- Momentum : "Strong ↑" / "↑ recovering" / "Neutral" / "↓ weakening" / "Strong ↑ OB" (overbought)
- Catalyst : Catalyst note or "None near-term"
- Verdict  : ENTER NOW / WAIT FOR DIP $X / WAIT FOR BREAKOUT $X / PASS

RATING RUBRIC:
🟢 BUY    — Strong setup. Trend aligned with bias, R/R ≥ 2:1, momentum confirms.
             Full-size candidate.
🟡 SPEC   — Valid setup but elevated risk: binary catalyst within 14 days,
             price extended, thin liquidity, or SHORT in uptrend.
             Half-size / wider stop.
🟠 HOLD   — No compelling entry now. Thesis not broken but no trigger either.
             Do not initiate; do not exit if already held.
🔴 EXIT   — Broken structure, extreme R/R failure, or structural decay instrument
             (leveraged inverse ETFs: SQQQ, SPXS, etc. are always 🔴 EXIT for new positions).

NOTES:
- Leveraged inverse ETFs (SQQQ, SPXS, SDS, UVXY, etc.) → always 🔴 EXIT regardless of momentum.
  State the reason clearly in the flag section.
- SHORT bias setups in a primary UPTREND → 🟡 SPEC minimum; flag as counter-trend.
- Earnings within 7 days → 🟡 SPEC minimum; flag with ⚠️ in Catalyst column.
- If two ratings are equally valid, prefer the more cautious one.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FLAG NOTES (after the table)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

After the table, write a **Flag Notes** section:

1. **Cleanest setup** — name the single best-risk/reward candidate and one sentence why.
2. **Catalyst cluster** — if ≥ 2 positions have earnings/events within 14 days, name them together.
3. **Counter-trend warning** — if any SHORT positions are in a primary UPTREND, name them.
4. **Structural decay** — if any leveraged inverse ETFs are in the list, explain why they are always 🔴.
5. **Sizing note** — one sentence on position sizing discipline (half-size for SPEC, standard for BUY).

Omit any section that does not apply. Keep the entire flag section to 6 sentences maximum.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Use ONLY data provided in the JSON input. Do not fabricate prices or levels.
- Every level in the table must match the input data exactly.
- Do NOT produce the full five-numbers framework or Word document — batch table + flag notes ONLY.
- Respond in plain Markdown. No preamble, no sign-off.
"""

# ─────────────────────────────────────────────────────────────────────────────
# USER MESSAGE BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_user_message(orders: list, run_time: str) -> str:
    lines = [
        f"Intraday scanner run: {run_time}",
        f"Top {len(orders)} ranked opportunities — apply ta-entry batch skill:",
        "",
        "```json",
        json.dumps(orders, indent=2),
        "```",
        "",
        "Produce the batch table and flag notes as specified.",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TA Narrative Generator")
    parser.add_argument("--orders", default="orders.json",
                        help="Path to orders.json (output of ta_runner.py)")
    parser.add_argument("--out",    default="ta_narrative.md",
                        help="Output Markdown file path")
    args = parser.parse_args()

    # Load orders
    if not os.path.exists(args.orders):
        print(f"ERROR: {args.orders} not found — ta_runner.py must run first")
        raise SystemExit(1)

    with open(args.orders) as f:
        orders = json.load(f)

    if not orders:
        print("No orders found — writing empty narrative")
        with open(args.out, "w") as f:
            f.write("*No actionable setups this run.*\n")
        return

    run_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    # Call Claude API
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("WARN: ANTHROPIC_API_KEY not set — writing placeholder narrative")
        with open(args.out, "w") as f:
            f.write(
                f"*TA narrative unavailable (ANTHROPIC_API_KEY not set).*\n\n"
                f"Run: {run_time}  |  {len(orders)} ranked setup(s)\n"
            )
        return

    print(f"Calling Claude API for {len(orders)} setup(s)...")
    client  = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": build_user_message(orders, run_time)}
        ],
    )

    narrative = message.content[0].text

    # Prepend a run header
    header = (
        f"## TA Narrative — {run_time}\n"
        f"_{len(orders)} ranked setup(s) | Generated by Claude API_\n\n"
    )
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(header + narrative + "\n")

    print(f"ta_narrative.md written ({len(narrative)} chars)")


if __name__ == "__main__":
    main()
