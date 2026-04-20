"""
send_report.py
────────────────────────────────────────────────────────────
Builds a styled HTML email and sends it.

Table is built directly from orders.json + ta_ratings.json so every
cell is under our control — no Markdown-to-HTML conversion.

Layout:
  1. Ranked opportunity table (grouped: BUY / SPEC BUY / HOLD / EXIT)
  2. Flag notes (from ta_narrative.md)
  3. Excel worksheet legend

Usage:
  python send_report.py [--orders orders.json]
                        [--ratings ta_ratings.json]
                        [--narrative ta_narrative.md]
                        [--excel scan_report_*.xlsx]
"""

import os
import re
import glob
import json
import argparse
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.base      import MIMEBase
from email.mime.text      import MIMEText
from email                import encoders


# ─────────────────────────────────────────────────────────────────────────────
# COLOUR PALETTE
# ─────────────────────────────────────────────────────────────────────────────

NAVY    = "#1F3864"
WHITE   = "#FFFFFF"
LGRAY   = "#F5F5F5"
DGRAY   = "#D0D0D0"
BUY_BG  = "#EAF5EA"   # light green  — BUY rows
SPEC_BG = "#FFF8E1"   # light amber  — SPEC rows
HOLD_BG = "#F7F7F7"   # light grey   — HOLD rows
EXIT_BG = "#FFF0F0"   # light red    — EXIT rows

GROUP_HEADERS = {
    "BUY":      ("BUY — BY R/R, STRENGTH AS TIEBREAKER",       "#D4EDDA", "#155724"),
    "SPEC BUY": ("SPEC BUY — BY R/R, STRENGTH AS TIEBREAKER",  "#FFF3CD", "#7F6000"),
    "HOLD":     ("HOLD",                                         "#E2E3E5", "#383D41"),
    "EXIT":     ("EXIT — STRUCTURAL DECAY PRODUCTS",            "#F8D7DA", "#721C24"),
}

ROW_BG = {
    "BUY":      BUY_BG,
    "SPEC BUY": SPEC_BG,
    "HOLD":     HOLD_BG,
    "EXIT":     EXIT_BG,
}

BADGE_STYLE = {
    "BUY":      ("background:#00B050;color:#FFFFFF;"),
    "SPEC BUY": ("background:#FFC000;color:#7F6000;"),
    "HOLD":     ("background:#AAAAAA;color:#FFFFFF;"),
    "EXIT":     ("background:#C00000;color:#FFFFFF;"),
}


# ─────────────────────────────────────────────────────────────────────────────
# TABLE BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def fmt_price(val, ccy=""):
    """Format a price value nicely."""
    if val is None:
        return "—"
    try:
        v = float(val)
        if v >= 10000:
            return f"{ccy}{v:,.0f}"
        if v >= 100:
            return f"{ccy}{v:,.1f}"
        return f"{ccy}{v:,.2f}"
    except (TypeError, ValueError):
        return "—"


def short_verdict(v: str) -> str:
    """Shorten full verdict string to ≤12-char table label."""
    if not v:
        return "—"
    if v.startswith("ENTER SHORT"):
        return "Short Now"
    if v.startswith("ENTER NOW"):
        return "Enter Now"
    m = re.search(r"DIP TO ([\d,. ]+)", v)
    if m:
        num = m.group(1).strip().replace(",", "").rstrip("0").rstrip(".")
        return f"Dip {num}"
    m = re.search(r"BREAKDOWN BELOW ([\d,. ]+)", v)
    if m:
        num = m.group(1).strip().replace(",", "").rstrip("0").rstrip(".")
        return f"BrkDwn {num}"
    m = re.search(r"BREAKOUT ABOVE ([\d,. ]+)", v)
    if m:
        num = m.group(1).strip().replace(",", "").rstrip("0").rstrip(".")
        return f"Brkout {num}"
    if "PASS" in v.upper():
        return "Pass"
    return v[:12]


def ccy_symbol(ccy: str) -> str:
    return {
        "USD": "$", "EUR": "€", "GBP": "£",
        "JPY": "¥", "KRW": "₩", "HKD": "HK$",
        "CAD": "C$", "AUD": "A$", "NOK": "NOK ",
        "SEK": "SEK ", "DKK": "DKK ", "CHF": "CHF ",
        "INR": "₹", "CNY": "¥", "SGD": "S$",
    }.get(ccy, f"{ccy} ")


def build_table_html(orders: list, ratings: dict) -> str:
    """Build the full opportunities table as an HTML string."""

    TH = (
        'style="background:{navy};color:{white};font-weight:bold;'
        'padding:8px 10px;text-align:center;border:1px solid #3A5080;'
        'white-space:nowrap;font-size:11px;"'
    ).format(navy=NAVY, white=WHITE)

    TD_BASE = (
        'padding:7px 9px;border:1px solid {border};'
        'text-align:center;vertical-align:middle;font-size:12px;'
        'font-family:Arial,sans-serif;'
    )

    def td(content, bg, extra="", align="center"):
        border = "#CCCCCC"
        return (
            f'<td style="{TD_BASE.format(border=border)}'
            f'background:{bg};text-align:{align};{extra}">'
            f'{content}</td>'
        )

    def badge(rating):
        style = BADGE_STYLE.get(rating, "background:#AAAAAA;color:#FFFFFF;")
        short = "Spec" if rating == "SPEC BUY" else rating.title()
        return (
            f'<span style="{style}font-weight:bold;padding:3px 9px;'
            f'border-radius:4px;font-size:11px;white-space:nowrap;">'
            f'{short}</span>'
        )

    # Group rows by rating
    ORDER = ["BUY", "SPEC BUY", "HOLD", "EXIT"]
    groups: dict[str, list] = {g: [] for g in ORDER}

    for o in orders:
        ticker   = o.get("ticker", "")
        r_data   = ratings.get(ticker, {})
        rating   = r_data.get("rating", "HOLD")
        if rating not in groups:
            rating = "HOLD"
        groups[rating].append((o, r_data))

    # Column headers
    cols = ["#", "TICKER", "RATING", "STR",
            "NOW", "ENTRY", "STOP", "T1", "T2", "EXIT", "R/R",
            "CATALYST", "VERDICT"]
    header_row = "".join(f"<th {TH}>{c}</th>" for c in cols)

    rows_html = []

    for group_key in ORDER:
        items = groups[group_key]
        if not items:
            continue

        # Group separator row
        gh_text, gh_bg, gh_fg = GROUP_HEADERS[group_key]
        rows_html.append(
            f'<tr><td colspan="{len(cols)}" style="'
            f'background:{gh_bg};color:{gh_fg};font-weight:bold;'
            f'font-size:11px;padding:5px 10px;border:1px solid #CCCCCC;'
            f'text-align:left;letter-spacing:0.4px;">'
            f'{gh_text}</td></tr>'
        )

        for o, r_data in items:
            ticker   = o.get("ticker", "")
            name     = o.get("name", ticker)
            # Truncate long names
            name_short = (name[:22] + "…") if len(name) > 24 else name
            rank     = o.get("rank", "")
            rating   = r_data.get("rating", "HOLD")
            strength = r_data.get("strength", "—")
            ccy      = o.get("currency", "USD")
            sym      = ccy_symbol(ccy)
            bg       = ROW_BG.get(rating, HOLD_BG)

            price    = fmt_price(o.get("current_price"), sym)
            entry_lo = fmt_price(o.get("entry_limit_low"))
            entry_hi = fmt_price(o.get("entry_limit_high"))
            entry    = f"{entry_lo}–{entry_hi}" if entry_lo != "—" else "—"
            stop     = fmt_price(o.get("stop_loss")) if rating not in ("HOLD","EXIT") else "—"
            t1       = fmt_price(o.get("target_1"))  if rating not in ("HOLD","EXIT") else "—"
            t2       = fmt_price(o.get("target_2"))  if rating not in ("HOLD","EXIT") else "—"
            ex       = fmt_price(o.get("recommended_exit")) if rating not in ("HOLD","EXIT") else "—"
            rr_val   = o.get("rr_exit")
            rr       = f"{rr_val:.1f}×" if rr_val and rating not in ("HOLD","EXIT") else "—"
            catalyst = o.get("catalyst") or "None"
            # HOLD/EXIT rows: override verdict to match Claude's rating
            if rating == "HOLD":
                verdict = "Hold"
            elif rating == "EXIT":
                verdict = "Pass · decay"
            else:
                verdict = short_verdict(o.get("verdict", ""))

            # Ticker cell: bold ticker + small name below
            ticker_cell = (
                f'<strong style="font-size:13px;">{ticker}</strong>'
                f'<br><span style="font-size:10px;color:#666666;">{name_short}</span>'
            )

            row = (
                f"<tr>"
                + td(rank,         bg)
                + td(ticker_cell,  bg, align="left")
                + td(badge(rating),bg)
                + td(strength,     bg, "font-size:14px;font-weight:bold;")
                + td(price,        bg, "font-weight:bold;")
                + td(entry,        bg)
                + td(stop,         bg)
                + td(t1,           bg)
                + td(t2,           bg)
                + td(ex,           bg)
                + td(rr,           bg, "font-weight:bold;")
                + td(catalyst,     bg, "font-size:11px;")
                + td(verdict,      bg, "font-size:11px;font-weight:bold;")
                + "</tr>"
            )
            rows_html.append(row)

    table = (
        f'<table style="border-collapse:collapse;width:100%;'
        f'font-family:Arial,sans-serif;">'
        f"<thead><tr>{header_row}</tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody>"
        f"</table>"
    )
    return table


# ─────────────────────────────────────────────────────────────────────────────
# EMAIL TEMPLATE
# ─────────────────────────────────────────────────────────────────────────────

def build_html_email(table_html: str, flag_notes: str, run_time: str) -> str:
    notes_html = ""
    if flag_notes.strip() and "unavailable" not in flag_notes:
        # Wrap paragraphs
        paras = [
            f'<p style="margin:6px 0;font-size:13px;color:#333333;">{p.strip()}</p>'
            for p in flag_notes.strip().split("\n") if p.strip()
        ]
        notes_html = (
            f'<h3 style="font-family:Arial,sans-serif;font-size:13px;'
            f'font-weight:bold;color:{NAVY};margin:20px 0 8px 0;'
            f'border-bottom:1px solid {DGRAY};padding-bottom:4px;">Flag Notes</h3>'
            + "".join(paras)
        )

    legend = f"""
<table style="border-collapse:collapse;width:100%;font-size:12px;
              font-family:Arial,sans-serif;margin-top:8px;">
  <tr><td colspan="2" style="background:#E8F5E9;padding:7px 10px;
      border:1px solid #CCCCCC;font-weight:bold;">
    📊 Excel attached — 3 sheets
  </td></tr>
  <tr><td style="padding:6px 10px;border:1px solid #DDDDDD;background:{LGRAY};width:38%;">
    <strong>Sheet 1 — TOP MOVERS</strong></td>
    <td style="padding:6px 10px;border:1px solid #DDDDDD;background:{LGRAY};">
    All qualifying candidates ranked by scanner score</td></tr>
  <tr><td style="padding:6px 10px;border:1px solid #DDDDDD;">
    <strong>Sheet 2 — TOP OPPORTUNITIES</strong></td>
    <td style="padding:6px 10px;border:1px solid #DDDDDD;">
    Top 10 ranked by entry quality (A+ first)</td></tr>
  <tr><td style="padding:6px 10px;border:1px solid #DDDDDD;background:{LGRAY};">
    <strong>Sheet 3 — DETAIL CARDS</strong></td>
    <td style="padding:6px 10px;border:1px solid #DDDDDD;background:{LGRAY};">
    Full TA: entry zone / stop / T1 / T2 / exit / sizing</td></tr>
</table>
<p style="font-family:Arial,sans-serif;font-size:11px;color:#888888;margin-top:10px;">
  Tier: 🔥 A+ highest conviction &nbsp;|&nbsp; 🔵 A strong &nbsp;|&nbsp; 🟡 B standard<br>
  Direction: 📈 LONG &nbsp;|&nbsp; 📉 SHORT<br><br>
  ⚠️ <em>Research only. Not financial advice.</em>
</p>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0"></head>
<body style="margin:0;padding:0;background:#EFEFEF;font-family:Arial,sans-serif;">

  <table width="100%" cellpadding="0" cellspacing="0"
         style="background:{NAVY};padding:18px 32px;">
    <tr>
      <td>
        <div style="color:{WHITE};font-size:20px;font-weight:bold;">📡 Intraday Scanner</div>
        <div style="color:#A8BFDA;font-size:12px;margin-top:3px;">{run_time}</div>
      </td>
    </tr>
  </table>

  <table width="100%" cellpadding="0" cellspacing="0" style="max-width:960px;margin:20px auto;">
    <tr>
      <td style="background:{WHITE};padding:24px 28px;border-radius:6px;
                 box-shadow:0 1px 4px rgba(0,0,0,0.10);">

        <h3 style="font-family:Arial,sans-serif;font-size:13px;font-weight:bold;
                   color:{NAVY};margin:0 0 12px 0;
                   border-bottom:1px solid {DGRAY};padding-bottom:4px;">
          Top Opportunities — Ranked by R/R &amp; Conviction
        </h3>

        {table_html}

        {notes_html}

        <hr style="border:none;border-top:1px solid {DGRAY};margin:22px 0;">

        {legend}

      </td>
    </tr>
  </table>

  <table width="100%" cellpadding="0" cellspacing="0"
         style="max-width:960px;margin:0 auto 28px;">
    <tr>
      <td style="text-align:center;font-size:10px;color:#AAAAAA;padding:8px;">
        JLG Hunt Bot &nbsp;·&nbsp; intraday-scanner &nbsp;·&nbsp; {run_time}
      </td>
    </tr>
  </table>

</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def find_latest_excel():
    candidates = glob.glob("scan_report_*.xlsx")
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--orders",    default="orders.json")
    parser.add_argument("--ratings",   default="ta_ratings.json")
    parser.add_argument("--narrative", default="ta_narrative.md")
    parser.add_argument("--excel",     default=None)
    args = parser.parse_args()

    email_to   = os.getenv("EMAIL_TO",       "")
    email_from = os.getenv("EMAIL_FROM",     "")
    email_pass = os.getenv("EMAIL_PASSWORD", "")
    smtp_host  = os.getenv("SMTP_HOST",      "smtp.gmail.com")
    smtp_port  = int(os.getenv("SMTP_PORT",  "587"))

    if not email_to or not email_from or not email_pass:
        print("WARN: EMAIL credentials not set — skipping email")
        return

    run_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    # Load data
    orders  = json.load(open(args.orders))  if os.path.exists(args.orders)  else []
    ratings = json.load(open(args.ratings)) if os.path.exists(args.ratings) else {}
    flag_notes = open(args.narrative, encoding="utf-8").read() if os.path.exists(args.narrative) else ""

    # Resolve Excel attachment
    xl_path = args.excel or find_latest_excel()
    if not xl_path or not os.path.exists(xl_path):
        print("WARN: No Excel found — sending without attachment")
        xl_path = None
    else:
        print(f"Attaching: {xl_path}")

    # Build HTML
    table_html = build_table_html(orders, ratings)
    html_body  = build_html_email(table_html, flag_notes, run_time)

    # Compose
    msg            = MIMEMultipart("mixed")
    msg["From"]    = email_from
    msg["To"]      = email_to
    msg["Subject"] = f"Intraday Scanner — {run_time}"

    alt = MIMEMultipart("alternative")
    alt.attach(MIMEText(html_body, "html", "utf-8"))
    msg.attach(alt)

    if xl_path:
        with open(xl_path, "rb") as fh:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(fh.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition",
                        f'attachment; filename="{os.path.basename(xl_path)}"')
        msg.attach(part)

    # Send
    try:
        srv = smtplib.SMTP(smtp_host, smtp_port)
        srv.ehlo()
        srv.starttls()
        srv.login(email_from, email_pass)
        srv.sendmail(email_from, email_to, msg.as_string())
        srv.quit()
        print(f"HTML email sent → {email_to}")
    except Exception as e:
        print(f"ERROR: {e}")
        raise


if __name__ == "__main__":
    main()
