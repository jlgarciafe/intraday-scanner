"""
send_report.py
────────────────────────────────────────────────────────────
Sends the combined HTML email report:
  - Body    : ta_narrative.md → styled HTML (batch table + flag notes)
              followed by the standard Excel worksheet legend
  - Attachment: scan_report_*.xlsx

Usage:
  python send_report.py [--narrative ta_narrative.md] [--excel scan_report_*.xlsx]
"""

import os
import re
import glob
import argparse
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.base      import MIMEBase
from email.mime.text      import MIMEText
from email                import encoders

import markdown as md_lib


# ─────────────────────────────────────────────────────────────────────────────
# COLOUR PALETTE (matches Excel report + Telegram)
# ─────────────────────────────────────────────────────────────────────────────

NAVY   = "#1F3864"
WHITE  = "#FFFFFF"
AMBER  = "#FFF2CC"
GREEN  = "#E2EFDA"
RED    = "#FCE4D6"
LGRAY  = "#F5F5F5"
DGRAY  = "#D9D9D9"

# Rating badge colours: (text colour, background colour)
RATING_BADGES = {
    # Match emoji variants + plain text
    "🟢 BUY":      ("#FFFFFF", "#00B050", "BUY"),
    "🟡 SPEC BUY": ("#7F6000", "#FFC000", "SPEC BUY"),
    "🟡 SPEC":     ("#7F6000", "#FFC000", "SPEC BUY"),
    "🟠 HOLD":     ("#833C00", "#F4B942", "HOLD"),
    "🔴 EXIT":     ("#FFFFFF", "#C00000", "EXIT"),
}


# ─────────────────────────────────────────────────────────────────────────────
# MARKDOWN → HTML
# ─────────────────────────────────────────────────────────────────────────────

def md_to_html(text: str) -> str:
    """Convert Markdown to HTML with table support."""
    return md_lib.markdown(text, extensions=["tables", "nl2br"])


def inject_rating_badges(html: str) -> str:
    """Replace rating emoji+text with coloured HTML badge spans."""
    for emoji_text, (fg, bg, label) in RATING_BADGES.items():
        badge = (
            f'<span style="display:inline-block;background:{bg};color:{fg};'
            f'font-weight:bold;padding:2px 10px;border-radius:4px;'
            f'font-size:11px;white-space:nowrap;">{label}</span>'
        )
        html = html.replace(emoji_text, badge)
    return html


def style_tables(html: str) -> str:
    """
    Inject inline styles on <table>, <th>, <td> so that email clients
    (Gmail, Outlook) render a clean bordered table without needing external CSS.
    """
    TABLE_STYLE = (
        'style="border-collapse:collapse;width:100%;font-size:12px;'
        'font-family:Arial,sans-serif;margin-bottom:16px;"'
    )
    TH_STYLE = (
        'style="background:#1F3864;color:#FFFFFF;font-weight:bold;'
        'padding:7px 10px;text-align:center;border:1px solid #CCCCCC;'
        'white-space:nowrap;"'
    )
    TD_STYLE = (
        'style="padding:6px 10px;border:1px solid #DDDDDD;'
        'text-align:center;vertical-align:middle;"'
    )

    html = re.sub(r"<table>", f"<table {TABLE_STYLE}>", html)
    html = re.sub(r"<th>",    f"<th {TH_STYLE}>",       html)
    html = re.sub(r"<td>",    f"<td {TD_STYLE}>",        html)

    # Alternate row shading — add light-gray to even <tr> inside <tbody>
    def shade_rows(m):
        rows  = m.group(0).split("<tr>")
        out   = [rows[0]]
        for i, row in enumerate(rows[1:], 1):
            bg = f'style="background:{LGRAY};"' if i % 2 == 0 else ""
            out.append(f'<tr {bg}>' if bg else f"<tr>{row}"[:4 + len(row)])
            if bg:
                out[-1] += row
        return "".join(out)

    html = re.sub(r"<tbody>.*?</tbody>", shade_rows, html, flags=re.DOTALL)
    return html


def style_headings(html: str) -> str:
    """Style H2/H3 section headers."""
    H2 = (
        'style="font-family:Arial,sans-serif;font-size:14px;font-weight:bold;'
        f'color:{NAVY};margin:20px 0 6px 0;border-bottom:2px solid {NAVY};padding-bottom:4px;"'
    )
    H3 = (
        'style="font-family:Arial,sans-serif;font-size:12px;font-weight:bold;'
        f'color:{NAVY};margin:14px 0 4px 0;"'
    )
    html = re.sub(r"<h2>", f"<h2 {H2}>", html)
    html = re.sub(r"<h3>", f"<h3 {H3}>", html)
    return html


def narrative_to_html(narrative_md: str) -> str:
    """Full pipeline: Markdown → HTML → styled."""
    html = md_to_html(narrative_md)
    html = style_tables(html)
    html = style_headings(html)
    html = inject_rating_badges(html)
    return html


# ─────────────────────────────────────────────────────────────────────────────
# EMAIL HTML TEMPLATE
# ─────────────────────────────────────────────────────────────────────────────

LEGEND_HTML = f"""
<table style="border-collapse:collapse;width:100%;font-size:12px;
              font-family:Arial,sans-serif;margin-top:8px;">
  <tr>
    <td style="padding:6px 10px;border:1px solid #DDDDDD;background:{GREEN};">
      📊 <strong>Excel attached — 3 sheets</strong>
    </td>
  </tr>
  <tr style="background:{LGRAY};">
    <td style="padding:6px 10px;border:1px solid #DDDDDD;">
      <strong>Sheet 1 — TOP MOVERS</strong> &nbsp;|&nbsp;
      All qualifying candidates ranked by scanner score
    </td>
  </tr>
  <tr>
    <td style="padding:6px 10px;border:1px solid #DDDDDD;">
      <strong>Sheet 2 — TOP OPPORTUNITIES</strong> &nbsp;|&nbsp;
      Top 10 ranked by entry quality (A+ first)
    </td>
  </tr>
  <tr style="background:{LGRAY};">
    <td style="padding:6px 10px;border:1px solid #DDDDDD;">
      <strong>Sheet 3 — DETAIL CARDS</strong> &nbsp;|&nbsp;
      Full TA: entry zone / stop / T1 / T2 / exit / sizing
    </td>
  </tr>
</table>

<p style="font-family:Arial,sans-serif;font-size:11px;color:#888888;margin-top:12px;">
  Tier legend: 🔥 A+ = highest conviction &nbsp;|&nbsp; 🔵 A = strong &nbsp;|&nbsp; 🟡 B = standard<br>
  Direction: 📈 LONG = buy & hold up &nbsp;|&nbsp; 📉 SHORT = sell & profit on fall<br><br>
  ⚠️ <em>Research only. Not financial advice.</em>
</p>
"""


def build_html_email(narrative_html: str, run_time: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1.0">
</head>
<body style="margin:0;padding:0;background:#F0F0F0;font-family:Arial,sans-serif;">

  <!-- Header bar -->
  <table width="100%" cellpadding="0" cellspacing="0"
         style="background:{NAVY};padding:20px 32px;">
    <tr>
      <td>
        <div style="color:{WHITE};font-size:20px;font-weight:bold;">
          📡 Intraday Scanner
        </div>
        <div style="color:#AABBD4;font-size:12px;margin-top:4px;">
          {run_time}
        </div>
      </td>
    </tr>
  </table>

  <!-- Body card -->
  <table width="100%" cellpadding="0" cellspacing="0"
         style="max-width:900px;margin:24px auto;">
    <tr>
      <td style="background:{WHITE};padding:28px 32px;border-radius:6px;
                 box-shadow:0 1px 4px rgba(0,0,0,0.12);">

        <!-- Claude narrative section -->
        <div style="font-family:Arial,sans-serif;font-size:13px;
                    color:#222222;line-height:1.6;">
          {narrative_html}
        </div>

        <!-- Divider -->
        <hr style="border:none;border-top:1px solid #DDDDDD;margin:24px 0;">

        <!-- Excel legend -->
        {LEGEND_HTML}

      </td>
    </tr>
  </table>

  <!-- Footer -->
  <table width="100%" cellpadding="0" cellspacing="0"
         style="max-width:900px;margin:0 auto 32px;">
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

def find_latest_excel() -> str | None:
    candidates = glob.glob("scan_report_*.xlsx")
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def read_narrative(path: str) -> str:
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return f.read()
    return "*TA narrative not available for this run.*\n"


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Send combined HTML email report")
    parser.add_argument("--narrative", default="ta_narrative.md")
    parser.add_argument("--excel",     default=None)
    args = parser.parse_args()

    email_to   = os.getenv("EMAIL_TO",       "")
    email_from = os.getenv("EMAIL_FROM",     "")
    email_pass = os.getenv("EMAIL_PASSWORD", "")
    smtp_host  = os.getenv("SMTP_HOST",      "smtp.gmail.com")
    smtp_port  = int(os.getenv("SMTP_PORT",  "587"))

    if not email_to or not email_from or not email_pass:
        print("WARN: EMAIL_TO / EMAIL_FROM / EMAIL_PASSWORD not set — skipping email")
        return

    run_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    # Resolve Excel
    xl_path = args.excel or find_latest_excel()
    if not xl_path or not os.path.exists(xl_path):
        print("WARN: No Excel report found — sending without attachment")
        xl_path = None
    else:
        print(f"Excel attachment: {xl_path}")

    # Build HTML body
    narrative_md   = read_narrative(args.narrative)
    narrative_html = narrative_to_html(narrative_md)
    html_body      = build_html_email(narrative_html, run_time)

    # Compose email
    msg            = MIMEMultipart("mixed")
    msg["From"]    = email_from
    msg["To"]      = email_to
    msg["Subject"] = f"Intraday Scanner — {run_time}"

    # HTML part
    alt = MIMEMultipart("alternative")
    alt.attach(MIMEText(html_body, "html", "utf-8"))
    msg.attach(alt)

    # Excel attachment
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
        print(f"ERROR sending email: {e}")
        raise


if __name__ == "__main__":
    main()
