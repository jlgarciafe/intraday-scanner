"""
send_report.py
────────────────────────────────────────────────────────────
Sends the combined email report:
  - Body    : ta_narrative.md content (Claude API batch table + flag notes)
              followed by the standard Excel worksheet summary
  - Attachment: scan_report_*.xlsx

This script runs AFTER ta_runner.py (--skip-email) and ta_narrative.py.

Usage:
  python send_report.py [--narrative ta_narrative.md] [--excel scan_report_*.xlsx]

If --excel is not provided, the script auto-detects the latest scan_report_*.xlsx
in the current directory.
"""

import os
import glob
import argparse
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.base      import MIMEBase
from email.mime.text      import MIMEText
from email                import encoders


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def find_latest_excel() -> str | None:
    """Return the most-recently-modified scan_report_*.xlsx, or None."""
    candidates = glob.glob("scan_report_*.xlsx")
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def read_narrative(path: str) -> str:
    """Read ta_narrative.md; return placeholder if missing."""
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return f.read()
    return "*TA narrative not available for this run.*\n"


def build_email_body(narrative: str, run_time: str) -> str:
    """
    Combines the Claude narrative with the standard Excel worksheet legend.
    Plain-text version (the narrative is Markdown, which reads fine as plain text).
    """
    return (
        f"Intraday Scanner — {run_time}\n"
        f"{'='*60}\n\n"
        f"{narrative}\n\n"
        f"{'='*60}\n"
        f"Excel report attached (3 sheets):\n"
        f"  1. TOP MOVERS         — all qualifying candidates ranked by score\n"
        f"  2. TOP OPPORTUNITIES  — top 10 ranked by entry quality (A+ first)\n"
        f"  3. DETAIL CARDS       — full TA: entry zone / stop / T1 / T2 / exit / sizing\n\n"
        f"Tier legend:  🔥 A+ = highest conviction  |  🔵 A = strong  |  🟡 B = standard\n"
        f"Direction:    📈 LONG = buy & hold up  |  📉 SHORT = sell & profit on fall\n\n"
        f"⚠️  Research only. Not financial advice.\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Send combined email report")
    parser.add_argument("--narrative", default="ta_narrative.md",
                        help="Path to ta_narrative.md")
    parser.add_argument("--excel",     default=None,
                        help="Path to Excel attachment (auto-detects if omitted)")
    args = parser.parse_args()

    # ── Env vars ──────────────────────────────────────────────────────────────
    email_to   = os.getenv("EMAIL_TO",       "")
    email_from = os.getenv("EMAIL_FROM",     "")
    email_pass = os.getenv("EMAIL_PASSWORD", "")
    smtp_host  = os.getenv("SMTP_HOST",      "smtp.gmail.com")
    smtp_port  = int(os.getenv("SMTP_PORT",  "587"))

    if not email_to or not email_from or not email_pass:
        print("WARN: EMAIL_TO / EMAIL_FROM / EMAIL_PASSWORD not set — skipping email")
        return

    run_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    # ── Resolve Excel path ────────────────────────────────────────────────────
    xl_path = args.excel or find_latest_excel()
    if not xl_path or not os.path.exists(xl_path):
        print("WARN: No Excel report found — sending email without attachment")
        xl_path = None
    else:
        print(f"Excel attachment: {xl_path}")

    # ── Build email ───────────────────────────────────────────────────────────
    narrative  = read_narrative(args.narrative)
    body       = build_email_body(narrative, run_time)

    msg            = MIMEMultipart()
    msg["From"]    = email_from
    msg["To"]      = email_to
    msg["Subject"] = f"Intraday Scanner Report — {run_time}"
    msg.attach(MIMEText(body, "plain", "utf-8"))

    if xl_path:
        with open(xl_path, "rb") as fh:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(fh.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition",
                        f'attachment; filename="{os.path.basename(xl_path)}"')
        msg.attach(part)

    # ── Send ──────────────────────────────────────────────────────────────────
    try:
        srv = smtplib.SMTP(smtp_host, smtp_port)
        srv.ehlo()
        srv.starttls()
        srv.login(email_from, email_pass)
        srv.sendmail(email_from, email_to, msg.as_string())
        srv.quit()
        print(f"Email sent → {email_to}")
    except Exception as e:
        print(f"ERROR sending email: {e}")
        raise


if __name__ == "__main__":
    main()
