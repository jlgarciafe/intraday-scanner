"""
═══════════════════════════════════════════════════════════════════════════════
  INTRADAY MOMENTUM SCANNER v6 — Dynamic Universe + ETF/Futures Tiers + 11 Markets

  Architecture:
    Stage 1 — Fetch full index constituents dynamically (Wikipedia/free APIs)
              ETF and Futures universes are hardcoded (stable, small sets)
    Stage 2 — Lightweight volume pre-screen across all constituents
              Per-tier thresholds: ETF RVOL≥1.2/ATR≥0.8%, Future RVOL≥1.1/ATR≥0.5%,
              Stock RVOL≥1.2/ATR≥2.5% (plus volume floor)
    Stage 3 — Full ATR + momentum analysis on pre-screened candidates
    Stage 4 — Filter (tier-aware), rank, alert

  Tiers in scan_results.json:
    etf     — ~30 global index + sector ETFs (SPY, QQQ, IWM, XLF, SMH, …)
    future  — ~15 index/commodity futures (ES=F, NQ=F, CL=F, GC=F, …)
    stock   — per-market index constituents (unchanged from v4)

  Markets covered (stock tier):
    US  — S&P 500 (~503) + NASDAQ supplement (~162) + Russell 2000 curated (~192)
    UK  — FTSE 100 (100 stocks) via Wikipedia
    DE  — DAX 40 (40 stocks) via Wikipedia
    JP  — Nikkei 225 (225 stocks) via Wikipedia
    ES  — Ibex 35 (35 stocks) via Wikipedia
    FR  — CAC 40 (40 stocks) via Wikipedia
    HK  — Hang Seng (82 stocks) via Wikipedia
    IN  — Nifty 50 (50 stocks, .NS suffix) via Wikipedia
    AU  — S&P/ASX 200 (200 stocks, .AX suffix) via Wikipedia
    CA  — S&P/TSX 60 (60 stocks, .TO suffix) via Wikipedia
    KR  — KOSPI 200 (200 stocks, .KS suffix) via Wikipedia

  ⚠️  RISK DISCLAIMER:
  Consistently achieving >5% net daily returns is statistically rare and
  highly speculative. This scanner identifies momentum candidates only.
  Use exclusively for research. Never risk capital you cannot afford to lose.
═══════════════════════════════════════════════════════════════════════════════
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
from concurrent.futures import ThreadPoolExecutor, as_completed

from datetime import datetime, timezone, timedelta
from calendar import monthcalendar, FRIDAY
from typing import Optional
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
MARKET_CONTEXT     = os.getenv("MARKET_CONTEXT", "all").lower()
MIN_MOVE_PCT       = float(os.getenv("MIN_MOVE_PCT", "5.0"))
DRY_RUN            = os.getenv("DRY_RUN", "false").lower() == "true"
OUTPUT_JSON        = os.getenv("OUTPUT_JSON", "scan_results.json")
HISTORY_FILE       = os.getenv("HISTORY_FILE", "scan_history.json")

# ── Pre-screen thresholds (stock tier defaults) ───────────────────────────────
PRESCREEN_RVOL     = 1.2    # Minimum RVOL to pass pre-screen (lowered from 1.5)
PRESCREEN_ATR      = 2.5    # ATR% floor — pass if ATR >= this even when RVOL < 1.2x
                            # Gate logic: vol >= min AND (RVOL >= 1.2x OR ATR >= 2.5%)
PRESCREEN_MIN_VOL  = {      # Minimum absolute volume per market (stock tier)
    "us": 500_000,
    "uk": 100_000,
    "de": 50_000,
    "jp": 100_000,
    "es": 50_000,
    "hk": 100_000,
    "fr": 50_000,
    "in": 100_000,
    "au": 50_000,
    "ca": 100_000,
    "kr": 100_000,
}
MAX_PRESCREEN_PASS = 60     # Max stocks to run full analysis on per market

# ── Parallelism ───────────────────────────────────────────────────────────────
PRESCREEN_BATCH_SIZE  = 100  # tickers per yf.download() batch call
MAX_WORKERS_PRESCREEN = 5    # parallel batch workers for pre-screen
MAX_WORKERS_ANALYSIS  = 15   # parallel workers for full analysis

# ── Tier configuration ────────────────────────────────────────────────────────
# Each tier has its own pre-screen and filter thresholds.
# Stock tier re-uses the globals above for 100% backward compatibility.
TIER_CONFIG = {
    "etf": {
        "prescreen_rvol": 1.2,
        "prescreen_atr":  0.8,
        "filter_rvol":    1.2,
        "filter_atr":     0.8,
        "filter_score":   20,
        "min_vol":        500_000,
        "max_pass":       30,
        "fee_key":        "us",
        "skip_vol_floor": False,
    },
    "future": {
        "prescreen_rvol": 1.1,
        "prescreen_atr":  0.5,
        "filter_rvol":    1.1,
        "filter_atr":     0.5,
        "filter_score":   15,
        "min_vol":        500,      # contracts, not shares
        "max_pass":       15,
        "fee_key":        "futures",
        "skip_vol_floor": True,     # don't reject futures on share-volume floors
    },
    "stock": {
        "prescreen_rvol": PRESCREEN_RVOL,
        "prescreen_atr":  PRESCREEN_ATR,
        "filter_rvol":    1.3,
        "filter_atr":     PRESCREEN_ATR,
        "filter_score":   35,
        "min_vol":        None,     # resolved per-market via PRESCREEN_MIN_VOL
        "max_pass":       MAX_PRESCREEN_PASS,
        "fee_key":        None,     # resolved per-market via FEE_MODEL
        "skip_vol_floor": False,
    },
}

# ── Hardcoded ETF universe ────────────────────────────────────────────────────
ETF_UNIVERSE = [
    # Broad US index
    "SPY", "QQQ", "IWM", "DIA", "VOO", "VTI", "MDY",
    # International index
    "EFA", "EWG", "EWU", "EWQ", "EWP", "EWJ", "EWH", "FXI", "EEM",
    # US sector
    "XLF", "XLE", "XLK", "XLV", "XLI", "XLU", "XLP", "XLB", "XLRE", "XLC",
    # Tech / semis
    "SMH", "SOXX", "ARKK",
    # Volatility / inverse
    "UVXY", "SQQQ", "SPXS",
]

# ── Hardcoded Futures universe ────────────────────────────────────────────────
FUTURES_UNIVERSE = [
    # Equity index futures
    "ES=F", "NQ=F", "RTY=F", "YM=F",
    # Bond futures
    "ZN=F", "ZB=F",
    # Commodities
    "CL=F", "NG=F", "GC=F", "SI=F", "HG=F", "PL=F", "PA=F",
    # Currency / vol
    "6E=F", "6J=F", "VX=F",
    # Crypto futures (CME)
    "BTC=F",
]

# ── Fee model ─────────────────────────────────────────────────────────────────
FEE_MODEL = {
    "us":      {"commission_pct": 0.005, "spread_pct": 0.05,  "slippage_pct": 0.05},
    "uk":      {"commission_pct": 0.100, "spread_pct": 0.10,  "slippage_pct": 0.10},
    "de":      {"commission_pct": 0.100, "spread_pct": 0.10,  "slippage_pct": 0.10},
    "jp":      {"commission_pct": 0.050, "spread_pct": 0.08,  "slippage_pct": 0.08},
    "es":      {"commission_pct": 0.100, "spread_pct": 0.10,  "slippage_pct": 0.10},
    "hk":      {"commission_pct": 0.080, "spread_pct": 0.10,  "slippage_pct": 0.10},
    "fr":      {"commission_pct": 0.100, "spread_pct": 0.10,  "slippage_pct": 0.10},
    "in":      {"commission_pct": 0.150, "spread_pct": 0.10,  "slippage_pct": 0.10},
    "au":      {"commission_pct": 0.100, "spread_pct": 0.10,  "slippage_pct": 0.10},
    "ca":      {"commission_pct": 0.100, "spread_pct": 0.08,  "slippage_pct": 0.08},
    "kr":      {"commission_pct": 0.150, "spread_pct": 0.12,  "slippage_pct": 0.12},
    "futures": {"commission_pct": 0.002, "spread_pct": 0.01,  "slippage_pct": 0.02},
}

# ── Ticker display names ──────────────────────────────────────────────────────
_TICKER_NAMES: dict = {
    # ETFs — broad index
    "SPY": "S&P 500 ETF", "QQQ": "Nasdaq 100 ETF", "IWM": "Russell 2000 ETF",
    "DIA": "Dow Jones ETF", "VOO": "Vanguard S&P 500", "VTI": "Total Market ETF",
    "MDY": "S&P 400 Mid-Cap ETF",
    # ETFs — international
    "EFA": "Intl Developed Mkts", "EWG": "Germany ETF", "EWU": "UK ETF",
    "EWQ": "France ETF", "EWP": "Spain ETF", "EWJ": "Japan ETF",
    "EWH": "Hong Kong ETF", "FXI": "China Large-Cap ETF", "EEM": "Emerging Mkts ETF",
    # ETFs — US sector
    "XLF": "Financials ETF", "XLE": "Energy ETF", "XLK": "Technology ETF",
    "XLV": "Healthcare ETF", "XLI": "Industrials ETF", "XLU": "Utilities ETF",
    "XLP": "Cons Staples ETF", "XLB": "Materials ETF", "XLRE": "Real Estate ETF",
    "XLC": "Comms Services ETF",
    # ETFs — tech / semis
    "SMH": "Semiconductor ETF", "SOXX": "iShares Semis ETF", "ARKK": "ARK Innovation ETF",
    # ETFs — volatility / inverse
    "UVXY": "Short-Term VIX ETF", "SQQQ": "Nasdaq 3x Inverse", "SPXS": "S&P 3x Inverse",
    # Futures — equity index
    "ES=F": "S&P 500 Futures", "NQ=F": "Nasdaq 100 Futures",
    "RTY=F": "Russell 2000 Futures", "YM=F": "Dow Jones Futures",
    # Futures — bonds
    "ZN=F": "10-Yr T-Note Futures", "ZB=F": "30-Yr T-Bond Futures",
    # Futures — commodities
    "CL=F": "Crude Oil Futures", "NG=F": "Natural Gas Futures",
    "GC=F": "Gold Futures", "SI=F": "Silver Futures", "HG=F": "Copper Futures",
    "PL=F": "Platinum Futures", "PA=F": "Palladium Futures",
    # Futures — FX / vol / crypto
    "6E=F": "Euro Futures", "6J=F": "Japanese Yen Futures",
    "VX=F": "VIX Futures", "BTC=F": "Bitcoin Futures",
}

_name_cache: dict = {}


def get_ticker_name(ticker: str) -> str:
    """Return human-readable name for a ticker. Hardcoded dict first, then yfinance."""
    if ticker in _TICKER_NAMES:
        return _TICKER_NAMES[ticker]
    if ticker in _name_cache:
        return _name_cache[ticker]
    try:
        info = yf.Ticker(ticker).info
        name = info.get("longName") or info.get("shortName") or ""
        if name:
            _name_cache[ticker] = name
            return name
    except Exception:
        pass
    _name_cache[ticker] = ""
    return ""


# ── Benchmark map (RS vs index, one per market) ───────────────────────────────
BENCHMARK_MAP = {
    "us": "SPY",  "uk": "EWU",  "de": "EWG",  "jp": "EWJ",
    "es": "EWP",  "fr": "EWQ",  "hk": "EWH",
    "in": "INDA", "au": "EWA",  "ca": "EWC",  "kr": "EWY",
}

# ── Sector ETF map (for RS vs sector, Improvement 5) ─────────────────────────
SECTOR_ETF_MAP: dict = {
    "Technology":             "XLK",
    "Financial Services":     "XLF",
    "Financials":             "XLF",
    "Healthcare":             "XLV",
    "Consumer Cyclical":      "XLY",
    "Consumer Defensive":     "XLP",
    "Industrials":            "XLI",
    "Communication Services": "XLC",
    "Energy":                 "XLE",
    "Basic Materials":        "XLB",
    "Materials":              "XLB",
    "Real Estate":            "XLRE",
    "Utilities":              "XLU",
}
_SECTOR_ETF_CACHE: dict = {}  # populated lazily, one fetch per ETF per run

def fetch_sector_etf_return(sector: str) -> float:
    """Day return % of the sector proxy ETF. Cached per run."""
    etf = SECTOR_ETF_MAP.get(sector)
    if not etf:
        return 0.0
    if etf in _SECTOR_ETF_CACHE:
        return _SECTOR_ETF_CACHE[etf]
    try:
        raw = yf.download(etf, period="5d", interval="1d",
                          auto_adjust=True, progress=False, timeout=15)
        if raw is None or len(raw) < 2:
            _SECTOR_ETF_CACHE[etf] = 0.0
            return 0.0
        df = raw.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        c = df["Close"].astype(float).values
        ret = round(((c[-1] - c[-2]) / c[-2] * 100), 2) if c[-2] > 0 else 0.0
        _SECTOR_ETF_CACHE[etf] = ret
        return ret
    except Exception:
        _SECTOR_ETF_CACHE[etf] = 0.0
        return 0.0


# ── Futures roll-date awareness (Improvement 7) ───────────────────────────────
FUTURES_QUARTERLY = {"ES=F", "NQ=F", "RTY=F", "YM=F", "ZN=F", "ZB=F", "6E=F", "6J=F", "VX=F"}
FUTURES_MONTHLY   = {"CL=F", "NG=F", "GC=F", "SI=F", "HG=F", "PL=F", "PA=F", "BTC=F"}
ROLL_WARNING_DAYS = 7   # warn when ≤ this many days to roll

def _nth_friday(year: int, month: int, n: int) -> "datetime.date":
    """Return the nth Friday (1-indexed) of the given month."""
    cal     = monthcalendar(year, month)
    fridays = [week[FRIDAY] for week in cal if week[FRIDAY] != 0]
    return datetime(year, month, fridays[n - 1]).date()

def _next_quarterly_expiry(from_date=None):
    """3rd Friday of March, June, September, December — CME equity/financial futures."""
    from_date = from_date or datetime.now(timezone.utc).date()
    quarterly = {3, 6, 9, 12}
    for offset in range(6):
        raw_m = from_date.month + offset
        yr    = from_date.year + (raw_m - 1) // 12
        mo    = ((raw_m - 1) % 12) + 1
        if mo in quarterly:
            try:
                exp = _nth_friday(yr, mo, 3)
                if exp >= from_date:
                    return exp
            except IndexError:
                continue
    return None

def _next_monthly_expiry(from_date=None):
    """Last Friday of the month — approximation for commodity futures."""
    from_date = from_date or datetime.now(timezone.utc).date()
    for offset in range(3):
        raw_m = from_date.month + offset
        yr    = from_date.year + (raw_m - 1) // 12
        mo    = ((raw_m - 1) % 12) + 1
        try:
            cal     = monthcalendar(yr, mo)
            fridays = [week[FRIDAY] for week in cal if week[FRIDAY] != 0]
            exp     = datetime(yr, mo, fridays[-1]).date()
            if exp > from_date:
                return exp
        except Exception:
            continue
    return None

def futures_roll_info(ticker: str) -> dict:
    """Return roll-warning fields for a futures ticker."""
    today = datetime.now(timezone.utc).date()
    if ticker in FUTURES_QUARTERLY:
        exp = _next_quarterly_expiry(today)
    elif ticker in FUTURES_MONTHLY:
        exp = _next_monthly_expiry(today)
    else:
        exp = None

    if exp is None:
        return {"futures_roll_warning": False, "days_to_roll": None, "roll_date": None}

    days = (exp - today).days
    return {
        "futures_roll_warning": days <= ROLL_WARNING_DAYS,
        "days_to_roll":         days,
        "roll_date":            exp.isoformat(),
    }

# ── Global run stats ──────────────────────────────────────────────────────────
_STATS = {
    "universe_total":    0,
    "errors":            0,   # tickers returning no data / delisted
    "prescreen_pass":    0,
    "data_fail":         0,
    "filtered":          0,
    "candidates":        0,
    "by_market":         {},
}


# ═══════════════════════════════════════════════════════════════════════════════
#  STAGE 1 — DYNAMIC UNIVERSE FETCH
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_sp500() -> list:
    """S&P 500 constituents — tries multiple sources."""
    # Source 1: GitHub hosted CSV (reliable, no bot detection)
    try:
        url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
        r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        lines = r.text.strip().split("\n")
        tickers = [l.split(",")[0].strip().replace(".", "-") for l in lines[1:] if l.strip()]
        if len(tickers) > 400:
            logger.info(f"  S&P 500: {len(tickers)} constituents fetched (GitHub CSV)")
            return tickers
    except Exception as e:
        logger.debug(f"  S&P 500 GitHub CSV failed: {e}")

    # Source 2: Wikipedia with browser headers
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(
            requests.get(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}, timeout=15).text
        )
        df = tables[0]
        tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
        logger.info(f"  S&P 500: {len(tickers)} constituents fetched (Wikipedia)")
        return tickers
    except Exception as e:
        logger.warning(f"  S&P 500 all sources failed: {e} — using fallback")
        return _sp500_fallback()


def _wiki_tables(url: str) -> list:
    """Fetch Wikipedia tables with browser headers to avoid 403."""
    try:
        html = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
            timeout=15,
        ).text
        return pd.read_html(html)
    except Exception as e:
        raise RuntimeError(f"Wikipedia fetch failed: {e}")


def fetch_ftse100() -> list:
    """FTSE 100 constituents from Wikipedia."""
    try:
        url = "https://en.wikipedia.org/wiki/FTSE_100_Index"
        tables = _wiki_tables(url)
        # Find the table with ticker column
        for t in tables:
            cols = [c.lower() for c in t.columns]
            if any("ticker" in c or "epic" in c or "symbol" in c for c in cols):
                col = [c for c in t.columns if any(k in c.lower() for k in ["ticker","epic","symbol"])][0]
                tickers = [f"{s}.L" for s in t[col].dropna().tolist()]
                logger.info(f"  FTSE 100: {len(tickers)} constituents fetched")
                return tickers
        raise ValueError("No ticker column found")
    except Exception as e:
        logger.warning(f"  FTSE 100 fetch failed: {e} — using fallback")
        return _ftse100_fallback()


def fetch_dax() -> list:
    """DAX 40 constituents from Wikipedia."""
    try:
        url = "https://en.wikipedia.org/wiki/DAX"
        tables = pd.read_html(url)
        for t in tables:
            cols = [c.lower() for c in t.columns]
            if any("ticker" in c or "symbol" in c for c in cols):
                col = [c for c in t.columns if any(k in c.lower() for k in ["ticker","symbol"])][0]
                tickers = [f"{s}.DE" if not str(s).endswith(".DE") else s
                           for s in t[col].dropna().tolist()]
                logger.info(f"  DAX: {len(tickers)} constituents fetched")
                return tickers
        raise ValueError("No ticker column found")
    except Exception as e:
        logger.warning(f"  DAX fetch failed: {e} — using fallback")
        return _dax_fallback()


def fetch_nikkei225() -> list:
    """Nikkei 225 constituents from Wikipedia."""
    try:
        url = "https://en.wikipedia.org/wiki/Nikkei_225"
        tables = pd.read_html(url)
        for t in tables:
            cols = [c.lower() for c in t.columns]
            if any("code" in c or "ticker" in c or "symbol" in c for c in cols):
                col = [c for c in t.columns if any(k in c.lower() for k in ["code","ticker","symbol"])][0]
                tickers = [f"{str(s).zfill(4)}.T" for s in t[col].dropna().tolist()
                           if str(s).strip().isdigit()]
                logger.info(f"  Nikkei 225: {len(tickers)} constituents fetched")
                return tickers[:225]
        raise ValueError("No ticker column found")
    except Exception as e:
        logger.warning(f"  Nikkei 225 fetch failed: {e} — using fallback")
        return _nikkei_fallback()


def fetch_ibex35() -> list:
    """Ibex 35 constituents from Wikipedia."""
    try:
        url = "https://en.wikipedia.org/wiki/IBEX_35"
        tables = pd.read_html(url)
        for t in tables:
            cols = [c.lower() for c in t.columns]
            if any("ticker" in c or "symbol" in c for c in cols):
                col = [c for c in t.columns if any(k in c.lower() for k in ["ticker","symbol"])][0]
                tickers = [f"{s}.MC" if not str(s).endswith(".MC") else s
                           for s in t[col].dropna().tolist()]
                logger.info(f"  Ibex 35: {len(tickers)} constituents fetched")
                return tickers
        raise ValueError("No ticker column found")
    except Exception as e:
        logger.warning(f"  Ibex 35 fetch failed: {e} — using fallback")
        return _ibex_fallback()


def fetch_cac40() -> list:
    """CAC 40 constituents from Wikipedia."""
    try:
        url = "https://en.wikipedia.org/wiki/CAC_40"
        tables = pd.read_html(url)
        for t in tables:
            cols = [c.lower() for c in t.columns]
            if any("ticker" in c or "symbol" in c for c in cols):
                col = [c for c in t.columns if any(k in c.lower() for k in ["ticker","symbol"])][0]
                tickers = [f"{s}.PA" if not str(s).endswith(".PA") else s
                           for s in t[col].dropna().tolist()]
                logger.info(f"  CAC 40: {len(tickers)} constituents fetched")
                return tickers
        raise ValueError("No ticker column found")
    except Exception as e:
        logger.warning(f"  CAC 40 fetch failed: {e} — using fallback")
        return _cac40_fallback()


def fetch_hangseng() -> list:
    """Hang Seng constituents from Wikipedia."""
    try:
        url = "https://en.wikipedia.org/wiki/Hang_Seng_Index"
        tables = pd.read_html(url)
        for t in tables:
            cols = [c.lower() for c in t.columns]
            if any("code" in c or "ticker" in c for c in cols):
                col = [c for c in t.columns if any(k in c.lower() for k in ["code","ticker"])][0]
                tickers = [f"{str(s).zfill(4)}.HK" for s in t[col].dropna().tolist()
                           if str(s).strip().replace(".","").isdigit()]
                logger.info(f"  Hang Seng: {len(tickers)} constituents fetched")
                return tickers
        raise ValueError("No ticker column found")
    except Exception as e:
        logger.warning(f"  Hang Seng fetch failed: {e} — using fallback")
        return _hangseng_fallback()


def fetch_nasdaq_supplement() -> list:
    """
    Top ~300 liquid NASDAQ stocks NOT in S&P 500.
    Focused on high-beta: tech, biotech, crypto-adjacent, AI, leveraged ETFs.
    These are the stocks where 5%+ intraday moves happen regularly.
    """
    # Leveraged & inverse ETFs — extremely high ATR, always relevant
    leveraged_etfs = [
        "TQQQ","SQQQ","SPXL","SPXS","UPRO","SPXU","LABU","LABD",
        "SOXL","SOXS","TNA","TZA","FAS","FAZ","FNGU","FNGD",
        "UVXY","SVXY","VIXY","TVIX","TECL","TECS","CURE","ERX",
        "ERY","GUSH","DRIP","BOIL","KOLD","DUST","NUGT","JNUG",
    ]
    # High-beta NASDAQ tech & AI (not in S&P 500)
    nasdaq_tech = [
        "NVDA","AMD","TSLA","META","NFLX","COIN","MSTR","PLTR","SNOW",
        "RBLX","ARM","DDOG","CRWD","NET","ZS","OKTA","GTLB","BILL","HUBS",
        "MNDY","CFLT","AFRM","HOOD","SOFI","LCID","RIVN","NIO","XPEV","LI",
        "SMCI","MRVL","QCOM","TXN","LRCX","KLAC","AMAT","ASML","ONTO",
        "WOLF","AEHR","AMBA","INDI","MPWR","SLAB","SITM","ALGM","PSIX",
    ]
    # Biotech / pharma — high event-driven volatility
    nasdaq_biotech = [
        "MRNA","BNTX","REGN","BIIB","VRTX","ILMN","SGEN","INCY","BMRN",
        "ALNY","IONS","SRPT","RARE","FOLD","AKRO","MDGL","VKTX","RYTM",
        "CRSP","BEAM","EDIT","NTLA","VERV","BLUE","FATE","KYMR","IMVT",
        "ARQT","PRAX","RETA","DAWN","ROIV","SAVA","IOVA","NKTR","ADMA",
        "AXSM","ACAD","SAGE","INVA","PTCT","EXEL","MGNX","TGTX","ITCI",
    ]
    # Crypto / fintech / speculative
    nasdaq_speculative = [
        "MARA","RIOT","CLSK","CIFR","HUT","BTBT","BITF","WGMI",
        "PYPL","SQ","AFRM","UPST","LC","OPEN","OPAD","PRPB",
        "CLOV","WKHS","NKLA","GOEV","HYLN","FSR","MULN","BLNK",
    ]
    # High-momentum mid-caps with frequent big moves
    nasdaq_momentum = [
        "SHOP","UBER","LYFT","ABNB","DASH","SNAP","PINS","TWLO",
        "ZM","DOCN","DBX","BOX","BAND","FSLY","ESTC","MDB","COUCHBASE",
        "APPN","NCNO","TOST","BRZE","TASK","PWSC","SMAR","ALTR","JAMF",
        "DUOL","COUR","UDMY","SEMR","SPRK","WEAV","PCVX","CRNX",
        "ACVA","ACMR","CDAY","JAMF","KNSL","RYAN","PCTY","PAYC",
    ]

    all_nasdaq = (leveraged_etfs + nasdaq_tech + nasdaq_biotech +
                  nasdaq_speculative + nasdaq_momentum)
    # Deduplicate
    seen = set()
    unique = []
    for t in all_nasdaq:
        if t not in seen:
            seen.add(t)
            unique.append(t)

    logger.info(f"  NASDAQ supplement: {len(unique)} tickers")
    return unique


def fetch_russell_curated() -> list:
    """
    Curated Russell 2000 high-volatility names.
    Full Russell 2000 (2000 tickers) would take too long — this targets
    the highest-beta, most liquid small-caps where intraday moves of
    5-15%+ are common, especially on earnings and news catalysts.
    ~150 tickers covering: biotech, mining, energy, fintech, speculative tech.
    """
    russell_biotech = [
        "ACLS","ACNB","ADPT","ADUS","AGIO","AGTC","AKBA","AKRO","ALDX",
        "ALEC","ALGS","ALLO","ALPN","ALRS","ALTO","ALVR","AMGN","AMPH",
        "ANAB","ANIK","APLT","APOG","APRE","APTX","ARCT","ARDX","ARGT",
        "ARHS","ARMP","AROW","ARRY","ARTL","ARVN","ASND","ASRT","ATAI",
        "ATEC","ATEX","ATNI","ATRC","ATRS","ATVI","ATUS","AUPH","AURA",
        "AVDL","AVEO","AVIR","AVNS","AVRO","AVXL","AXNX","AYTU","AZTA",
    ]
    russell_energy_mining = [
        "AMR","ARCH","BTU","CEIX","CONSOL","ARLP","FELP","FANG","PDCE",
        "REI","REPX","RRC","SM","SBOW","TALO","VTLE","WTI","GPOR",
        "SWN","RKT","MTDR","CHRD","CIVI","BATL","ESTE","MNRL","NOG",
        "PHX","PPSI","SPRO","STNG","TRMD","TNK","DHT","EURN","FRO",
        "NMM","GOGL","SBLK","SALT","SHIP","TOPS","FREE","GNK","HSHP",
    ]
    russell_fintech_spec = [
        "ATLC","ENVA","EZCORP","FCFS","GPMT","HFBL","HIFS","HMNF",
        "ITIC","LMST","LNDC","LOAN","MBCN","MBIN","MFIN","MOFG",
        "NBTB","NKSH","NRIM","NWIN","OBNK","OCFC","OCSL","OPBK",
        "OPHC","OPOF","ORRF","OSBC","OVBC","OVLY","PBAM","PBFS",
        "CURO","PRAA","QD","QFIN","TREE","UWMC","GHLD","PFSI",
    ]
    russell_spec_tech = [
        "ACMR","AIOT","ALIT","ALLT","ALRM","ALRS","ALTI","ALTG",
        "ALTM","ALTR","ALTU","ALTV","ALVR","ALXO","ALYA","AMAM",
        "AMBC","AMBI","AMBO","AMCX","AMEH","AMED","AMIX","AMKR",
        "BFLY","BIGC","BLZE","BRSP","BTRS","BZFD","CABA","CELU",
        "CENN","CERE","CERT","CGEM","CHDN","CHGG","CHPT","CIFR",
        "CLOV","CLPT","CLRB","CLRO","CLSK","CLST","CLVS","CLXT",
        "CMPS","CNCE","CNDT","CNET","CNFN","CNMD","CNOB","CNSL",
        "CODA","CODX","COEP","COFS","COIN","COLB","COLI","COLL",
    ]

    all_russell = (russell_biotech + russell_energy_mining +
                   russell_fintech_spec + russell_spec_tech)
    seen = set()
    unique = [t for t in all_russell if not (t in seen or seen.add(t))]
    logger.info(f"  Russell 2000 curated: {len(unique)} tickers")
    return unique


def get_universe(market_key: str) -> list:
    """
    Fetch full index universe for a market.
    US combines S&P 500 + NASDAQ supplement + Russell 2000 curated.
    Other markets fetch official index constituents from Wikipedia.
    Falls back to static lists if fetch fails.
    """
    if market_key == "us":
        sp500   = fetch_sp500()
        nasdaq  = fetch_nasdaq_supplement()
        russell = fetch_russell_curated()
        # Combine and deduplicate — S&P 500 is authoritative, others supplement
        sp500_set = set(sp500)
        combined  = sp500[:]
        for t in nasdaq + russell:
            if t not in sp500_set:
                combined.append(t)
                sp500_set.add(t)
        logger.info(
            f"  US total universe: {len(combined)} "
            f"(S&P500={len(sp500)} + NASDAQ_supp={sum(1 for t in nasdaq if t not in set(sp500))} "
            f"+ Russell_curated={sum(1 for t in russell if t not in set(sp500) and t not in set(nasdaq))})"
        )
        return combined

    fetchers = {
        "uk": fetch_ftse100,
        "de": fetch_dax,
        "jp": fetch_nikkei225,
        "es": fetch_ibex35,
        "fr": fetch_cac40,
        "hk": fetch_hangseng,
        "in": fetch_nifty50,
        "au": fetch_asx200,
        "ca": fetch_tsx60,
        "kr": fetch_kospi,
    }
    fn = fetchers.get(market_key)
    if fn:
        tickers = fn()
        if tickers:
            return tickers
    return []


# ── Static fallbacks (used if Wikipedia fetch fails) ──────────────────────────

def _sp500_fallback():
    return [
        "NVDA","AMD","TSLA","META","AMZN","GOOGL","MSFT","AAPL","NFLX","CRM",
        "PLTR","SNOW","RBLX","COIN","MSTR","SMCI","ARM","AVGO","MU","INTC",
        "MRNA","BNTX","REGN","BIIB","VRTX","TQQQ","SQQQ","SPXL","UPRO",
        "LABU","SOXL","TNA","FAS","CRWD","DDOG","NET","ZS","PANW","OKTA",
        "XOM","CVX","OXY","HAL","SLB","FCX","NEM","GOLD","JPM","BAC","GS",
        "MS","WFC","C","V","MA","PYPL","SQ","SHOP","UBER","LYFT","ABNB",
    ]

def _ftse100_fallback():
    return [f"{t}.L" for t in [
        "HSBA","BP","SHEL","AZN","ULVR","RIO","GSK","BARC","LLOY","NWG",
        "TSCO","VOD","BT-A","IAG","FERG","CRH","EXPN","SGRO","LGEN","PHNX",
        "ABF","ADM","AHT","ANTO","AUTO","AV","AVV","BA","BAB","BATS",
        "BDEV","BKG","BLND","BME","BNZL","BOC","BRBY","BT-A","CCH","CNA",
    ]]

def _dax_fallback():
    return [f"{t}.DE" for t in [
        "SAP","SIE","ALV","MRK","BAYN","BMW","MBG","VOW3","DTE","DBK",
        "CON","ADS","EOAN","RWE","BAS","HEI","FRE","MUV2","DPW","IFX",
        "MTX","MUV2","PAH3","PUM","QIA","RHM","SAP","SHL","SY1","ZAL",
    ]]

def _nikkei_fallback():
    return [f"{t}.T" for t in [
        "7203","9984","6758","6861","8306","9432","6501","7974","4519",
        "8316","6902","4661","9433","8035","6954","7267","4543","2914","9020","4568",
    ]]

def _ibex_fallback():
    return [f"{t}.MC" for t in [
        "SAN","BBVA","ITX","IBE","REP","TEF","AMS","ANA","ELE","CLNX",
        "FER","GRF","IAG","MAP","MEL","MTS","NTGY","RED","ROVI","SAB",
    ]]

def _cac40_fallback():
    return [f"{t}.PA" for t in [
        "MC","OR","TTE","SAN","AIR","BNP","SU","RI","CAP","ACA",
        "BN","KER","DG","GLE","HO","LR","ML","ORA","PUB","RMS",
        "SAF","SGO","STLA","STM","SW","VIE","VIV","WLN","CS","EDF",
    ]]

def _hangseng_fallback():
    return [f"{t}.HK" for t in [
        "0700","0005","0939","1299","0941","2318","0388","1398","2628","0003",
        "0011","0002","0016","0027","1810","9988","0175","1177","2020","6862",
    ]]


def fetch_nifty50() -> list:
    """Nifty 50 constituents from Wikipedia (.NS suffix)."""
    try:
        url = "https://en.wikipedia.org/wiki/NIFTY_50"
        tables = _wiki_tables(url)
        for t in tables:
            cols = [c.lower() for c in t.columns]
            if any("symbol" in c or "ticker" in c for c in cols):
                col = [c for c in t.columns if any(k in c.lower() for k in ["symbol","ticker"])][0]
                tickers = [f"{str(s).strip()}.NS" if not str(s).endswith(".NS") else s
                           for s in t[col].dropna().tolist() if str(s).strip()]
                tickers = [tk for tk in tickers if len(tk) > 3]
                logger.info(f"  Nifty 50: {len(tickers)} constituents fetched")
                return tickers
        raise ValueError("No symbol column found")
    except Exception as e:
        logger.warning(f"  Nifty 50 fetch failed: {e} — using fallback")
        return _nifty50_fallback()


def fetch_asx200() -> list:
    """S&P/ASX 200 constituents from Wikipedia (.AX suffix)."""
    try:
        url = "https://en.wikipedia.org/wiki/S%26P/ASX_200"
        tables = _wiki_tables(url)
        for t in tables:
            cols = [c.lower() for c in t.columns]
            if any("ticker" in c or "code" in c or "symbol" in c for c in cols):
                col = [c for c in t.columns if any(k in c.lower() for k in ["ticker","code","symbol"])][0]
                tickers = [f"{str(s).strip()}.AX" if not str(s).endswith(".AX") else s
                           for s in t[col].dropna().tolist() if str(s).strip()]
                tickers = [tk for tk in tickers if len(tk) > 3]
                logger.info(f"  ASX 200: {len(tickers)} constituents fetched")
                return tickers
        raise ValueError("No ticker column found")
    except Exception as e:
        logger.warning(f"  ASX 200 fetch failed: {e} — using fallback")
        return _asx200_fallback()


def fetch_tsx60() -> list:
    """S&P/TSX 60 constituents from Wikipedia (.TO suffix)."""
    try:
        url = "https://en.wikipedia.org/wiki/S%26P/TSX_60"
        tables = _wiki_tables(url)
        for t in tables:
            cols = [c.lower() for c in t.columns]
            if any("ticker" in c or "symbol" in c for c in cols):
                col = [c for c in t.columns if any(k in c.lower() for k in ["ticker","symbol"])][0]
                tickers = [f"{str(s).strip()}.TO" if not str(s).endswith(".TO") else s
                           for s in t[col].dropna().tolist() if str(s).strip()]
                tickers = [tk for tk in tickers if len(tk) > 3]
                logger.info(f"  TSX 60: {len(tickers)} constituents fetched")
                return tickers
        raise ValueError("No ticker column found")
    except Exception as e:
        logger.warning(f"  TSX 60 fetch failed: {e} — using fallback")
        return _tsx60_fallback()


def fetch_kospi() -> list:
    """KOSPI 200 constituents from Wikipedia (.KS suffix)."""
    try:
        url = "https://en.wikipedia.org/wiki/KOSPI_200"
        tables = _wiki_tables(url)
        for t in tables:
            cols = [c.lower() for c in t.columns]
            if any("code" in c or "ticker" in c or "symbol" in c for c in cols):
                col = [c for c in t.columns if any(k in c.lower() for k in ["code","ticker","symbol"])][0]
                tickers = [f"{str(s).strip().zfill(6)}.KS" if not str(s).endswith(".KS") else s
                           for s in t[col].dropna().tolist()
                           if str(s).strip().replace(" ","").replace("-","").isdigit()]
                logger.info(f"  KOSPI 200: {len(tickers)} constituents fetched")
                return tickers
        raise ValueError("No code column found")
    except Exception as e:
        logger.warning(f"  KOSPI 200 fetch failed: {e} — using fallback")
        return _kospi_fallback()


def _nifty50_fallback():
    return [f"{t}.NS" for t in [
        "RELIANCE","TCS","HDFCBANK","INFY","ICICIBANK","HINDUNILVR","KOTAKBANK",
        "SBIN","BAJFINANCE","BHARTIARTL","ITC","AXISBANK","LT","WIPRO","ULTRACEMCO",
        "HCLTECH","SUNPHARMA","TITAN","ASIANPAINT","MARUTI","ADANIENT","POWERGRID",
        "NTPC","TECHM","BAJAJFINSV","NESTLEIND","ONGC","DIVISLAB","DRREDDY","EICHERMOT",
    ]]

def _asx200_fallback():
    return [f"{t}.AX" for t in [
        "BHP","CBA","CSL","NAB","ANZ","WBC","WES","MQG","RIO","WOW",
        "TCL","TLS","GMG","MIN","FMG","NCM","REA","COL","STO","QAN",
        "AGL","ASX","AMC","AMP","ALX","BXB","IAG","MPL","ORG","SHL",
    ]]

def _tsx60_fallback():
    return [f"{t}.TO" for t in [
        "RY","TD","BNS","BMO","CM","MFC","SLF","ENB","CNQ","TRP",
        "SU","CNR","CP","BCE","T","SHOP","BAM","ATD","WN","ABX",
        "IMO","CVE","CCO","QSR","POW","GIB-A","DOL","L","EMA","FTS",
    ]]

def _kospi_fallback():
    return [
        "005930.KS","000660.KS","035420.KS","005380.KS","051910.KS",
        "006400.KS","035720.KS","207940.KS","068270.KS","055550.KS",
        "105560.KS","012330.KS","028260.KS","066570.KS","032830.KS",
        "086790.KS","017670.KS","030200.KS","010950.KS","096770.KS",
    ]


# ── Benchmark return helper (Point 2 — RS vs index) ──────────────────────────

def fetch_benchmark_return(market_key: str) -> float:
    """Return today's day_return % for the benchmark index of a market."""
    ticker = BENCHMARK_MAP.get(market_key)
    if not ticker:
        return 0.0
    try:
        raw = yf.download(ticker, period="5d", interval="1d",
                          auto_adjust=True, progress=False, timeout=15)
        if raw is None or len(raw) < 2:
            return 0.0
        df = flatten_df(raw)
        c  = df["Close"].astype(float).values
        return round(((c[-1] - c[-2]) / c[-2] * 100), 2) if c[-2] > 0 else 0.0
    except Exception:
        return 0.0


# ═══════════════════════════════════════════════════════════════════════════════
#  STAGE 2 — VOLUME PRE-SCREEN
# ═══════════════════════════════════════════════════════════════════════════════

def flatten_df(raw: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance v0.2+ MultiIndex columns."""
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    return raw.loc[:, ~raw.columns.duplicated()]


def _prescreen_batch(batch: list, rvol_thr: float, atr_thr: float, min_vol: int) -> tuple:
    """
    Download and screen one batch of tickers via a single yf.download() call.
    Returns (list_of_passed_dicts, error_count).
    """
    passed = []
    errors = 0
    try:
        if len(batch) == 1:
            raw = yf.download(batch[0], period="5d", interval="1d",
                              auto_adjust=True, progress=False, timeout=30)
            if raw is None or raw.empty:
                return passed, 1
            ticker_dfs = {batch[0]: flatten_df(raw)}
        else:
            raw = yf.download(batch, period="5d", interval="1d",
                              auto_adjust=True, progress=False, timeout=30,
                              group_by="ticker")
            ticker_dfs = {}
            for ticker in batch:
                try:
                    df = raw[ticker].dropna(how="all")
                    if len(df) >= 3:
                        ticker_dfs[ticker] = df
                except (KeyError, TypeError):
                    errors += 1
    except Exception:
        return passed, len(batch)

    for ticker, df in ticker_dfs.items():
        try:
            if "Volume" not in df.columns:
                continue
            vols      = df["Volume"].astype(float).values
            today_vol = float(vols[-1])
            avg_vol   = float(vols[:-1].mean()) if len(vols) > 1 else today_vol
            rvol      = today_vol / avg_vol if avg_vol > 0 else 1.0

            atr_pct = 0.0
            if all(c in df.columns for c in ["High", "Low", "Close"]):
                h = df["High"].astype(float).values
                l = df["Low"].astype(float).values
                c = df["Close"].astype(float).values
                if len(c) >= 2 and c[-1] > 0:
                    trs = [max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
                           for i in range(1, len(c))]
                    atr_pct = (sum(trs) / len(trs)) / c[-1] * 100

            passes_rvol = rvol >= rvol_thr
            passes_atr  = atr_pct >= atr_thr
            if today_vol >= min_vol and (passes_rvol or passes_atr):
                passed.append({
                    "ticker":    ticker,
                    "rvol":      round(rvol, 2),
                    "atr_pct":   round(atr_pct, 2),
                    "volume":    int(today_vol),
                    "passed_by": "RVOL" if passes_rvol else "ATR",
                })
        except Exception:
            errors += 1

    return passed, errors


def prescreen_volume(
    tickers: list,
    market_key: str,
    rvol_threshold: float = None,
    atr_threshold: float = None,
    min_vol_override: int = None,
    max_pass: int = None,
) -> list:
    """
    Batch + parallel pre-screen. Splits tickers into batches of PRESCREEN_BATCH_SIZE,
    downloads each batch in a single yf.download() call, runs MAX_WORKERS_PRESCREEN
    batches concurrently. ~10x faster than sequential single-ticker downloads.
    """
    rvol_thr = rvol_threshold if rvol_threshold is not None else PRESCREEN_RVOL
    atr_thr  = atr_threshold  if atr_threshold  is not None else PRESCREEN_ATR
    min_vol  = min_vol_override if min_vol_override is not None else PRESCREEN_MIN_VOL.get(market_key, 50_000)
    cap      = max_pass if max_pass is not None else MAX_PRESCREEN_PASS

    logger.info(f"  Pre-screening {len(tickers)} tickers (batch={PRESCREEN_BATCH_SIZE}, workers={MAX_WORKERS_PRESCREEN})...")

    batches  = [tickers[i:i+PRESCREEN_BATCH_SIZE] for i in range(0, len(tickers), PRESCREEN_BATCH_SIZE)]
    screened = []
    errors   = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS_PRESCREEN) as ex:
        futures = {ex.submit(_prescreen_batch, b, rvol_thr, atr_thr, min_vol): b for b in batches}
        for fut in as_completed(futures):
            batch_passed, batch_errors = fut.result()
            screened.extend(batch_passed)
            errors   += batch_errors

    _STATS["errors"] += errors
    screened.sort(key=lambda x: x["rvol"], reverse=True)

    if screened:
        by_rvol = sum(1 for s in screened if s["passed_by"] == "RVOL")
        by_atr  = sum(1 for s in screened if s["passed_by"] == "ATR")
        logger.info(
            f"  Pre-screen: {len(screened)} passed from {len(tickers)} "
            f"(RVOL>={rvol_thr}x: {by_rvol} | ATR>={atr_thr}%: {by_atr}) "
            f"| Top: {screened[0]['ticker']} RVOL={screened[0]['rvol']}x ATR={screened[0]['atr_pct']}%"
        )
    else:
        logger.info(f"  Pre-screen: 0 passed from {len(tickers)} — market may be closed or inactive")

    if len(screened) > cap:
        logger.info(f"  Capping at top {cap} by RVOL for full analysis")
        screened = screened[:cap]

    return [s["ticker"] for s in screened]


# ═══════════════════════════════════════════════════════════════════════════════
#  STAGE 3 — FULL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_snapshot(ticker: str) -> Optional[pd.DataFrame]:
    """Fetch 1 year daily OHLCV for full analysis (enables 52-week range metrics)."""
    for period in ["1y", "6mo"]:
        try:
            raw = yf.download(
                ticker, period=period, interval="1d",
                auto_adjust=True, progress=False, timeout=20,
            )
            if raw is None or len(raw) < 5:
                continue
            df = flatten_df(raw)
            required = {"Open","High","Low","Close","Volume"}
            if not required.issubset(set(df.columns)):
                continue
            return df
        except Exception as e:
            logger.debug(f"    {ticker}: fetch error — {e}")
    return None


def _fetch_and_analyse(ticker: str) -> tuple:
    """Fetch + analyse one ticker. Returns (ticker, metrics_dict_or_None)."""
    df = fetch_snapshot(ticker)
    if df is None:
        return ticker, None
    m = analyse_ticker(ticker, df)
    return ticker, m if m else None


def compute_atr(df: pd.DataFrame, period: int = 10) -> float:
    if len(df) < 3:
        return 0.0
    try:
        period = min(period, len(df) - 1)
        h = df["High"].astype(float).values
        l = df["Low"].astype(float).values
        c = df["Close"].astype(float).values
        trs = [max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
               for i in range(1, len(c))]
        atr = sum(trs[-period:]) / period
        return (atr / c[-1]) * 100 if c[-1] > 0 else 0.0
    except Exception:
        return 0.0


def compute_rvol(df: pd.DataFrame) -> float:
    try:
        vols = df["Volume"].astype(float).values
        lb   = min(20, len(vols) - 1)
        avg  = sum(vols[-lb-1:-1]) / lb
        return float(vols[-1]) / avg if avg > 0 else 1.0
    except Exception:
        return 1.0


def compute_rsi(closes, period: int = 10) -> float:
    period = min(period, len(closes) - 2)
    if period < 2:
        return 50.0
    gains = [max(float(closes[i])-float(closes[i-1]), 0) for i in range(1, len(closes))]
    losses= [max(float(closes[i-1])-float(closes[i]), 0) for i in range(1, len(closes))]
    ag = sum(gains[-period:]) / period
    al = sum(losses[-period:]) / period
    return 100.0 if al == 0 else 100 - (100 / (1 + ag/al))


def analyse_ticker(ticker: str, df: pd.DataFrame) -> dict:
    try:
        c = df["Close"].astype(float).values
        o = df["Open"].astype(float).values
        h = df["High"].astype(float).values
        l = df["Low"].astype(float).values
        v = df["Volume"].astype(float).values

        atr_pct    = compute_atr(df)
        rvol       = compute_rvol(df)
        rsi        = compute_rsi(c)
        day_return = ((c[-1]-c[-2])/c[-2]*100) if len(c)>=2 and c[-2]>0 else 0.0
        day_range  = ((h[-1]-l[-1])/l[-1]*100) if l[-1]>0 else 0.0

        # ── 52-week range position (Point 5) ──────────────────────────────────
        w52_h     = float(h[-252:].max()) if len(h) >= 252 else float(h.max())
        w52_l     = float(l[-252:].min()) if len(l) >= 252 else float(l.min())
        range_pos = round((c[-1] - w52_l) / (w52_h - w52_l), 2) if w52_h > w52_l else 0.5
        # Bonus: near 52w high = breakout candidate (+10), mid-range (+5), below mid (0)
        range52_bonus = 10 if range_pos >= 0.85 else (5 if range_pos >= 0.50 else 0)

        # ── MA alignment bonus ─────────────────────────────────────────────────
        # Full bull alignment MA20>MA50>MA200 = +10, partial = +5, none = 0
        ma20  = float(pd.Series(c).rolling(20).mean().iloc[-1])  if len(c) >= 20  else c[-1]
        ma50  = float(pd.Series(c).rolling(50).mean().iloc[-1])  if len(c) >= 50  else c[-1]
        ma200 = float(pd.Series(c).rolling(200).mean().iloc[-1]) if len(c) >= 200 else c[-1]
        if c[-1] > ma20 > ma50 > ma200:
            ma_bonus = 10; ma_align = "MA↑"
        elif c[-1] > ma50 and ma50 > ma200:
            ma_bonus = 5;  ma_align = "MA→"
        else:
            ma_bonus = 0;  ma_align = "MA↓"

        # ── Improvement 2: Structural SHORT qualification ──────────────────────
        # A ticker only gets SHORT bias when ALL four conditions hold:
        # (1) confirmed downtrend: price < MA50 < MA200
        # (2) elevated sell-side volume: RVOL ≥ 1.5×
        # (3) bearish momentum: RSI < 50
        # (4) price below 20-day MA (no near-term support bounce)
        # Everything else defaults to LONG (follow the primary trend).
        if day_return >= 0:
            bias = "LONG"
        elif (c[-1] < ma50) and (ma50 < ma200) and (rvol >= 1.5) and (rsi < 50) and (c[-1] < ma20):
            bias = "SHORT"
        else:
            bias = "LONG"

        # ── Improvement 4: Weekly trend confirmation ───────────────────────────
        # Sample every 5 daily closes → synthetic weekly series → 20-week MA.
        # weekly_trend_up = True means price is above its 20-week moving average.
        weekly_closes = c[::5]
        if len(weekly_closes) >= 20:
            wma20 = float(pd.Series(weekly_closes).rolling(20).mean().iloc[-1])
            weekly_trend_up = bool(c[-1] > wma20)
        else:
            # Not enough history — fall back to 50-day MA as proxy
            weekly_trend_up = bool(c[-1] > ma50)

        score = (
            min(atr_pct/5.0, 1.0)*30 +
            min(rvol/3.0, 1.0)*25 +
            min(abs(day_return)/3.0, 1.0)*25 +
            (1.0 if 60<=rsi<=80 else 0.6 if 40<=rsi<60 else 0.4)*20 +
            range52_bonus + ma_bonus
        )

        logger.info(
            f"    {ticker}: price={c[-1]:.2f} | ATR={atr_pct:.2f}% | "
            f"RVOL={rvol:.2f}x | RSI={rsi:.0f} | day={day_return:+.2f}% | "
            f"range={day_range:.2f}% | 52w={range_pos:.0%} | vol={int(v[-1]):,} | score={score:.0f}"
        )

        return {
            "ticker": ticker, "price": round(float(c[-1]),4),
            "atr_pct": round(atr_pct,2), "rvol": round(rvol,2),
            "rsi": round(rsi,1), "day_return": round(day_return,2),
            "day_range": round(day_range,2), "volume": int(v[-1]),
            "score": round(score,1),
            "week52_high": round(w52_h,4), "week52_low": round(w52_l,4),
            "range_pos": range_pos, "ma_align": ma_align,
            "bias": bias,
            "weekly_trend_up": weekly_trend_up,
        }
    except Exception as e:
        logger.info(f"    {ticker}: analysis error — {e}")
        return {}


def passes_filters(m: dict, market: str, tier: str = "stock") -> tuple:
    """
    Tier-aware filter gate.

    - ETF/Future tiers use TIER_CONFIG thresholds and skip stock-style
      min-price / share-volume floors where appropriate.
    - Stock tier retains existing behaviour (100% backward compatible).
    - ATR/score gates for all tiers are only tightened when MIN_MOVE_PCT > 0.
    """
    vol   = m.get("volume", 0)
    rvol  = m.get("rvol", 0)
    atr   = m.get("atr_pct", 0)
    score = m.get("score", 0)

    tc = TIER_CONFIG.get(tier, TIER_CONFIG["stock"])

    # Volume floor (skipped for futures — contract volume is tiny by design)
    if not tc["skip_vol_floor"]:
        min_vol = tc["min_vol"] if tc["min_vol"] is not None else PRESCREEN_MIN_VOL.get(market, 50_000)
        if vol < min_vol:
            return False, f"vol {vol:,} < {min_vol:,}"

    # Momentum gate: RVOL OR ATR
    filter_rvol = tc["filter_rvol"]
    filter_atr  = tc["filter_atr"]
    if rvol < filter_rvol and atr < filter_atr:
        return False, f"RVOL {rvol:.2f}x < {filter_rvol}x and ATR {atr:.2f}% < {filter_atr}%"

    min_score = tc["filter_score"]
    if MIN_MOVE_PCT > 0 and tier == "stock":
        # MIN_MOVE_PCT ATR gate applies to stocks only — ETF/Future use TIER_CONFIG thresholds
        if atr < MIN_MOVE_PCT * 0.7:
            return False, f"ATR {atr:.2f}% < {MIN_MOVE_PCT*0.7:.2f}%"
        if score < min_score:
            return False, f"score {score:.0f} < {min_score}"
    else:
        floor = max(min_score - 15, 10)
        if score < floor:
            return False, f"score {score:.0f} < {floor}"

    return True, "passed"


def net_yield(atr_pct: float, market: str, tier: str = "stock") -> float:
    tc      = TIER_CONFIG.get(tier, TIER_CONFIG["stock"])
    fee_key = tc["fee_key"] if tc["fee_key"] is not None else market
    f       = FEE_MODEL.get(fee_key, FEE_MODEL["us"])
    cost    = (f["commission_pct"] + f["spread_pct"] + f["slippage_pct"]) * 2
    return round(atr_pct - cost, 2)


# ═══════════════════════════════════════════════════════════════════════════════
#  STAGE 4 — ORCHESTRATION
# ═══════════════════════════════════════════════════════════════════════════════

def scan_tier(tier_name: str, universe: list) -> list:
    """
    Run the full pipeline for a hardcoded-universe tier (etf or future).
    Uses tier-specific pre-screen thresholds and filter gates.
    Tags every candidate with {"tier": tier_name, "market": "GLOBAL"}.
    """
    tc = TIER_CONFIG[tier_name]
    logger.info(f"\n{'='*60}")
    logger.info(f"TIER: {tier_name.upper()} | {len(universe)} instruments | Target: >{MIN_MOVE_PCT}%")
    logger.info(f"{'='*60}")

    prescreened = prescreen_volume(
        universe,
        market_key="us",
        rvol_threshold=tc["prescreen_rvol"],
        atr_threshold=tc["prescreen_atr"],
        min_vol_override=tc["min_vol"],
        max_pass=tc["max_pass"],
    )
    if not prescreened:
        logger.info(f"  {tier_name.upper()}: no instruments passed pre-screen")
        _STATS["by_market"][tier_name.upper()] = {
            "universe": len(universe), "prescreen_pass": 0,
            "data_fail": 0, "filtered": 0, "candidates": 0,
        }
        _STATS["universe_total"] += len(universe)
        return []

    logger.info(f"\n  Full analysis on {len(prescreened)} pre-screened {tier_name}s (parallel, workers={MAX_WORKERS_ANALYSIS}):")
    candidates, data_fail, filtered = [], 0, 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS_ANALYSIS) as ex:
        futures = {ex.submit(_fetch_and_analyse, t): t for t in prescreened}
        for fut in as_completed(futures):
            ticker, m = fut.result()
            if m is None:
                data_fail += 1
                continue
            passed, reason = passes_filters(m, market="us", tier=tier_name)
            if not passed:
                logger.info(f"    {ticker}: FAIL {reason}")
                filtered += 1
                continue
            logger.info(f"    {ticker}: CANDIDATE")
            roll = futures_roll_info(ticker) if tier_name == "future" else {
                "futures_roll_warning": False, "days_to_roll": None, "roll_date": None,
            }
            candidates.append({
                **m,
                "tier":        tier_name,
                "market":      "GLOBAL",
                "net_yield":   net_yield(m["atr_pct"], market="us", tier=tier_name),
                "rs_vs_bench": 0.0,
                "rs_vs_sector":       None,
                "short_interest_pct": None,
                "earnings_surprise_avg": None,
                **roll,
            })

    _STATS["universe_total"] += len(universe)
    _STATS["prescreen_pass"] += len(prescreened)
    _STATS["data_fail"]      += data_fail
    _STATS["filtered"]       += filtered
    _STATS["candidates"]     += len(candidates)
    _STATS["by_market"][tier_name.upper()] = {
        "universe":       len(universe),
        "prescreen_pass": len(prescreened),
        "data_fail":      data_fail,
        "filtered":       filtered,
        "candidates":     len(candidates),
    }

    logger.info(
        f"\n  {tier_name.upper()}: universe={len(universe)} | "
        f"prescreen={len(prescreened)} | data_fail={data_fail} | "
        f"filtered={filtered} | candidates={len(candidates)}"
    )
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates


def scan_market(market_key: str) -> list:
    logger.info(f"\n{'='*60}")
    logger.info(f"MARKET: {market_key.upper()} | Target: >{MIN_MOVE_PCT}%")
    logger.info(f"{'='*60}")

    # Stage 1: Get full universe
    universe = get_universe(market_key)
    if not universe:
        logger.warning(f"  No universe for {market_key}")
        return []
    logger.info(f"  Universe: {len(universe)} stocks")

    # Stage 2: Volume pre-screen
    prescreened = prescreen_volume(universe, market_key)
    if not prescreened:
        logger.info(f"  No stocks passed pre-screen — market may be inactive")
        _STATS["by_market"][market_key.upper()] = {
            "universe": len(universe), "prescreen_pass": 0,
            "data_fail": 0, "filtered": 0, "candidates": 0,
        }
        _STATS["universe_total"] += len(universe)
        return []

    logger.info(f"\n  Full analysis on {len(prescreened)} pre-screened stocks (parallel, workers={MAX_WORKERS_ANALYSIS}):")

    # Fetch benchmark return once for relative-strength calculation (Point 2)
    bench_return = fetch_benchmark_return(market_key)
    if bench_return != 0.0:
        logger.info(f"  Benchmark ({BENCHMARK_MAP.get(market_key,'?')}): {bench_return:+.2f}%")

    # Stage 3: Full analysis — parallel
    candidates, data_fail, filtered = [], 0, 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS_ANALYSIS) as ex:
        futures = {ex.submit(_fetch_and_analyse, t): t for t in prescreened}
        for fut in as_completed(futures):
            ticker, m = fut.result()
            if m is None:
                data_fail += 1
                continue
            passed, reason = passes_filters(m, market_key, tier="stock")
            if not passed:
                logger.info(f"    {ticker}: FAIL {reason}")
                filtered += 1
                continue
            logger.info(f"    {ticker}: CANDIDATE")
            candidates.append({
                **m,
                "tier":        "stock",
                "market":      market_key.upper(),
                "net_yield":   net_yield(m["atr_pct"], market_key, tier="stock"),
                "rs_vs_bench": round(m["day_return"] - bench_return, 2),
                "rs_vs_sector":          None,
                "short_interest_pct":    None,
                "earnings_surprise_avg": None,
                "futures_roll_warning":  False,
                "days_to_roll":          None,
                "roll_date":             None,
            })

    # Accumulate stats
    mkt_errors      = _STATS["errors"] - sum(s.get("errors",0) for s in _STATS["by_market"].values())
    addressable     = len(universe) - mkt_errors
    _STATS["universe_total"] += len(universe)
    _STATS["prescreen_pass"] += len(prescreened)
    _STATS["data_fail"]      += data_fail
    _STATS["filtered"]       += filtered
    _STATS["candidates"]     += len(candidates)
    _STATS["by_market"][market_key.upper()] = {
        "universe":       len(universe),
        "errors":         mkt_errors,
        "addressable":    addressable,
        "prescreen_pass": len(prescreened),
        "data_fail":      data_fail,
        "filtered":       filtered,
        "candidates":     len(candidates),
    }

    logger.info(
        f"\n  {market_key.upper()}: universe={len(universe)} | "
        f"prescreen={len(prescreened)} | data_fail={data_fail} | "
        f"filtered={filtered} | candidates={len(candidates)}"
    )

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates


def run_all_markets() -> list:
    _STATS.update({
        "universe_total": 0, "prescreen_pass": 0,
        "data_fail": 0, "filtered": 0, "candidates": 0, "by_market": {},
    })

    if MARKET_CONTEXT == "all":
        markets = ["us", "ca", "uk", "de", "jp", "es", "hk", "fr", "in", "au", "kr"]
    elif MARKET_CONTEXT == "eu":
        markets = ["uk", "de", "es", "fr"]
    elif MARKET_CONTEXT == "asia":
        markets = ["jp", "hk", "au", "kr"]   # 00:00-06:00 UTC block
    elif MARKET_CONTEXT == "us":
        markets = ["us", "ca"]               # TSX opens same time as NYSE
    else:
        markets = [MARKET_CONTEXT] if MARKET_CONTEXT in FEE_MODEL else ["us"]

    all_c = []

    # ETF and Futures tiers always run (global, not market-gated)
    all_c.extend(scan_tier("etf",    ETF_UNIVERSE))
    all_c.extend(scan_tier("future", FUTURES_UNIVERSE))

    # Stock tier per market
    for m in markets:
        all_c.extend(scan_market(m))

    all_c.sort(key=lambda x: x["score"], reverse=True)

    # Global summary
    total_errors      = _STATS["errors"]
    total_addressable = _STATS["universe_total"] - total_errors
    logger.info("\n" + "=" * 90)
    logger.info("GLOBAL SCAN SUMMARY")
    logger.info("=" * 90)
    logger.info(f"  {'Market':<8} {'Universe':>10} {'Errors':>8} {'Addressable':>13} {'Pre-screen':>12} {'Failed':>8} {'Filtered':>10} {'Final':>7}")
    logger.info(f"  {'-'*8} {'-'*10} {'-'*8} {'-'*13} {'-'*12} {'-'*8} {'-'*10} {'-'*7}")
    for mkt, s in _STATS["by_market"].items():
        logger.info(
            f"  {mkt:<8} {s['universe']:>10} {s.get('errors',0):>8} {s.get('addressable', s['universe']):>13} "
            f"{s['prescreen_pass']:>12} {s['data_fail']:>8} {s['filtered']:>10} {s['candidates']:>7}"
        )
    logger.info(f"  {'─'*8} {'─'*10} {'─'*8} {'─'*13} {'─'*12} {'─'*8} {'─'*10} {'─'*7}")
    logger.info(
        f"  {'TOTAL':<8} {_STATS['universe_total']:>10} {total_errors:>8} {total_addressable:>13} "
        f"{_STATS['prescreen_pass']:>12} {_STATS['data_fail']:>8} {_STATS['filtered']:>10} {_STATS['candidates']:>7}"
    )
    logger.info("=" * 90)

    return all_c


# ═══════════════════════════════════════════════════════════════════════════════
#  OUTPUT
# ═══════════════════════════════════════════════════════════════════════════════

def format_markdown(candidates: list) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    stats_by_mkt = _STATS.get("by_market", {})

    if not candidates:
        return (
            f"## 📊 Intraday Scanner — {now}\n\n"
            f"**Market:** {MARKET_CONTEXT.upper()}  |  **Target:** >{MIN_MOVE_PCT}%\n\n"
            f"**Universe scanned:** {_STATS['universe_total']:,} instruments  |  "
            f"**Pre-screened:** {_STATS['prescreen_pass']}  |  **Candidates:** 0\n\n"
            "**No candidates met the filter criteria.**\n\n"
            "> ⚠️ Research only. Not financial advice.\n"
        )

    lines = [
        f"## 📊 Intraday Scanner — {now}",
        f"**Market:** {MARKET_CONTEXT.upper()}  |  **Target:** >{MIN_MOVE_PCT}%  |  **Candidates:** {len(candidates)}",
        f"**Universe:** {_STATS['universe_total']:,} instruments scanned  |  "
        f"**Pre-screen pass:** {_STATS['prescreen_pass']}  |  "
        f"**Final candidates:** {_STATS['candidates']}",
        "",
        "> ⚠️ Research only. Not financial advice. Past ATR ≠ future moves.",
        "",
        "| # | Tier | Ticker | Mkt | Price | Day% | Range% | ATR% | RVOL | RSI | Score | Net |",
        "|---|------|--------|-----|-------|------|--------|------|------|-----|-------|-----|",
    ]
    for i, c in enumerate(candidates[:30], 1):
        e    = "🟢" if c["day_return"] > 0 else "🔴"
        n    = "✅" if c["net_yield"] > 0 else "❌"
        tier = c.get("tier", "stock")
        tier_label = {"etf": "ETF", "future": "FUT", "stock": "STK"}.get(tier, tier.upper())
        lines.append(
            f"| {i} | {tier_label} | **{c['ticker']}** | {c['market']} | {c['price']:.2f} | "
            f"{e}{c['day_return']:+.1f}% | {c['day_range']:.1f}% | {c['atr_pct']:.1f}% | "
            f"{c['rvol']:.1f}x | {c['rsi']:.0f} | **{c['score']:.0f}** | {n}{c['net_yield']:+.1f}% |"
        )

    # Per-market breakdown table
    if stats_by_mkt:
        total_errors      = _STATS.get("errors", 0)
        total_addressable = _STATS["universe_total"] - total_errors
        lines += [
            "",
            "### Market Breakdown",
            "| Market | Universe | Errors | Addressable | Pre-screen | Failed | Filtered | Candidates |",
            "|--------|----------|--------|-------------|------------|--------|----------|------------|",
        ]
        for mkt, s in stats_by_mkt.items():
            err  = s.get("errors", 0)
            addr = s.get("addressable", s["universe"])
            lines.append(
                f"| {mkt} | {s['universe']:,} | {err} | {addr:,} | {s['prescreen_pass']} | "
                f"{s['data_fail']} | {s['filtered']} | {s['candidates']} |"
            )
        lines.append(
            f"| **TOTAL** | **{_STATS['universe_total']:,}** | **{total_errors}** | "
            f"**{total_addressable:,}** | **{_STATS['prescreen_pass']}** | "
            f"**{_STATS['data_fail']}** | **{_STATS['filtered']}** | **{_STATS['candidates']}** |"
        )

    return "\n".join(lines)


def _is_speculative(c: dict) -> bool:
    """Quality: RSI 40-70, ATR<=8%, RVOL<=10x. Outside those bounds = speculative."""
    return c.get("rsi", 50) > 70 or c.get("rsi", 50) < 30 or c.get("atr_pct", 0) > 8 or c.get("rvol", 1) > 10


def format_telegram(candidates: list) -> str:
    now = datetime.now(timezone.utc).strftime("%H:%M UTC")
    lines = [
        f"📊 <b>TOP MOVERS — {now}</b>",
        f"Market: {MARKET_CONTEXT.upper()} | Target: >{MIN_MOVE_PCT}% | "
        f"Universe: {_STATS['universe_total']:,} | Found: {len(candidates)}",
        "",
    ]

    def tier_sort(lst):
        """Sort: building trend (↑) first, then days desc, then score desc."""
        trend_rank = {"↑": 2, "→": 1, "↓": 0, "": -1}
        return sorted(
            lst,
            key=lambda x: (
                trend_rank.get(x.get("score_trend", ""), -1),
                x.get("repeat_days", 0),
                x["score"],
            ),
            reverse=True,
        )

    def fmt(c: dict, i: int) -> str:
        e         = "🟢" if c["day_return"] > 0 else "🔴"
        label     = {"etf": "ETF", "future": "FUT", "stock": "STK"}.get(c.get("tier", "stock"), "STK")
        rpt       = c.get("repeat_days", 0)
        trend     = c.get("score_trend", "")
        flag      = f" 🔁{rpt}d{trend}" if rpt >= 3 else ""
        earn      = " ⚠️EARN" if c.get("earnings_soon") else ""
        name      = get_ticker_name(c["ticker"])
        name_str  = f" <i>({name})</i>" if name else ""
        rs        = c.get("rs_vs_bench")
        rs_str    = f" RS{rs:+.1f}%" if rs is not None and c.get("tier") == "stock" else ""
        ma        = c.get("ma_align", "")
        ma_str    = f" {ma}" if ma else ""
        # Catalyst tier badge (A+/A/B) — additive signal
        ct        = c.get("catalyst_tier", "B")
        ct_badge  = {"A+": "🔥<b>A+</b>", "A": "🔵<b>A</b>", "B": "🟡B"}.get(ct, "🟡B")
        # SHORT flag — explicit when bias is SHORT so user knows direction
        short_tag = " 📉<b>SHORT</b>" if c.get("bias") == "SHORT" else ""
        return (
            f"{i}. {ct_badge}[{label}] <b>{c['ticker']}</b>{name_str}{flag}{earn}{short_tag} "
            f"{e}{c['day_return']:+.1f}%{rs_str}{ma_str} | "
            f"ATR {c['atr_pct']:.1f}% | RVOL {c['rvol']:.1f}x | Score <b>{c['score']:.0f}</b>"
        )

    etfs        = tier_sort([c for c in candidates if c.get("tier") == "etf"])
    futures     = tier_sort([c for c in candidates if c.get("tier") == "future"])
    stocks      = [c for c in candidates if c.get("tier", "stock") == "stock"]
    quality     = tier_sort([c for c in stocks if not _is_speculative(c)])
    speculative = tier_sort([c for c in stocks if     _is_speculative(c)])

    n = 1
    lines.append("🏦 <b>TOP ETFs</b>")
    for c in etfs[:2]:
        lines.append(fmt(c, n)); n += 1
    if not etfs:
        lines.append("  — none today")

    lines.append("")
    lines.append("📈 <b>TOP FUTURES</b>")
    for c in futures[:2]:
        lines.append(fmt(c, n)); n += 1
    if not futures:
        lines.append("  — none today")

    lines.append("")
    lines.append("📊 <b>TOP STOCKS — Quality</b>")
    for c in quality[:3]:
        lines.append(fmt(c, n)); n += 1
    if not quality:
        lines.append("  — none today")

    lines.append("")
    lines.append("⚡ <b>TOP STOCKS — Speculative</b>")
    for c in speculative[:3]:
        lines.append(fmt(c, n)); n += 1
    if not speculative:
        lines.append("  — none today")

    lines += ["", "⚠️ Research only. Not financial advice."]
    return "\n".join(lines)


def send_telegram(text: str) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
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


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
#  PERSISTENCE TRACKER — rolling 7-day history
# ═══════════════════════════════════════════════════════════════════════════════

def load_scan_history() -> dict:
    """Load rolling daily scan history. Returns empty structure if file missing."""
    try:
        with open(HISTORY_FILE, encoding="utf-8") as f:
            data = json.load(f)
            if "daily" not in data:
                return {"daily": {}}
            return data
    except (FileNotFoundError, json.JSONDecodeError):
        return {"daily": {}}


def save_scan_history(history: dict, candidates: list) -> None:
    """Merge today's candidates into history and keep a rolling 7-day window."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    existing = history["daily"].get(today, {})
    for c in candidates:
        existing[c["ticker"]] = {"tier": c.get("tier", "stock"), "score": c.get("score", 0)}
    history["daily"][today] = existing
    all_dates = sorted(history["daily"].keys(), reverse=True)
    history["daily"] = {d: history["daily"][d] for d in all_dates[:7]}
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Scan history saved — {len(history['daily'])} day(s) on record")


def compute_persistence(ticker: str, history: dict) -> dict:
    """
    Return how many distinct past days a ticker qualified and which direction
    its score is trending.

    trend:
      '↑'  score rose  >=5 pts from first to last recorded day  (momentum building)
      '↓'  score fell  >=5 pts                                   (momentum fading)
      '→'  score flat  within ±5 pts                             (sustained)
      ''   fewer than 2 past days — not enough data
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    past = []
    for date in sorted(history["daily"].keys()):
        if date == today:
            continue
        entry = history["daily"][date].get(ticker)
        if entry is None:
            continue
        # backward-compat: old entries stored just the tier string
        score = entry.get("score", 0) if isinstance(entry, dict) else 0
        past.append(score)

    days = len(past)
    if days < 2:
        trend = ""
    else:
        delta = past[-1] - past[0]
        trend = "↑" if delta >= 5 else ("↓" if delta <= -5 else "→")

    return {"days": days, "trend": trend}


# ── Point 3 + 4 — earnings flag and sector deduplication ─────────────────────

def enrich_candidates(candidates: list) -> None:
    """
    Parallel yfinance enrichment for ALL candidates.
    Adds fields in-place:
      earnings_soon        (bool)  — earnings within 5 days (stocks only)
      sector               (str)   — yfinance sector/industry string
      has_news_today       (bool)  — any news item published in the last 24 h
      news_headline        (str)   — title of the most recent news item (or "")
      short_interest_pct   (float) — short interest as % of float (stocks, Improvement 9)
      earnings_surprise_avg(float) — avg EPS surprise % last 4 quarters (stocks, Improvement 10)
      rs_vs_sector         (float) — day_return minus sector ETF return (stocks, Improvement 5)
    """
    if not candidates:
        return

    def _fetch(c):
        ticker_str = c["ticker"]
        is_stock   = c.get("tier") == "stock"
        try:
            tk   = yf.Ticker(ticker_str)
            info = tk.info

            # ── Earnings flag + sector (stocks only) ─────────────────────────
            if is_stock:
                ed = info.get("earningsDate") or info.get("earningsTimestamp")
                if ed:
                    if isinstance(ed, (int, float)):
                        ed_date = datetime.fromtimestamp(ed).date()
                    else:
                        ed_date = datetime.strptime(str(ed)[:10], "%Y-%m-%d").date()
                    days_to = (ed_date - datetime.now(timezone.utc).date()).days
                    c["earnings_soon"] = 0 <= days_to <= 5
                else:
                    c["earnings_soon"] = False
                c["sector"] = info.get("sector") or info.get("industry") or "Other"

                # ── Improvement 9: Short interest ─────────────────────────────
                si = info.get("shortPercentOfFloat")
                c["short_interest_pct"] = round(float(si) * 100, 1) if si else None

                # ── Improvement 10: Earnings surprise magnitude ───────────────
                try:
                    surprise = None
                    # Attempt 1: earningsHistory key inside info
                    eh = info.get("earningsHistory")
                    if eh and isinstance(eh, dict):
                        hist_list = eh.get("history", [])
                        surps = [
                            h.get("surprisePercent", 0) * 100
                            for h in hist_list[-4:]
                            if h.get("surprisePercent") is not None
                        ]
                        if surps:
                            surprise = round(sum(surps) / len(surps), 1)
                    # Attempt 2: quarterly_earnings DataFrame
                    if surprise is None:
                        q = tk.quarterly_earnings
                        if q is not None and not q.empty:
                            for act_col, est_col in [("Actual", "Estimate"), ("actual", "estimate")]:
                                if act_col in q.columns and est_col in q.columns:
                                    q2  = q[[act_col, est_col]].dropna()
                                    est = q2[est_col].abs().replace(0, float("nan"))
                                    surp = ((q2[act_col] - q2[est_col]) / est * 100).dropna()
                                    if len(surp) > 0:
                                        surprise = round(surp.head(4).mean(), 1)
                                    break
                    c["earnings_surprise_avg"] = surprise
                except Exception:
                    c["earnings_surprise_avg"] = None

                # ── Improvement 5: RS vs sector ETF ──────────────────────────
                sector_ret  = fetch_sector_etf_return(c["sector"])
                c["rs_vs_sector"] = round(c.get("day_return", 0.0) - sector_ret, 2)

            else:
                c["earnings_soon"]        = False
                c["sector"]               = c.get("tier", "other").upper()
                c["short_interest_pct"]   = None
                c["earnings_surprise_avg"]= None
                c["rs_vs_sector"]         = None

            # ── Improvement 1: News enrichment (all tiers) ───────────────────
            try:
                import time as _time
                news_items  = tk.news or []
                cutoff_ts   = _time.time() - 86400
                today_news  = [
                    n for n in news_items
                    if n.get("providerPublishTime", 0) >= cutoff_ts
                ]
                c["has_news_today"] = bool(today_news)
                c["news_headline"]  = today_news[0].get("title", "") if today_news else ""
            except Exception:
                c["has_news_today"] = False
                c["news_headline"]  = ""

        except Exception:
            c["earnings_soon"]        = False
            c["sector"]               = "Other" if is_stock else c.get("tier", "other").upper()
            c["has_news_today"]       = False
            c["news_headline"]        = ""
            c["short_interest_pct"]   = None
            c["earnings_surprise_avg"]= None
            c["rs_vs_sector"]         = None

    logger.info(f"  Enriching {len(candidates)} candidates (earnings + sector + news + SI + surprise + sector RS)...")
    with ThreadPoolExecutor(max_workers=10) as ex:
        futs = [ex.submit(_fetch, c) for c in candidates]
        for f in as_completed(futs):
            f.result()


def deduplicate_by_sector(candidates: list, max_per_sector: int = 2) -> list:
    """
    For stock candidates: keep top max_per_sector per sector per market.
    ETF and Futures candidates are always kept (already capped in scan_tier).
    Candidates must be pre-sorted by score descending.
    """
    counts  = {}
    result  = []
    for c in sorted(candidates, key=lambda x: x["score"], reverse=True):
        if c.get("tier") in ("etf", "future"):
            result.append(c)
            continue
        key = f"{c.get('market','?')}:{c.get('sector','Other')}"
        if counts.get(key, 0) < max_per_sector:
            counts[key] = counts.get(key, 0) + 1
            result.append(c)
    return sorted(result, key=lambda x: x["score"], reverse=True)


def classify_catalyst_tier(c: dict) -> str:
    """
    Classify each candidate into A+, A, or B.

    A+ — earnings+gap+volume | large surprise move | news+volume | short squeeze setup
    A  — structural breakout | strong momentum | news catalyst | strong earnings history
    B  — all other qualifying setups
    """
    has_earnings        = c.get("earnings_soon",       False)
    has_news_today      = c.get("has_news_today",      False)
    rvol                = c.get("rvol",                1.0)
    move_pct            = abs(c.get("day_return",      0.0))
    range_pos           = c.get("range_pos",           0.5)
    ma_align            = c.get("ma_align",            "")
    si_pct              = c.get("short_interest_pct")         # float or None (Improvement 9)
    earn_surp           = c.get("earnings_surprise_avg")      # float or None (Improvement 10)

    # A+ = earnings + significant gap + elevated volume
    if has_earnings and move_pct >= 2.0 and rvol >= 2.0:
        return "A+"
    # A+ = large surprise move + very high RVOL
    if move_pct >= 4.0 and rvol >= 3.0:
        return "A+"
    # A+ = breaking news + big move + volume surge
    if has_news_today and move_pct >= 2.0 and rvol >= 2.0:
        return "A+"
    # A+ = short squeeze setup: high short interest + near 52w high breakout + volume (Improvement 9)
    if si_pct and si_pct >= 15.0 and range_pos >= 0.85 and rvol >= 1.5:
        return "A+"
    # A+ = strong earnings beat history + upcoming earnings (Improvement 10)
    if has_earnings and earn_surp is not None and earn_surp >= 10.0:
        return "A+"
    # A = near 52-week high breakout + full trend + volume
    if range_pos >= 0.85 and rvol >= 1.5 and ma_align == "MA↑":
        return "A"
    # A = strong directional move + volume + trend
    if move_pct >= 2.0 and rvol >= 2.0 and ma_align in ("MA↑", "MA→"):
        return "A"
    # A = news today + elevated volume
    if has_news_today and rvol >= 1.5:
        return "A"
    # A = consistent earnings beats (positive history even without imminent earnings) (Improvement 10)
    if earn_surp is not None and earn_surp >= 10.0 and rvol >= 1.5:
        return "A"
    # B = everything else
    return "B"


def main():
    logger.info("=" * 60)
    logger.info("INTRADAY MOMENTUM SCANNER v6 — Dynamic Universe + ETF/Futures Tiers + 11 Markets")
    logger.info(f"Market:  {MARKET_CONTEXT.upper()}")
    logger.info(f"Target:  >{MIN_MOVE_PCT}%")
    logger.info(f"Mode:    {'DRY RUN' if DRY_RUN else 'LIVE'}")
    logger.info("=" * 60)

    # ── Load history and run scan ─────────────────────────────────────────────
    history    = load_scan_history()
    candidates = run_all_markets()

    # ── Enrich with earnings flag + sector (Points 3 & 4) ────────────────────
    enrich_candidates(candidates)
    before = len(candidates)
    candidates = deduplicate_by_sector(candidates, max_per_sector=2)
    logger.info(f"Sector dedup: {before} → {len(candidates)} candidates")

    # Tag each candidate with persistence data (days + score trend)
    for c in candidates:
        p = compute_persistence(c["ticker"], history)
        c["repeat_days"]  = p["days"]
        c["score_trend"]  = p["trend"]

    repeaters = [c for c in candidates if c["repeat_days"] >= 3]
    if repeaters:
        logger.info(
            f"Trend candidates (>=3 days): "
            + ", ".join(f"{c['ticker']}({c['repeat_days']}d{c['score_trend']})" for c in repeaters)
        )

    # ── Catalyst tier classification (A+/A/B) — additive, uses existing signals ─
    for c in candidates:
        c["catalyst_tier"] = classify_catalyst_tier(c)
    tier_counts = {}
    for c in candidates:
        t = c["catalyst_tier"]
        tier_counts[t] = tier_counts.get(t, 0) + 1
    logger.info(
        f"Catalyst tiers: A+={tier_counts.get('A+',0)} | "
        f"A={tier_counts.get('A',0)} | B={tier_counts.get('B',0)}"
    )

    save_scan_history(history, candidates)

    # ── Write scan_results.json for ta_runner.py ──────────────────────────────
    with open(OUTPUT_JSON, "w") as f:
        json.dump(candidates, f, indent=2)
    logger.info(f"scan_results.json written — {len(candidates)} candidate(s)")

    report     = format_markdown(candidates)
    timestamp  = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    filename   = f"scan_{MARKET_CONTEXT}_{timestamp}.md"

    with open(filename, "w") as f:
        f.write(report)
    logger.info(f"Report saved: {filename}")
    print("\n" + report)

    if not DRY_RUN:
        if candidates:
            send_telegram(format_telegram(candidates))
            send_telegram_doc(filename, f"Scan — {MARKET_CONTEXT.upper()} {timestamp}")
        else:
            send_telegram(
                f"📊 <b>Scanner — {datetime.now(timezone.utc).strftime('%H:%M UTC')}</b>\n"
                f"Market: {MARKET_CONTEXT.upper()} | Target: >{MIN_MOVE_PCT}%\n"
                f"Universe: {_STATS['universe_total']:,} stocks\n"
                "No candidates this scan."
            )

    logger.info(f"Complete — {len(candidates)} candidates from "
                f"{_STATS['universe_total']:,} stocks scanned")


if __name__ == "__main__":
    main()
