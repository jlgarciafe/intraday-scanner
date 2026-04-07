"""
═══════════════════════════════════════════════════════════════════════════════
  INTRADAY MOMENTUM SCANNER v4 — Dynamic Universe
  
  Architecture:
    Stage 1 — Fetch full index constituents dynamically (Wikipedia/free APIs)
    Stage 2 — Lightweight volume pre-screen across all constituents
              Only stocks with RVOL > 2x pass to full analysis
    Stage 3 — Full ATR + momentum analysis on pre-screened candidates
    Stage 4 — Filter, rank, alert

  Markets covered:
    US  — S&P 500 (500 stocks) via Wikipedia
    UK  — FTSE 100 (100 stocks) via Wikipedia  
    DE  — DAX 40 (40 stocks) via Wikipedia
    JP  — Nikkei 225 (225 stocks) via Wikipedia
    ES  — Ibex 35 (35 stocks) via Wikipedia
    FR  — CAC 40 (40 stocks) via Wikipedia
    HK  — Hang Seng (82 stocks) via Wikipedia

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

from datetime import datetime, timezone
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

# ── Pre-screen thresholds ─────────────────────────────────────────────────────
PRESCREEN_RVOL     = 1.2    # Minimum RVOL to pass pre-screen (lowered from 1.5)
PRESCREEN_ATR      = 2.5    # ATR% floor — pass if ATR >= this even when RVOL < 1.2x
                            # Gate logic: vol >= min AND (RVOL >= 1.2x OR ATR >= 2.5%)
PRESCREEN_MIN_VOL  = {      # Minimum absolute volume per market
    "us": 500_000,
    "uk": 100_000,
    "de": 50_000,
    "jp": 100_000,
    "es": 50_000,
    "hk": 100_000,
    "fr": 50_000,
}
MAX_PRESCREEN_PASS = 60     # Max stocks to run full analysis on per market

# ── Fee model ─────────────────────────────────────────────────────────────────
FEE_MODEL = {
    "us": {"commission_pct": 0.005, "spread_pct": 0.05,  "slippage_pct": 0.05},
    "uk": {"commission_pct": 0.100, "spread_pct": 0.10,  "slippage_pct": 0.10},
    "de": {"commission_pct": 0.100, "spread_pct": 0.10,  "slippage_pct": 0.10},
    "jp": {"commission_pct": 0.050, "spread_pct": 0.08,  "slippage_pct": 0.08},
    "es": {"commission_pct": 0.100, "spread_pct": 0.10,  "slippage_pct": 0.10},
    "hk": {"commission_pct": 0.080, "spread_pct": 0.10,  "slippage_pct": 0.10},
    "fr": {"commission_pct": 0.100, "spread_pct": 0.10,  "slippage_pct": 0.10},
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


# ═══════════════════════════════════════════════════════════════════════════════
#  STAGE 2 — VOLUME PRE-SCREEN
# ═══════════════════════════════════════════════════════════════════════════════

def flatten_df(raw: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance v0.2+ MultiIndex columns."""
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    return raw.loc[:, ~raw.columns.duplicated()]


def prescreen_volume(tickers: list, market_key: str) -> list:
    """
    Lightweight volume pre-screen across all index constituents.
    Downloads 5 days of data per ticker individually (batch MultiIndex is unreliable).
    Returns tickers sorted by RVOL descending, capped at MAX_PRESCREEN_PASS.
    RVOL > PRESCREEN_RVOL = elevated interest = worth full analysis.
    """
    min_vol  = PRESCREEN_MIN_VOL.get(market_key, 50_000)
    screened = []

    logger.info(f"  Pre-screening {len(tickers)} tickers for RVOL≥{PRESCREEN_RVOL}x...")

    errors = 0
    for i, ticker in enumerate(tickers):
        try:
            raw = yf.download(
                ticker,
                period="5d",
                interval="1d",
                auto_adjust=True,
                progress=False,
                timeout=10,
            )
            if raw is None or raw.empty or len(raw) < 3:
                errors += 1
                continue

            df = flatten_df(raw)
            if "Volume" not in df.columns:
                errors += 1
                continue

            vols      = df["Volume"].astype(float).values
            today_vol = float(vols[-1])
            avg_vol   = float(vols[:-1].mean()) if len(vols) > 1 else today_vol
            rvol      = today_vol / avg_vol if avg_vol > 0 else 1.0

            # Compute lightweight ATR% for the OR condition
            atr_pct = 0.0
            if "High" in df.columns and "Low" in df.columns and "Close" in df.columns:
                h = df["High"].astype(float).values
                l = df["Low"].astype(float).values
                c = df["Close"].astype(float).values
                if len(c) >= 2 and c[-1] > 0:
                    trs = [max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
                           for i in range(1, len(c))]
                    atr_pct = (sum(trs) / len(trs)) / c[-1] * 100

            # Gate: volume floor always required, then RVOL OR ATR
            passes_rvol = rvol >= PRESCREEN_RVOL
            passes_atr  = atr_pct >= PRESCREEN_ATR
            if today_vol >= min_vol and (passes_rvol or passes_atr):
                screened.append({
                    "ticker":  ticker,
                    "rvol":    round(rvol, 2),
                    "atr_pct": round(atr_pct, 2),
                    "volume":  int(today_vol),
                    "passed_by": "RVOL" if passes_rvol else "ATR",
                })

        except Exception as e:
            errors += 1
            logger.debug(f"    {ticker}: prescreen error — {e}")

        # Progress log every 50 tickers
        if (i + 1) % 50 == 0:
            logger.info(f"    Progress: {i+1}/{len(tickers)} checked | errors: {errors} | passed: {len(screened)}")

        time.sleep(0.15)

    _STATS["errors"] += errors

    screened.sort(key=lambda x: x["rvol"], reverse=True)

    if screened:
        by_rvol = sum(1 for s in screened if s["passed_by"] == "RVOL")
        by_atr  = sum(1 for s in screened if s["passed_by"] == "ATR")
        logger.info(
            f"  Pre-screen: {len(screened)} passed from {len(tickers)} "
            f"(RVOL≥{PRESCREEN_RVOL}x: {by_rvol} | ATR≥{PRESCREEN_ATR}%: {by_atr}) "
            f"| Top: {screened[0]['ticker']} RVOL={screened[0]['rvol']}x ATR={screened[0]['atr_pct']}%"
        )
    else:
        logger.info(f"  Pre-screen: 0 passed from {len(tickers)} — market may be closed or inactive")

    if len(screened) > MAX_PRESCREEN_PASS:
        logger.info(f"  Capping at top {MAX_PRESCREEN_PASS} by RVOL for full analysis")
        screened = screened[:MAX_PRESCREEN_PASS]

    return [s["ticker"] for s in screened]


# ═══════════════════════════════════════════════════════════════════════════════
#  STAGE 3 — FULL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_snapshot(ticker: str) -> Optional[pd.DataFrame]:
    """Fetch 30 days daily OHLCV for full analysis."""
    for period in ["30d", "1mo"]:
        try:
            raw = yf.download(
                ticker, period=period, interval="1d",
                auto_adjust=True, progress=False, timeout=15,
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
        time.sleep(0.3)
    return None


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

        score = (
            min(atr_pct/5.0, 1.0)*30 +
            min(rvol/3.0, 1.0)*25 +
            min(abs(day_return)/3.0, 1.0)*25 +
            (1.0 if 60<=rsi<=80 else 0.6 if 40<=rsi<60 else 0.4)*20
        )

        logger.info(
            f"    {ticker}: price={c[-1]:.2f} | ATR={atr_pct:.2f}% | "
            f"RVOL={rvol:.2f}x | RSI={rsi:.0f} | day={day_return:+.2f}% | "
            f"range={day_range:.2f}% | vol={int(v[-1]):,} | score={score:.0f}"
        )

        return {
            "ticker": ticker, "price": round(float(c[-1]),4),
            "atr_pct": round(atr_pct,2), "rvol": round(rvol,2),
            "rsi": round(rsi,1), "day_return": round(day_return,2),
            "day_range": round(day_range,2), "volume": int(v[-1]),
            "score": round(score,1),
        }
    except Exception as e:
        logger.info(f"    {ticker}: analysis error — {e}")
        return {}


def passes_filters(m: dict, market: str) -> tuple:
    """Volume and RVOL always enforced. ATR/score only when target > 0."""
    vol  = m.get("volume", 0)
    rvol = m.get("rvol", 0)
    atr  = m.get("atr_pct", 0)
    score= m.get("score", 0)

    min_vol = PRESCREEN_MIN_VOL.get(market, 50_000)
    if vol < min_vol:
        return False, f"vol {vol:,} < {min_vol:,}"
    if rvol < 1.3:
        return False, f"RVOL {rvol:.2f}x < 1.3x"

    if MIN_MOVE_PCT > 0:
        if atr < MIN_MOVE_PCT * 0.7:
            return False, f"ATR {atr:.2f}% < {MIN_MOVE_PCT*0.7:.2f}%"
        if score < 35:
            return False, f"score {score:.0f} < 35"
    else:
        if score < 20:
            return False, f"score {score:.0f} < 20"

    return True, "passed"


def net_yield(atr_pct: float, market: str) -> float:
    f = FEE_MODEL.get(market, FEE_MODEL["us"])
    cost = (f["commission_pct"] + f["spread_pct"] + f["slippage_pct"]) * 2
    return round(atr_pct - cost, 2)


# ═══════════════════════════════════════════════════════════════════════════════
#  STAGE 4 — ORCHESTRATION
# ═══════════════════════════════════════════════════════════════════════════════

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

    logger.info(f"\n  Full analysis on {len(prescreened)} pre-screened stocks:")

    # Stage 3: Full analysis
    candidates, data_fail, filtered = [], 0, 0

    for i, ticker in enumerate(prescreened, 1):
        logger.info(f"  [{i}/{len(prescreened)}] {ticker}")
        df = fetch_snapshot(ticker)
        if df is None:
            data_fail += 1
            continue

        m = analyse_ticker(ticker, df)
        if not m:
            data_fail += 1
            continue

        passed, reason = passes_filters(m, market_key)
        if not passed:
            logger.info(f"    {ticker}: ❌ {reason}")
            filtered += 1
            continue

        logger.info(f"    {ticker}: ✅ CANDIDATE")
        candidates.append({
            **m,
            "market":    market_key.upper(),
            "net_yield": net_yield(m["atr_pct"], market_key),
        })
        time.sleep(0.2)

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
        markets = ["us", "uk", "de", "jp", "es", "hk", "fr"]
    elif MARKET_CONTEXT == "eu":
        markets = ["uk", "de", "es", "fr"]
    elif MARKET_CONTEXT == "asia":
        markets = ["jp", "hk"]
    else:
        markets = [MARKET_CONTEXT] if MARKET_CONTEXT in FEE_MODEL else ["us"]

    all_c = []
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
            f"**Universe scanned:** {_STATS['universe_total']:,} stocks  |  "
            f"**Pre-screened:** {_STATS['prescreen_pass']}  |  **Candidates:** 0\n\n"
            "**No candidates met the filter criteria.**\n\n"
            "> ⚠️ Research only. Not financial advice.\n"
        )

    lines = [
        f"## 📊 Intraday Scanner — {now}",
        f"**Market:** {MARKET_CONTEXT.upper()}  |  **Target:** >{MIN_MOVE_PCT}%  |  **Candidates:** {len(candidates)}",
        f"**Universe:** {_STATS['universe_total']:,} stocks scanned  |  "
        f"**Pre-screen pass:** {_STATS['prescreen_pass']}  |  "
        f"**Final candidates:** {_STATS['candidates']}",
        "",
        "> ⚠️ Research only. Not financial advice. Past ATR ≠ future moves.",
        "",
        "| # | Ticker | Mkt | Price | Day% | Range% | ATR% | RVOL | RSI | Score | Net |",
        "|---|--------|-----|-------|------|--------|------|------|-----|-------|-----|",
    ]
    for i, c in enumerate(candidates[:30], 1):
        e = "🟢" if c["day_return"] > 0 else "🔴"
        n = "✅" if c["net_yield"] > 0 else "❌"
        lines.append(
            f"| {i} | **{c['ticker']}** | {c['market']} | {c['price']:.2f} | "
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


def format_telegram(candidates: list) -> str:
    now = datetime.now(timezone.utc).strftime("%H:%M UTC")
    lines = [
        f"📊 <b>Intraday Scanner — {now}</b>",
        f"Market: {MARKET_CONTEXT.upper()} | Target: >{MIN_MOVE_PCT}% | "
        f"Universe: {_STATS['universe_total']:,} | Found: {len(candidates)}",
        "",
    ]
    for i, c in enumerate(candidates[:10], 1):
        e = "🟢" if c["day_return"] > 0 else "🔴"
        lines.append(
            f"{i}. <b>{c['ticker']}</b> {e}{c['day_return']:+.1f}% | "
            f"ATR {c['atr_pct']:.1f}% | RVOL {c['rvol']:.1f}x | Score <b>{c['score']:.0f}</b>"
        )
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

def main():
    logger.info("=" * 60)
    logger.info("INTRADAY MOMENTUM SCANNER v4 — Dynamic Universe")
    logger.info(f"Market:  {MARKET_CONTEXT.upper()}")
    logger.info(f"Target:  >{MIN_MOVE_PCT}%")
    logger.info(f"Mode:    {'DRY RUN' if DRY_RUN else 'LIVE'}")
    logger.info("=" * 60)

    candidates = run_all_markets()

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
