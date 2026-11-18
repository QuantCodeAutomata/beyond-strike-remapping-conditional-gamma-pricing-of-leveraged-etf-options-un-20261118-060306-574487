"""Listed QQQ/TQQQ option market data for experiment 4 (paper Section 13.5.1).

Reference OHLCV bars are not used by experiments 1-3 and 5, which build their
Dupire surfaces from the parametric SVI construction in
``src.dupire_surface`` (see ``src/config.py`` for the documented reason).
Experiment 4 instead needs listed American-style option quotes for QQQ (the
underlying) and TQQQ (the three-times leveraged product) on three market
dates, which are not OHLCV bars: they are loaded directly from the Massive
REST API (options reference contracts and historical quotes), per the
"massive-rest-market-data" data-access rules. ``MASSIVE_API_KEY`` is read from
the environment with ``python-dotenv`` and used only in the Authorization
header of these requests.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

MASSIVE_API_ROOT = "https://api.massive.com"
CACHE_DIR = Path(__file__).resolve().parent / "cache"  # never committed, see .gitignore
RETRY_STATUSES = {429, 503, 504}


class MassiveDataError(RuntimeError):
    """A failed Massive REST request or an empty result where data was expected."""


def _api_key() -> str:
    load_dotenv()
    key = os.environ.get("MASSIVE_API_KEY")
    if not key:
        raise MassiveDataError("MASSIVE_API_KEY must be set in the environment")
    return key


def _get(path: str, params: dict) -> dict:
    params = {**params, "apiKey": _api_key()}
    for attempt in range(4):
        response = requests.get(f"{MASSIVE_API_ROOT}{path}", params=params, timeout=60)
        if response.status_code not in RETRY_STATUSES or attempt == 3:
            break
        time.sleep(float(response.headers.get("Retry-After") or 2 * 2 ** attempt))
    if response.status_code != 200:
        raise MassiveDataError(f"GET {path}: HTTP {response.status_code}: {response.text[:300]}")
    return response.json()


def _paginate(path: str, params: dict, max_pages: int = 20) -> list[dict]:
    """Follow Massive's ``next_url`` until absent; stop on a repeated URL."""
    out: list[dict] = []
    payload = _get(path, params)
    out.extend(payload.get("results") or [])
    next_url = payload.get("next_url")
    seen = set()
    pages = 1
    while next_url and next_url not in seen and pages < max_pages:
        seen.add(next_url)
        for attempt in range(4):
            response = requests.get(next_url, params={"apiKey": _api_key()}, timeout=60)
            if response.status_code not in RETRY_STATUSES or attempt == 3:
                break
            time.sleep(float(response.headers.get("Retry-After") or 2 * 2 ** attempt))
        if response.status_code != 200:
            raise MassiveDataError(f"GET next_url: HTTP {response.status_code}: {response.text[:300]}")
        payload = response.json()
        out.extend(payload.get("results") or [])
        next_url = payload.get("next_url")
        pages += 1
    return out


def list_option_contracts(underlying_ticker: str, as_of_date: str, expiry_min: str,
                           expiry_max: str, contract_type: str) -> pd.DataFrame:
    """Reference option contracts for ``underlying_ticker`` expiring in
    [expiry_min, expiry_max], as known from the ``as_of_date`` reference
    snapshot (``GET /v3/reference/options/contracts``). Past expiries require
    ``expired=true``; this follows ``next_url`` pagination."""
    # The Massive reference-contracts endpoint returns an empty list when
    # ``as_of`` is combined with an ``expiration_date`` range filter; the
    # ``expired=true`` flag together with the expiration-date range already
    # restricts the result to contracts that existed on ``as_of_date``.
    rows = _paginate("/v3/reference/options/contracts", {
        "underlying_ticker": underlying_ticker,
        "expired": "true",
        "expiration_date.gte": expiry_min,
        "expiration_date.lte": expiry_max,
        "contract_type": contract_type,
        "limit": 1000,
    })
    if not rows:
        raise MassiveDataError(
            f"no option contracts for {underlying_ticker} in [{expiry_min}, {expiry_max}]"
        )
    frame = pd.DataFrame(rows)
    required = {"ticker", "strike_price", "expiration_date", "contract_type", "exercise_style"}
    missing = required - set(frame.columns)
    if missing:
        raise MassiveDataError(f"option contracts response missing fields {missing}")
    return frame


def quote_at_or_before(option_ticker: str, as_of_date: str) -> dict | None:
    """Most recent NBBO quote at or before the close of ``as_of_date``
    (``GET /v3/quotes/{optionsTicker}``), bid/ask in USD. Returns ``None`` when
    the contract has no recorded quote in that window (illiquid/far strike)."""
    cutoff = datetime.strptime(as_of_date, "%Y-%m-%d").replace(
        hour=23, minute=59, second=59, tzinfo=timezone.utc
    )
    timestamp_ns = int(cutoff.timestamp() * 1e9)
    payload = _get(f"/v3/quotes/{option_ticker}", {
        "timestamp.lte": timestamp_ns, "order": "desc", "limit": 1, "sort": "timestamp",
    })
    results = payload.get("results") or []
    if not results:
        return None
    quote = results[0]
    bid, ask = quote.get("bid_price"), quote.get("ask_price")
    if bid is None or ask is None or bid <= 0 or ask <= 0 or ask < bid:
        return None
    return {"bid": float(bid), "ask": float(ask), "mid": 0.5 * (bid + ask),
            "quote_timestamp": quote.get("sip_timestamp")}


def _nearest_expiry(contracts: pd.DataFrame, as_of_date: str, target_months: float) -> str:
    """Nearest *standard monthly* expiry to ``target_months`` after
    ``as_of_date``. Monthly expiries list far more strikes than ad hoc
    weeklies, so among the five calendar expiries closest to the target we
    keep the one with the most listed contracts."""
    target = datetime.strptime(as_of_date, "%Y-%m-%d") + timedelta(days=30.4375 * target_months)
    counts = contracts["expiration_date"].value_counts()
    expiries = pd.to_datetime(counts.index)
    ranked = sorted(zip(expiries, counts.values), key=lambda item: abs((item[0] - target).total_seconds()))
    candidates = ranked[:5]
    best = max(candidates, key=lambda item: item[1])
    return best[0].strftime("%Y-%m-%d")


@dataclass
class OptionChainSnapshot:
    ticker: str
    as_of_date: str
    expiry: str
    target_months: float
    quotes: pd.DataFrame  # columns: strike, contract_type, bid, ask, mid


def load_option_chain(ticker: str, as_of_date: str, target_months: float,
                       moneyness_band: tuple[float, float], spot: float,
                       use_cache: bool = True) -> OptionChainSnapshot:
    """Build a two-sided (call+put) listed option chain snapshot for ``ticker``
    nearest to ``target_months`` months after ``as_of_date``, restricted to
    strikes within ``moneyness_band`` of ``spot``, with bid/ask quotes as of
    the close of ``as_of_date``."""
    cache_path = CACHE_DIR / f"chain_{ticker}_{as_of_date}_{target_months}.csv"
    meta_path = CACHE_DIR / f"chain_{ticker}_{as_of_date}_{target_months}.json"
    if use_cache and cache_path.exists() and meta_path.exists():
        import json
        meta = json.loads(meta_path.read_text())
        return OptionChainSnapshot(ticker=ticker, as_of_date=as_of_date, expiry=meta["expiry"],
                                    target_months=target_months, quotes=pd.read_csv(cache_path))

    expiry_min = as_of_date
    expiry_max = (datetime.strptime(as_of_date, "%Y-%m-%d") + timedelta(days=750)).strftime("%Y-%m-%d")
    calls = list_option_contracts(ticker, as_of_date, expiry_min, expiry_max, "call")
    expiry = _nearest_expiry(calls, as_of_date, target_months)
    calls = calls[calls["expiration_date"] == expiry]
    puts = list_option_contracts(ticker, as_of_date, expiry_min, expiry_max, "put")
    puts = puts[puts["expiration_date"] == expiry]

    lo, hi = moneyness_band
    rows = []
    for frame, side in ((calls, "call"), (puts, "put")):
        band = frame[(frame["strike_price"] >= lo * spot) & (frame["strike_price"] <= hi * spot)]
        for _, contract in band.iterrows():
            quote = quote_at_or_before(contract["ticker"], as_of_date)
            if quote is None:
                continue
            rows.append({"strike": float(contract["strike_price"]), "contract_type": side, **quote})
    quotes = pd.DataFrame(rows)
    if quotes.empty:
        raise MassiveDataError(f"no quoted {ticker} contracts found for expiry {expiry}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    quotes.to_csv(cache_path, index=False)
    import json
    meta_path.write_text(json.dumps({"expiry": expiry, "n_quotes": len(quotes)}))
    return OptionChainSnapshot(ticker=ticker, as_of_date=as_of_date, expiry=expiry,
                                target_months=target_months, quotes=quotes)
