# -*- coding: utf-8 -*-
"""KRX stock-name resolution for Korean-language news search.

KRX tickers reach the analysis pipeline as Yahoo Finance suffix symbols
(``005930.KS`` / ``247540.KQ``) and yfinance usually returns an English company
name (``317330.KQ`` -> ``Duksan Tecopia``). Korean financial media publish a
company under its Hangul name, so English-only queries cannot retrieve Korean
news. This module resolves the Hangul name once per KRX base code and caches
it, so query construction and relevance scoring can use the name Korean
readers actually see.

Design boundaries:

- Only ``.KS`` / ``.KQ`` symbols are looked up; A-share / HK / US symbols are
  never touched.
- A caller-supplied name that already contains Hangul always wins, so no
  network call is made when the pipeline already knows the Korean name.
- Lookup failures fail open: callers keep their existing name and query path.
- Lookups are cached per base code for the process lifetime, including
  negative results, so a blocked/unreachable endpoint costs one timeout per
  code rather than one per search dimension.
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Dict, Optional

import requests

from src.services.market_symbol_utils import get_suffix_market, split_suffix_symbol

logger = logging.getLogger(__name__)

# Naver Finance mobile quote endpoint: unauthenticated JSON that carries
# ``itemCode`` and ``stockName``. Used only to read the Hangul display name.
NAVER_STOCK_BASIC_URL = "https://m.stock.naver.com/api/stock/{code}/basic"
NAVER_LOOKUP_TIMEOUT_SECONDS = 3

_HANGUL_RE = re.compile(r"[\uac00-\ud7a3]")
# Keep only plausible company names: Hangul plus the punctuation Korean
# company names actually use. Anything else (HTML, error copy, JSON blobs) is
# rejected so a bad response never becomes a search query.
_PLAUSIBLE_NAME_RE = re.compile(r"^[0-9A-Za-z&+.\-()\s\uac00-\ud7a3]{1,40}$")

_NAME_CACHE: Dict[str, Optional[str]] = {}
_NAME_CACHE_LOCK = threading.Lock()


def contains_hangul(value: Optional[str]) -> bool:
    """Return True when the value contains at least one Hangul syllable."""
    return bool(value and _HANGUL_RE.search(value))


def is_korean_stock_code(stock_code: Optional[str]) -> bool:
    """Return True for KOSPI/KOSDAQ suffix symbols (``.KS`` / ``.KQ``)."""
    return get_suffix_market(stock_code) == "kr"


def korean_stock_base_code(stock_code: Optional[str]) -> Optional[str]:
    """Return the 6-digit KRX base code for a ``.KS``/``.KQ`` symbol."""
    if not is_korean_stock_code(stock_code):
        return None
    parts = split_suffix_symbol(stock_code or "")
    if parts is None:
        return None
    base, _suffix = parts
    return base or None


def resolve_korean_stock_name(
    stock_code: Optional[str],
    stock_name: Optional[str] = "",
    *,
    allow_network: bool = True,
) -> Optional[str]:
    """Return the Hangul company name for a KOSPI/KOSDAQ symbol, if known.

    Args:
        stock_code: Symbol such as ``317330.KQ``.
        stock_name: Name the pipeline already holds (may already be Hangul).
        allow_network: When False, only the caller-supplied name and the cache
            are used. Tests use this to stay offline.

    Returns:
        The Hangul name, or None when the code is not Korean or no Hangul name
        is known.
    """
    base_code = korean_stock_base_code(stock_code)
    if not base_code:
        return None

    supplied = (stock_name or "").strip()
    if contains_hangul(supplied):
        return supplied

    with _NAME_CACHE_LOCK:
        if base_code in _NAME_CACHE:
            return _NAME_CACHE[base_code]

    resolved = _fetch_naver_korean_name(base_code) if allow_network else None

    with _NAME_CACHE_LOCK:
        _NAME_CACHE[base_code] = resolved
    return resolved


def _fetch_naver_korean_name(base_code: str) -> Optional[str]:
    """Read the Hangul display name from Naver Finance; never raises."""
    url = NAVER_STOCK_BASIC_URL.format(code=base_code)
    try:
        response = requests.get(
            url,
            timeout=NAVER_LOOKUP_TIMEOUT_SECONDS,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; stock-news-search/1.0)",
                "Accept": "application/json",
            },
        )
        if response.status_code != 200:
            logger.info(
                "[韩股名称] %s 조회 실패: HTTP %s", base_code, response.status_code
            )
            return None
        payload = response.json()
    except Exception as exc:  # fail open: queries keep the English name
        logger.info("[韩股名称] %s 조회 예외: %s", base_code, exc)
        return None

    if not isinstance(payload, dict):
        return None

    item_code = str(payload.get("itemCode") or "").strip()
    if item_code and item_code != base_code:
        logger.warning(
            "[韩股名称] %s 응답 코드 불일치(itemCode=%s), 무시", base_code, item_code
        )
        return None

    name = str(payload.get("stockName") or "").strip()
    if not name or not contains_hangul(name) or not _PLAUSIBLE_NAME_RE.match(name):
        logger.info("[韩股名称] %s 응답 이름이 사용 불가: %r", base_code, name)
        return None

    logger.info("[韩股名称] %s -> %s", base_code, name)
    return name


def clear_korean_name_cache_for_tests() -> None:
    """Reset the process-wide name cache (tests only)."""
    with _NAME_CACHE_LOCK:
        _NAME_CACHE.clear()
