# -*- coding: utf-8 -*-
"""Regression tests for Korean (KOSPI/KOSDAQ) news search localization.

KRX symbols arrive as ``.KS`` / ``.KQ`` codes with an English yfinance name.
Before this fix they fell through to the Chinese A-share query templates
(``减持 处罚 ... 上交所 深交所 cninfo``), so providers returned unrelated
foreign pages and the relevance admission filter dropped every result.
"""

import sys
from datetime import datetime
from unittest.mock import MagicMock, patch

# Mock newspaper before search_service import (optional dependency)
if "newspaper" not in sys.modules:
    mock_np = MagicMock()
    mock_np.Article = MagicMock()
    mock_np.Config = MagicMock()
    sys.modules["newspaper"] = mock_np

import pytest

from src.search_service import SearchResponse, SearchResult, SearchService
from src.services import korean_stock_names

KR_CODE = "317330.KQ"
KR_EN_NAME = "Duksan Tecopia Co., Ltd."
KR_KO_NAME = "덕산테코피아"


def _result(
    title: str,
    published_date: str | None,
    *,
    snippet: str = "",
    url: str | None = None,
    source: str = "example.co.kr",
) -> SearchResult:
    return SearchResult(
        title=title,
        snippet=snippet or title,
        url=url or f"https://example.co.kr/{abs(hash(title))}",
        source=source,
        published_date=published_date,
    )


def _response(results) -> SearchResponse:
    return SearchResponse(
        query="test",
        results=list(results),
        provider="Mock",
        success=True,
    )


@pytest.fixture(autouse=True)
def _reset_korean_name_cache():
    korean_stock_names.clear_korean_name_cache_for_tests()
    yield
    korean_stock_names.clear_korean_name_cache_for_tests()


def _service_with_mock_provider(response: SearchResponse | None = None):
    service = SearchService(
        bocha_keys=["dummy_key"],
        searxng_public_instances_enabled=False,
        news_max_age_days=3,
        news_strategy_profile="short",
    )
    mock_search = MagicMock(return_value=response or _response([]))
    service._providers[0].search = mock_search
    return service, mock_search


def test_korean_symbol_identity_terms_do_not_become_a_share_terms() -> None:
    terms = SearchService._stock_code_identity_terms(KR_CODE)

    assert KR_CODE in terms
    assert "317330" in terms
    assert "A317330" in terms
    assert not any(term.endswith((".SH", ".SZ")) for term in terms)
    # A-share identity behavior is untouched.
    assert "600519.SH" in SearchService._stock_code_identity_terms("600519")


def test_korean_stock_news_query_is_korean() -> None:
    today = datetime.now().date().isoformat()
    service, mock_search = _service_with_mock_provider(
        _response([_result(f"{KR_KO_NAME} 주가 약세 지속", today)])
    )

    with patch.object(
        korean_stock_names, "_fetch_naver_korean_name", return_value=KR_KO_NAME
    ):
        response = service.search_stock_news(KR_CODE, KR_EN_NAME, max_results=5)

    query = mock_search.call_args.args[0]
    assert KR_KO_NAME in query
    assert "317330" in query
    assert "股票" not in query
    assert "减持" not in query
    assert "cninfo" not in query
    assert response.success and response.results


def test_comprehensive_intel_uses_korean_dimension_queries() -> None:
    today = datetime.now().date().isoformat()
    service, mock_search = _service_with_mock_provider(
        _response([_result(f"{KR_KO_NAME} 공시 발표", today)])
    )

    with patch.object(
        korean_stock_names, "_fetch_naver_korean_name", return_value=KR_KO_NAME
    ):
        service.search_comprehensive_intel(KR_CODE, KR_EN_NAME, max_searches=6)

    queries = [call.args[0] for call in mock_search.call_args_list]
    assert len(queries) == 6
    for query in queries:
        assert KR_KO_NAME in query
        for a_share_term in ("减持", "上交所", "深交所", "cninfo", "研报"):
            assert a_share_term not in query
    assert any("공시" in query for query in queries)
    assert any("증권사 리포트" in query for query in queries)


def test_korean_headline_scores_as_direct_company_news() -> None:
    today = datetime.now().date().isoformat()
    item = _result(
        f"{KR_KO_NAME} 주가 약세 지속, 8,130원선 하회",
        today,
        snippet="코스닥 상장사 덕산테코피아가 하락 흐름을 이어갔다.",
    )

    with patch.object(
        korean_stock_names, "_fetch_naver_korean_name", return_value=KR_KO_NAME
    ):
        scored = SearchService._score_news_relevance(
            item, stock_code=KR_CODE, stock_name=KR_EN_NAME
        )

    assert scored.relevance_category == SearchService._DIRECT_NEWS_CATEGORY
    assert (scored.relevance_score or 0) >= 45
    assert any("韩文公司名" in reason for reason in scored.relevance_reasons or [])


def test_korean_article_survives_zero_relevance_admission() -> None:
    today = datetime.now().date().isoformat()
    korean_item = _result(f"{KR_KO_NAME} 실적 발표", today, snippet="영업이익 개선")
    unrelated = _result(
        "Kindle paperwhite review", today, snippet="A gadget blog post."
    )

    with patch.object(
        korean_stock_names, "_fetch_naver_korean_name", return_value=KR_KO_NAME
    ):
        ranked = SearchService._rank_news_response(
            _response([unrelated, korean_item]),
            stock_code=KR_CODE,
            stock_name=KR_EN_NAME,
            prefer_chinese=False,
            max_results=10,
            log_scope="test:kr:rank",
        )
        admitted = SearchService._filter_ranked_news_for_context(
            ranked, log_scope="test:kr:admission"
        )

    titles = [item.title for item in admitted.results]
    assert korean_item.title in titles
    assert unrelated.title not in titles


def test_resolve_prefers_supplied_hangul_name_without_lookup() -> None:
    with patch.object(korean_stock_names, "_fetch_naver_korean_name") as fetch:
        assert (
            korean_stock_names.resolve_korean_stock_name("005930.KS", "삼성전자")
            == "삼성전자"
        )
        fetch.assert_not_called()


def test_resolve_looks_up_english_named_korean_stock_once() -> None:
    with patch.object(
        korean_stock_names, "_fetch_naver_korean_name", return_value=KR_KO_NAME
    ) as fetch:
        assert (
            korean_stock_names.resolve_korean_stock_name(KR_CODE, KR_EN_NAME)
            == KR_KO_NAME
        )
        assert (
            korean_stock_names.resolve_korean_stock_name(KR_CODE, KR_EN_NAME)
            == KR_KO_NAME
        )
        assert fetch.call_count == 1


def test_resolve_ignores_non_korean_symbols_and_offline_mode() -> None:
    with patch.object(korean_stock_names, "_fetch_naver_korean_name") as fetch:
        assert korean_stock_names.resolve_korean_stock_name("600519", "贵州茅台") is None
        assert (
            korean_stock_names.resolve_korean_stock_name(
                KR_CODE, KR_EN_NAME, allow_network=False
            )
            is None
        )
        fetch.assert_not_called()


def _naver_stub(payload=None, *, status_code: int = 200, raises: Exception | None = None):
    response = MagicMock()
    response.status_code = status_code
    response.json = MagicMock(return_value=payload)
    get = MagicMock(side_effect=raises) if raises else MagicMock(return_value=response)
    return get


def test_naver_lookup_returns_hangul_name() -> None:
    get = _naver_stub({"itemCode": "317330", "stockName": KR_KO_NAME})
    with patch.object(korean_stock_names.requests, "get", get):
        assert korean_stock_names._fetch_naver_korean_name("317330") == KR_KO_NAME


def test_naver_lookup_rejects_mismatched_or_unusable_payloads() -> None:
    mismatched = _naver_stub({"itemCode": "005930", "stockName": KR_KO_NAME})
    no_hangul = _naver_stub({"itemCode": "317330", "stockName": "Duksan"})
    http_error = _naver_stub({"itemCode": "317330", "stockName": KR_KO_NAME}, status_code=500)
    broken = _naver_stub(raises=ValueError("boom"))

    for stub in (mismatched, no_hangul, http_error, broken):
        with patch.object(korean_stock_names.requests, "get", stub):
            assert korean_stock_names._fetch_naver_korean_name("317330") is None


def test_brave_locale_follows_korean_market() -> None:
    assert SearchService._brave_search_locale(KR_CODE, prefer_chinese=False) == {
        "search_lang": "ko",
        "country": "KR",
    }
    assert SearchService._brave_search_locale("600519", prefer_chinese=True) == {
        "search_lang": "zh-hans",
        "country": "CN",
    }


def test_stock_events_use_korean_event_terms() -> None:
    service, mock_search = _service_with_mock_provider(
        _response([_result(f"{KR_KO_NAME} 유상증자 결정", datetime.now().date().isoformat())])
    )

    with patch.object(
        korean_stock_names, "_fetch_naver_korean_name", return_value=KR_KO_NAME
    ):
        service.search_stock_events(KR_CODE, KR_EN_NAME)

    query = mock_search.call_args.args[0]
    assert "실적 발표" in query
    assert "减持公告" not in query


def test_price_fallback_uses_korean_keyword_templates() -> None:
    service, mock_search = _service_with_mock_provider(
        _response([_result(f"{KR_KO_NAME} 주가", None, snippet="8,020원")])
    )

    with patch.object(
        korean_stock_names, "_fetch_naver_korean_name", return_value=KR_KO_NAME
    ):
        response = service.search_stock_price_fallback(KR_CODE, KR_EN_NAME, max_attempts=2)

    queries = [call.args[0] for call in mock_search.call_args_list]
    assert queries
    assert all(KR_KO_NAME in query for query in queries)
    assert all("股价" not in query for query in queries)
    assert response.success


def test_a_share_news_query_still_chinese_preferred() -> None:
    today = datetime.now().date().isoformat()
    service, mock_search = _service_with_mock_provider(
        _response([_result("贵州茅台 最新消息", today)])
    )

    service.search_stock_news("600519", "贵州茅台", max_results=5)

    query = mock_search.call_args.args[0]
    assert "股票" in query
    assert "주가" not in query
