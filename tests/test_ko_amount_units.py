# -*- coding: utf-8 -*-
"""Korean-report amount units (억/만 원) tests.

A Korean-language report (``REPORT_LANGUAGE=ko``) or a KRX stock (``.KS``/``.KQ``,
whose financial statements are reported in ``KRW``) must never render a Chinese
"亿元" unit. Amounts use 억/만 + a Korean currency word instead, while Chinese
behaviour (A-shares and every other currency) stays byte-identical.
"""

import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import litellm  # noqa: F401
except ModuleNotFoundError:
    from tests.litellm_stub import ensure_litellm_stub

    ensure_litellm_stub()

from src.notification import NotificationService
from src.report_language import get_report_labels


def _financial_context(currency="KRW"):
    return {
        "earnings": {
            "status": "ok",
            "data": {
                "financial_report": {
                    "report_date": "2026-06-30",
                    # 4.2182e10 KRW -> 421.82 억 원
                    "revenue": 42_182_000_000.0,
                    "net_profit_parent": -12_242_000_000.0,
                    "operating_cash_flow": -6_570_000_000.0,
                    "roe": -54.88,
                    "currency": currency,
                },
                "dividend": {
                    "ttm_cash_dividend_per_share": 361.0,
                    "ttm_event_count": 1,
                    "ttm_dividend_yield_pct": 1.85,
                    "events": [{"ex_dividend_date": "2026-06-26"}],
                    "currency": currency,
                },
            },
        },
        "growth": {
            "status": "ok",
            "data": {
                "revenue_yoy": 45.62,
                "net_profit_yoy": -132.05,
                "gross_margin": -5.93,
            },
        },
    }


class TestKoreanAmountUnits(unittest.TestCase):
    """`_format_amount_cn` in Korean language mode."""

    def test_korean_units_for_krw(self):
        self.assertEqual(
            NotificationService._format_amount_cn(42_182_000_000.0, "KRW", "ko"), "421.82억 원"
        )
        self.assertEqual(
            NotificationService._format_amount_cn(-12_242_000_000.0, "KRW", "ko"), "-122.42억 원"
        )

    def test_no_chinese_unit_in_korean_language(self):
        out = NotificationService._format_amount_cn(1e8, "KRW", "ko")
        self.assertEqual(out, "1.00억 원")
        for token in ("亿", "万", "元"):
            self.assertNotIn(token, out)

    def test_korean_units_for_other_currencies(self):
        self.assertEqual(NotificationService._format_amount_cn(1e8, "USD", "ko"), "1.00억 달러")
        self.assertEqual(NotificationService._format_amount_cn(1e8, "CNY", "ko"), "1.00억 위안")
        # unknown currency falls back to 원
        self.assertEqual(NotificationService._format_amount_cn(1e8, None, "ko"), "1.00억 원")

    def test_korean_man_and_plain_units(self):
        self.assertEqual(NotificationService._format_amount_cn(12_345, "KRW", "ko"), "1.23만 원")
        self.assertEqual(NotificationService._format_amount_cn(9_876, "KRW", "ko"), "9,876원")

    def test_korean_rounding_moves_to_next_unit(self):
        self.assertEqual(NotificationService._format_amount_cn(99_999_999, "KRW", "ko"), "1.00억 원")
        self.assertEqual(NotificationService._format_amount_cn(-99_999_999.5, "KRW", "ko"), "-1.00억 원")
        self.assertEqual(NotificationService._format_amount_cn(9_999.6, "KRW", "ko"), "1.00만 원")

    def test_korean_per_share_uses_won(self):
        self.assertIn("원", NotificationService._format_per_share(361.0, "KRW", "ko"))
        self.assertEqual(NotificationService._format_per_share(361.0, "KRW", "ko"), "361.00 원")

    def test_non_numeric_stays_na(self):
        self.assertEqual(NotificationService._format_amount_cn(None, "KRW", "ko"), "N/A")
        self.assertEqual(NotificationService._format_amount_cn(float("nan"), "KRW", "ko"), "N/A")


class TestChineseBehaviourPreserved(unittest.TestCase):
    """Chinese output must stay exactly as before."""

    def test_cny_unchanged(self):
        self.assertEqual(NotificationService._format_amount_cn(1e8, "CNY"), "1.00 亿元")
        self.assertEqual(NotificationService._format_amount_cn(1e8, "CNY", "zh"), "1.00 亿元")
        self.assertEqual(NotificationService._format_amount_cn(1e8), "1.00 亿元")

    def test_krw_in_chinese_report_is_won_not_rmb(self):
        # A KRX stock inside a Chinese report reads 亿韩元, never 亿元 (RMB).
        out = NotificationService._format_amount_cn(42_182_000_000.0, "KRW", "zh")
        self.assertEqual(out, "421.82 亿韩元")
        self.assertNotIn("亿元", out)

    def test_other_currencies_unchanged(self):
        self.assertIn("美元", NotificationService._format_amount_cn(1e8, "USD"))
        self.assertIn("港元", NotificationService._format_amount_cn(1e8, "HKD"))
        self.assertIn("新台币", NotificationService._format_amount_cn(1e8, "TWD"))
        self.assertIn("元", NotificationService._format_amount_cn(1e8, "ZZZ"))
        self.assertEqual(NotificationService._format_per_share(24.0, "TWD"), "24.0000 新台币")


class TestKoreanFinancialSummaryRender(unittest.TestCase):
    """End-to-end render of the 재무 요약 block for a Korean stock."""

    def _render(self, language, currency="KRW"):
        svc = NotificationService.__new__(NotificationService)  # renderer needs no instance state
        result = SimpleNamespace(
            report_language=language,
            fundamental_context=_financial_context(currency),
        )
        lines = []
        svc._append_fundamental_blocks(lines, result)
        return "\n".join(lines)

    def test_korean_report_renders_won_units(self):
        out = self._render("ko")
        self.assertIn("재무 요약", out)
        self.assertIn("421.82억 원", out)
        self.assertIn("-122.42억 원", out)
        self.assertIn("-65.70억 원", out)
        self.assertNotIn("亿元", out)
        self.assertNotIn("亿韩元", out)
        self.assertNotIn("421.82 亿", out)
        # 주주 환원 block also uses won
        self.assertIn("361.00 원", out)

    def test_chinese_report_of_korean_stock_renders_won_suffix(self):
        out = self._render("zh")
        self.assertIn("财务摘要", out)
        self.assertIn("421.82 亿韩元", out)
        self.assertNotIn("亿元", out)
        self.assertIn("361.0000 韩元", out)


if __name__ == "__main__":
    unittest.main()
