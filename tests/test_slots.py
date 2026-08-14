"""槽位解析测试 —— 意图分类 + 目录式槽位提取 + 反问澄清.

依赖已 seed 的本地库（python -m src.data.seed -y）。
"""

import pytest

from src.agent.conversation.retriever import StructuredRetriever
from src.agent.conversation.slots import SlotParser
from src.data.models import get_session, Product


def _has_products() -> bool:
    session = get_session()
    try:
        return session.query(Product).count() > 0
    finally:
        session.close()


pytestmark = pytest.mark.skipif(not _has_products(), reason="数据库未灌数据（先运行 seed）")


@pytest.fixture(scope="module")
def parser():
    return SlotParser(StructuredRetriever())


def test_explain_ambiguous_sku_needs_clarification(parser):
    """「薯片」命中多口味 → 歧义，需反问确认."""
    parsed = parser.parse("为什么薯片补这么多？")
    assert parsed["intent"] == "explain"
    assert parsed["needs_clarification"] is True
    assert "store" in parsed["missing"]
    assert "sku" in parsed["missing"]


def test_explain_fully_resolved(parser):
    """「旗舰店的可乐」→ 门店 + SKU 唯一确定，无需澄清."""
    parsed = parser.parse("旗舰店的可乐为什么补这么多")
    assert parsed["intent"] == "explain"
    assert parsed["needs_clarification"] is False
    assert parsed["slots"]["store"]["store_code"] == "S001"
    assert parsed["slots"]["sku"]["sku_code"] == "SKU-021"


def test_store_and_sku_both_resolved(parser):
    """旗舰店 + 可乐 → store/sku 都唯一，无需澄清."""
    parsed = parser.parse("旗舰店为什么可乐补这么多")
    assert parsed["slots"]["store"]["store_code"] == "S001"
    assert parsed["slots"]["sku"]["sku_code"] == "SKU-021"
    assert parsed["needs_clarification"] is False


def test_greeting_intent(parser):
    assert parser.parse("你好").get("intent") == "greeting"


def test_outofscope_intent(parser):
    assert parser.parse("今天天气怎么样").get("intent") == "outofscope"


def test_whatif_missing_slots(parser):
    """What-if 未给实体 → 澄清."""
    parsed = parser.parse("如果促销多3天呢")
    assert parsed["intent"] == "whatif"
    assert parsed["needs_clarification"] is True
    assert "store" in parsed["missing"] or "sku" in parsed["missing"]


def test_whatif_extracts_scenario(parser):
    parsed = parser.parse("如果促销多3天呢")
    assert parsed["slots"]["scenario"] and "促销" in parsed["slots"]["scenario"]


def test_clarification_offers_options(parser):
    """反问话术应带上候选，方便用户确认."""
    parsed = parser.parse("为什么薯片补这么多？")
    text = parser.build_clarification(parsed)
    assert "薯片" in text
