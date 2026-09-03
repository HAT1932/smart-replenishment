"""多轮对话状态追踪测试 —— 修复"反问死循环".

复现用户实际遇到的对话（修复前）：
  你: 薯片烧烤味 150g（SKU-002）   → SKU 编号写明了仍被反问商品（编号无优先权）
  你: 社区店-回龙观                 → 丢掉上一轮 SKU，重新反问商品 → 循环
  你: SKU-002 薯片烧烤味            → 丢掉上一轮门店，重新反问门店 → 循环
  你: 进货量多少？                  → 永远得不到回答

修复后：编号/完整名优先裁决 + 多轮槽位合并（对话状态追踪）。
"""

import pytest

from src.agent.conversation.agent import ConversationAgent
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
def agent():
    return ConversationAgent()


@pytest.fixture(scope="module")
def parser():
    return SlotParser(StructuredRetriever())


# ── 分层裁决：编号 > 完整名 > 2字窗口 ──────────────

def test_sku_code_wins_over_ambiguous_name(parser):
    """SKU 编号明确写在问题里 → 直接裁决，不再因「薯片」命中 3 口味而反问."""
    parsed = parser.parse("薯片烧烤味 150g（SKU-002）")
    assert parsed["slots"]["sku"]["sku_code"] == "SKU-002"
    assert parsed["missing"] == ["store"]  # 只缺门店


def test_full_name_segment_wins(parser):
    """「薯片烧烤味」完整名命中 → 压过其他口味的 2 字窗口命中."""
    parsed = parser.parse("薯片烧烤味")
    assert parsed["slots"]["sku"]["sku_code"] == "SKU-002"


def test_store_district_disambiguation(parser):
    """「社区店-回龙观」→ 名称段+窗口全命中 → 唯一裁决."""
    parsed = parser.parse("社区店-回龙观")
    assert parsed["slots"]["store"] is not None
    assert "回龙观" in parsed["slots"]["store"]["name"]


def test_store_resolved_by_single_segment(parser):
    """只说商圈名「回龙观」/「望京」→ 也能唯一命中."""
    parsed = parser.parse("回龙观")
    assert "回龙观" in parsed["slots"]["store"]["name"]
    parsed = parser.parse("望京")
    assert parsed["slots"]["store"]["store_code"] == "S003"


# ── 多轮状态追踪：不再循环反问 ──────────────────────

def test_multi_turn_no_loop(agent):
    """复现用户死循环对话：反问 → 补答 → 合并槽位 → 直接解析."""
    sid = "test-dialogue-loop"

    # turn1: SKU 编号明确 → 只反问门店，不反问商品
    ctx, diag, rule = agent._preprocess("薯片烧烤味 150g（SKU-002）", None, sid)
    assert ctx is None and rule is not None
    assert "门店" in rule
    assert "商品" not in rule  # 关键：商品已由 SKU 编号裁决

    # turn2: 用户只补答门店 → 与已积累的 SKU 合并 → 解析成功（不反问）
    ctx, diag, rule = agent._preprocess("社区店-回龙观", None, sid)
    assert rule is None and ctx is not None
    assert "回龙观" in diag["store"]
    assert "SKU-002" in diag["sku"]

    # turn3: 换个问法（无任何实体词）→ 继承上一轮门店/SKU，直接出上下文
    ctx, diag, rule = agent._preprocess("进货量多少？", None, sid)
    assert rule is None and ctx is not None
    assert "SKU-002" in diag["sku"]


def test_new_session_starts_fresh(agent):
    """新会话不继承旧槽位：歧义的「薯片」仍会反问商品."""
    ctx, diag, rule = agent._preprocess("为什么薯片补这么多？", None, "brand-new-session")
    assert rule is not None and "商品" in rule


def test_explicit_context_seeds_followup(agent):
    """审核台显式上下文 → 播种槽位状态，后续 What-if 追问直接继承."""
    sid = "test-dialogue-seed"
    explicit = {"store_id": 1, "sku_id": 21, "suggestion_id": None}
    ctx, diag, rule = agent._preprocess("为什么补这么多", explicit, sid)
    assert ctx == explicit  # 显式上下文直通

    ctx, diag, rule = agent._preprocess("如果促销多3天呢？", None, sid)
    assert rule is None and ctx is not None
    assert ctx["store_id"] == 1 and ctx["sku_id"] == 21


def test_summarize_store_level(agent):
    """「旗舰店缺货的有哪些？」→ summarize 意图只要门店 → 门店级汇总上下文."""
    sid = "test-dialogue-summarize"
    ctx, diag, rule = agent._preprocess("旗舰店缺货的有哪些？", None, sid)
    assert rule is None and ctx is not None
    assert ctx.get("type") == "store_summary"
    assert "stats" in ctx and "suggestions" in ctx


def test_store_summary_orders_by_urgency():
    """门店汇总：建议按紧急度排序（critical 在最前）."""
    retriever = StructuredRetriever()
    summary = retriever.store_summary(1)
    assert summary is not None
    assert summary["stats"]["total"] > 0
    if summary["suggestions"]:
        assert summary["suggestions"][0]["urgency"] == "critical"
