"""阶段一 结构化检索器测试（FTS5 trigram + LIKE + 目录查询）.

依赖已 seed 的本地库（python -m src.data.seed -y）。若库为空则整组跳过。
"""

import pytest

from src.agent.conversation.retriever import StructuredRetriever
from src.data.models import get_session, Product


def _has_products() -> bool:
    session = get_session()
    try:
        return session.query(Product).count() > 0
    finally:
        session.close()


pytestmark = pytest.mark.skipif(not _has_products(), reason="数据库未灌数据（先运行 seed）")


@pytest.fixture(scope="module")
def retriever():
    return StructuredRetriever()


def test_search_by_short_chinese_name(retriever):
    """2 字词走 LIKE：薯片 命中全部口味."""
    hits = retriever.search_products("薯片")
    assert len(hits) >= 3
    assert all("薯片" in h["name"] for h in hits)


def test_search_unique_product(retriever):
    """可乐 唯一命中 SKU-021."""
    hits = retriever.search_products("可乐")
    assert len(hits) == 1
    assert hits[0]["name"].startswith("可乐")


def test_search_by_sku_code(retriever):
    """SKU 编号精确检索."""
    hits = retriever.search_products("SKU-001")
    assert hits and hits[0]["sku_code"] == "SKU-001"


def test_search_long_term_uses_fts(retriever):
    """≥3 字连续词走 FTS trigram 索引."""
    hits = retriever.search_products("薯片原味")
    assert hits and "薯片原味" in hits[0]["name"]


def test_search_stores(retriever):
    """门店名子串检索."""
    hits = retriever.search_stores("旗舰")
    assert hits and "旗舰" in hits[0]["name"]


def test_all_products_returns_catalog(retriever):
    """目录式匹配的数据源：全量商品."""
    catalog = retriever.all_products()
    assert len(catalog) >= 1
    assert {"id", "sku_code", "name"} <= set(catalog[0])


def test_build_context(retriever):
    """旗舰店 × 可乐 → 建议 + 商品 + 库存 + 门店."""
    ctx = retriever.build_context(store_id=1, sku_id=21)
    assert ctx is not None
    assert "suggestion" in ctx
    assert "product" in ctx and ctx["product"]["name"].startswith("可乐")
    assert "store" in ctx and "旗舰" in ctx["store"]["name"]
