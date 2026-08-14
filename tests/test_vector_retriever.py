"""向量检索器测试（阶段二 · 完整版）.

- 未配置 EMBEDDING_API_KEY → 自动降级（不崩，相似案例为空）
- 注入 fake embedder（bigram 词袋向量）→ 验证余弦排序 + 上下文注入
  用假 embedder 隔离外部 API，测试只验证检索逻辑本身。
"""

import hashlib
import re

import numpy as np
import pytest

from src.agent.conversation.vector_retriever import VectorRetriever
from src.data.models import get_session, Suggestion


def _has_suggestions():
    s = get_session()
    try:
        return s.query(Suggestion).count() > 0
    finally:
        s.close()


pytestmark = pytest.mark.skipif(not _has_suggestions(), reason="数据库未灌数据")


@pytest.fixture(scope="module", autouse=True)
def cleanup_embeddings():
    """测试用假 embedder 会向缓存表写入假向量，结束后清理."""
    yield
    from src.data.models import engine
    from sqlalchemy import text
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS suggestion_embeddings"))


class FakeEmbedder:
    """bigram 词袋 → 确定性向量：共享 2 字词则余弦高.

    模拟 OpenAI 兼容 SDK：`client.embeddings.create(...)` 返回 {data: [{embedding}]}.
    """

    DIM = 512

    @staticmethod
    def _vec(text: str) -> np.ndarray:
        vec = np.zeros(FakeEmbedder.DIM, dtype=np.float32)
        for run in re.findall(r"[一-鿿]{2,}", text):
            for i in range(len(run) - 1):
                bg = run[i:i + 2]
                idx = int(hashlib.md5(bg.encode()).hexdigest()[:4], 16) % FakeEmbedder.DIM
                vec[idx] += 1.0
        return vec

    class _Data:
        def __init__(self, embedding):
            self.embedding = embedding

    class _Resp:
        def __init__(self, vecs):
            self.data = [FakeEmbedder._Data(v) for v in vecs]

    class _Embeddings:
        @staticmethod
        def create(model=None, input=None):
            return FakeEmbedder._Resp([FakeEmbedder._vec(t) for t in input])

    embeddings = _Embeddings()


@pytest.fixture(scope="module")
def vr():
    r = VectorRetriever()
    r._client = FakeEmbedder()  # 注入假 embedder，隔离外部 API
    return r


def test_degrades_without_key():
    """无 embedding key → 语料空、检索空、不崩."""
    r = VectorRetriever()
    if r._client is not None:
        pytest.skip("已配置 EMBEDDING_API_KEY，跳过降级测试")
    assert r._build_corpus() == []
    assert r.search_similar("可乐补货") == []


def test_corpus_built(vr):
    corpus = vr._build_corpus()
    assert len(corpus) > 0
    assert "vec" in corpus[0] and "suggestion_id" in corpus[0]
    assert "text" in corpus[0]


def test_search_similar_finds_identical(vr):
    """查询=某条决策原文 → 自身应排第一（自相似≈1）."""
    corpus = vr._build_corpus()
    target = corpus[0]
    hits = vr.search_similar(target["text"], top_k=3)
    assert hits, "未返回任何相似决策"
    assert hits[0]["suggestion_id"] == target["suggestion_id"]
    assert hits[0]["score"] > 0.95


def test_build_context_injects_similar(vr):
    """构建上下文 → 注入 similar_historical，且排除自身."""
    ctx = vr.build_context(store_id=1, sku_id=21)  # 旗舰店 × 可乐
    assert ctx is not None
    assert "similar_historical" in ctx
    own = ctx["suggestion"]["id"]
    assert all(s["suggestion_id"] != own for s in ctx["similar_historical"])
