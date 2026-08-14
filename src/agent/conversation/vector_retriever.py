"""向量检索器（阶段二 · 完整版）—— 语义检索历史决策.

与结构化检索器共用 `Retriever` 接口，双版本同仓库共存：
- 极速版（main / StructuredRetriever）：小数据量，FTS5 + 目录匹配 + 反问
- 完整版（本分支 / VectorRetriever）：大数据量，embedding 语义检索历史案例

实现要点：
1. 语料 = suggestions 表（门店/商品/建议量/异常/决策理由/反馈），embedding 后
   持久化到 suggestion_embeddings 表（增量构建，重灌数据后自动补新）
2. 检索 = 查询 → embedding → 余弦相似度 top-k（内存 numpy，几千行足够；
   接口预留将来接向量库）
3. 降级 = 未配置 EMBEDDING_API_KEY → 自动回退结构化检索，系统不崩
"""

import json
import logging
from datetime import datetime

import numpy as np
from openai import OpenAI
from sqlalchemy import text

from src.config import config
from src.data.models import get_session, Suggestion, Product, Store
from src.agent.conversation.retriever import Retriever, StructuredRetriever

logger = logging.getLogger(__name__)

_EMBED_BATCH = 16  # 批量 embedding 大小


class VectorRetriever(Retriever):
    """语义检索历史决策；目录/上下文回退结构化实现."""

    def __init__(self):
        self.structured = StructuredRetriever()
        self._client = None
        self._corpus = None
        if config.EMBEDDING_API_KEY:
            self._client = OpenAI(
                api_key=config.EMBEDDING_API_KEY,
                base_url=config.EMBEDDING_BASE_URL or None,
            )
        else:
            logger.warning("[vector] 未配置 EMBEDDING_API_KEY，向量检索不可用，自动回退结构化检索")
        self._ensure_table()

    # ── Retriever 接口：目录方法委托结构化（槽位解析共用）──

    def search_products(self, term: str, limit: int = 8) -> list[dict]:
        return self.structured.search_products(term, limit)

    def search_stores(self, term: str) -> list[dict]:
        return self.structured.search_stores(term)

    def all_products(self) -> list[dict]:
        return self.structured.all_products()

    def all_stores(self) -> list[dict]:
        return self.structured.all_stores()

    # ── embedding ───────────────────────────────────

    def embed(self, texts: list[str]) -> list[list[float]] | None:
        if not self._client or not texts:
            return None
        try:
            resp = self._client.embeddings.create(
                model=config.EMBEDDING_MODEL, input=texts
            )
            # 兼容 list / numpy.ndarray，统一转 list 便于 json 持久化
            return [
                d.embedding.tolist() if isinstance(d.embedding, np.ndarray) else d.embedding
                for d in resp.data
            ]
        except Exception as e:
            logger.error(f"[vector] embedding 失败: {e}")
            return None

    def _ensure_table(self):
        from src.data.models import engine
        with engine.begin() as conn:
            conn.execute(
                text(
                    "CREATE TABLE IF NOT EXISTS suggestion_embeddings ("
                    " suggestion_id INTEGER PRIMARY KEY,"
                    " model TEXT, vector_json TEXT, text_preview TEXT, updated_at TEXT)"
                )
            )

    # ── 语料构建（增量）─────────────────────────────

    def _build_corpus(self) -> list[dict]:
        """加载/增量构建 历史决策语料 + 向量."""
        if self._corpus is not None:
            return self._corpus
        if not self._client:
            self._corpus = []
            return self._corpus

        session = get_session()
        new_embeddings = []
        try:
            texts: dict[int, str] = {}
            sug_rows: dict[int, Suggestion] = {}
            for sug in session.query(Suggestion).all():
                product = session.query(Product).filter(Product.id == sug.product_id).first()
                store = session.query(Store).filter(Store.id == sug.store_id).first()
                texts[sug.id] = self._suggestion_text(sug, product, store)
                sug_rows[sug.id] = sug

            # 已有缓存
            vec_cache: dict[int, list[float]] = {}
            rows = session.execute(
                text(
                    "SELECT suggestion_id, vector_json FROM suggestion_embeddings WHERE model = :model"
                ),
                {"model": config.EMBEDDING_MODEL},
            ).fetchall()
            for sid, vj in rows:
                if sid in texts:
                    vec_cache[sid] = json.loads(vj)

            # 增量 embedding 未缓存项
            missing = [sid for sid in texts if sid not in vec_cache]
            for i in range(0, len(missing), _EMBED_BATCH):
                ids = missing[i:i + _EMBED_BATCH]
                vecs = self.embed([texts[sid] for sid in ids])
                if vecs is None:
                    break
                for sid, v in zip(ids, vecs):
                    vec_cache[sid] = v
                    new_embeddings.append((sid, v, texts[sid]))
        finally:
            session.close()

        # 持久化新向量
        if new_embeddings:
            from src.data.models import engine
            with engine.begin() as conn:
                for sid, v, txt in new_embeddings:
                    conn.execute(
                        text(
                            "INSERT OR REPLACE INTO suggestion_embeddings"
                            "(suggestion_id, model, vector_json, text_preview, updated_at) "
                            "VALUES (:sid, :model, :vec, :preview, :ts)"
                        ),
                        {
                            "sid": sid,
                            "model": config.EMBEDDING_MODEL,
                            "vec": json.dumps(v),
                            "preview": txt[:60],
                            "ts": datetime.now().isoformat(),
                        },
                    )
            logger.info(f"[vector] 新增 embedding: {len(new_embeddings)} 条")

        self._corpus = [
            {
                "suggestion_id": sid,
                "text": texts[sid],
                "vec": np.asarray(vec_cache[sid], dtype=np.float32),
                "feedback": sug_rows[sid].feedback_rating,
            }
            for sid in texts if sid in vec_cache
        ]
        logger.info(f"[vector] 语料就绪: {len(self._corpus)} 条（模型 {config.EMBEDDING_MODEL}）")
        return self._corpus

    # ── 语义检索 ───────────────────────────────────

    def search_similar(self, query: str, top_k: int | None = None) -> list[dict]:
        """查询语义最相似的历史决策（含反馈），按相似度降序."""
        top_k = top_k or config.EMBEDDING_TOP_K
        corpus = self._build_corpus()
        if not corpus:
            return []
        qv = self.embed([query])
        if qv is None:
            return []
        q = np.asarray(qv[0], dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []

        scored = []
        for item in corpus:
            v = item["vec"]
            norm = np.linalg.norm(v)
            if norm == 0:
                continue
            score = float(np.dot(q, v) / (q_norm * norm))
            scored.append({**item, "score": round(score, 4)})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return [s for s in scored if s["score"] > 0.30][:top_k]

    # ── 上下文构建：结构化 + 历史相似案例 ────────────

    def build_context(self, store_id: int, sku_id: int,
                      suggestion_id: int | None = None) -> dict | None:
        ctx = self.structured.build_context(store_id, sku_id, suggestion_id)
        if not ctx:
            return None

        # 语义检索相似历史决策（排除自身），回答"上次同样情况结果如何"
        query = self._ctx_query_text(ctx)
        similar = self.search_similar(query, top_k=config.EMBEDDING_TOP_K + 1)
        own_id = ctx.get("suggestion", {}).get("id")
        if own_id:
            similar = [s for s in similar if s["suggestion_id"] != own_id]
        ctx["similar_historical"] = [
            {
                "suggestion_id": s["suggestion_id"],
                "text": s["text"][:200],
                "score": s["score"],
                "feedback": s.get("feedback"),
            }
            for s in similar[:config.EMBEDDING_TOP_K]
        ]
        return ctx

    # ── 文本构造 ───────────────────────────────────

    def _suggestion_text(self, sug: Suggestion, product, store) -> str:
        name = product.name if product else "?"
        store_name = store.name if store else "?"
        factors = ""
        if sug.decision_factors:
            try:
                df = json.loads(sug.decision_factors)
                parts = [f"日均销量={df.get('avg_daily_sales')}",
                         f"当前库存={df.get('current_stock')}",
                         f"安全线={df.get('safety_stock_line')}"]
                factors = " " + " ".join(p for p in parts if p)
            except Exception:
                pass
        fb = {"like": "用户反馈合理", "dislike": "用户反馈不合理"}.get(sug.feedback_rating, "")
        return (
            f"[{store_name}] {name} 建议补货{sug.suggested_qty}件 紧急度{sug.urgency} "
            f"置信度{sug.confidence} 异常[{sug.anomaly_type or '无'}]{factors} "
            f"理由:{sug.reasoning or '无'} {fb}"
        )

    def _ctx_query_text(self, ctx: dict) -> str:
        store = ctx.get("store", {}).get("name", "")
        product = ctx.get("product", {}).get("name", "")
        sug = ctx.get("suggestion", {})
        return (
            f"[{store}] {product} 建议补货{sug.get('suggested_qty')}件 "
            f"紧急度{sug.get('urgency')} 异常[{sug.get('anomaly_type') or '无'}]"
        )
