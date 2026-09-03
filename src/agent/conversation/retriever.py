"""可插拔检索器 —— 阶段一(极速版)与阶段二(完整版)共用抽象.

设计目标：
- StructuredRetriever（本文件）：SQLite FTS5 trigram 全文索引 + LIKE 兜底 + 结构化查询。
  适用「小数据量」：几十~几万条决策记录，索引查询毫秒级，零向量依赖。
- VectorRetriever（阶段二，feature/vector-rag 分支）：embedding 语义检索，
  适用「大数据量」：百万级历史决策，回答"上次同样情况结果如何"。
- 两版共用 Retriever 接口 → main 极速版 / 分支完整版互不冲突。

混合匹配策略（中文场景）：
- 查询词 ≥3 字符 → FTS5 trigram 子串索引（SQLite ≥3.34）
- 查询词 2 字符（如"薯片"）→ LIKE '%词%' 兜底（trigram 无法索引 <3 字符词元）
- SKU 编号 → 精确/前缀匹配
"""

import json
import logging
from abc import ABC, abstractmethod

from sqlalchemy import text
from src.data.models import (
    get_session, Store, Product, Suggestion, InventorySnapshot,
)

logger = logging.getLogger(__name__)


# ── 抽象接口 ───────────────────────────────────────

class Retriever(ABC):
    """检索器接口：阶段一/阶段二 共用."""

    @abstractmethod
    def search_products(self, term: str, limit: int = 8) -> list[dict]:
        """按名称/SKU 检索商品，返回 [{id, sku_code, name, category}]"""

    @abstractmethod
    def search_stores(self, term: str) -> list[dict]:
        """按名称/编号检索门店，返回 [{id, store_code, name, type}]"""

    @abstractmethod
    def build_context(self, store_id: int, sku_id: int,
                      suggestion_id: int | None = None) -> dict | None:
        """为已解析的 store+sku 构建决策上下文（快照+实时库存+诊断）."""

    @abstractmethod
    def all_products(self) -> list[dict]:
        """全量商品目录（供槽位解析做目录式匹配）."""

    @abstractmethod
    def all_stores(self) -> list[dict]:
        """全量门店目录（供槽位解析做目录式匹配）."""


# ── 结构化检索器（极速版）──────────────────────────

class StructuredRetriever(Retriever):
    """FTS5 + LIKE + SQL 结构化查询，零向量依赖."""

    def __init__(self):
        self._fts_ready = False

    # ── FTS 索引 ───────────────────────────────────

    def _ensure_fts(self):
        """惰性构建 products_fts 全文索引（内容内嵌，重建幂等）."""
        if self._fts_ready:
            return
        session = get_session()
        try:
            session.execute(
                text(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS products_fts "
                    "USING fts5(sku_code, name, category, tokenize='trigram')"
                )
            )
            # 清空重建（产品表可能被重新灌数据）
            session.execute(text("DELETE FROM products_fts"))
            rows = session.query(Product).all()
            for p in rows:
                session.execute(
                    text(
                        "INSERT INTO products_fts(rowid, sku_code, name, category) "
                        "VALUES (:rowid, :sku, :name, :cat)"
                    ),
                    {"rowid": p.id, "sku": p.sku_code or "", "name": p.name or "", "cat": p.category or ""},
                )
            session.commit()
            self._fts_ready = True
            logger.info(f"[retriever] products_fts 索引就绪: {len(rows)} 条")
        except Exception as e:
            session.rollback()
            logger.warning(f"[retriever] FTS 初始化失败，降级为 LIKE: {e}")
        finally:
            session.close()

    # ── 商品检索 ───────────────────────────────────

    def search_products(self, term: str, limit: int = 8) -> list[dict]:
        term = (term or "").strip()
        if not term:
            return []
        self._ensure_fts()

        results: dict[int, dict] = {}

        # 策略 1: SKU 编号精确 / 前缀
        if "SKU" in term.upper() or term.isdigit():
            code_clean = term.upper().replace("SKU-", "").strip()
            pattern = f"SKU-{code_clean:0>3}" if code_clean.isdigit() else f"SKU-{code_clean}"
            session = get_session()
            try:
                for p in session.query(Product).filter(Product.sku_code.like(f"%{pattern}%")).limit(limit).all():
                    results[p.id] = self._product_dict(p, score=100)
            finally:
                session.close()

        # 策略 2: FTS5 trigram（≥3 字符连续子串）
        if len(term) >= 3 and len(results) < limit:
            try:
                session = get_session()
                try:
                    q = term.replace('"', '""')
                    rows = session.execute(
                        text(
                            "SELECT rowid, bm25(products_fts) AS rank FROM products_fts "
                            "WHERE products_fts MATCH :q ORDER BY rank LIMIT :limit"
                        ),
                        {"q": f'"{q}"', "limit": limit},
                    ).fetchall()
                    ids = [r[0] for r in rows]
                    if ids:
                        for p in session.query(Product).filter(Product.id.in_(ids)).all():
                            results.setdefault(p.id, self._product_dict(p, score=90))
                finally:
                    session.close()
            except Exception as e:
                logger.debug(f"[retriever] FTS 检索失败走 LIKE: {e}")

        # 策略 3: LIKE 兜底（覆盖 2 字词 / FTS 未命中）
        if len(results) < limit:
            session = get_session()
            try:
                like = f"%{term}%"
                for p in session.query(Product).filter(
                    Product.name.like(like) | Product.sku_code.like(like) | Product.category.like(like)
                ).limit(limit).all():
                    results.setdefault(p.id, self._product_dict(p, score=70))
            finally:
                session.close()

        return sorted(results.values(), key=lambda x: x["score"], reverse=True)[:limit]

    @staticmethod
    def _product_dict(p: Product, score: int) -> dict:
        return {
            "id": p.id, "sku_code": p.sku_code, "name": p.name,
            "category": p.category, "score": score,
        }

    # ── 门店检索 ───────────────────────────────────

    def search_stores(self, term: str) -> list[dict]:
        term = (term or "").strip()
        if not term:
            return []
        session = get_session()
        try:
            # 编号精确 + 名称/类型/商圈 子串
            like = f"%{term}%"
            stores = (
                session.query(Store)
                .filter(
                    Store.store_code.like(like) | Store.name.like(like)
                    | Store.type.like(like) | Store.district.like(like)
                )
                .limit(5)
                .all()
            )
            return [
                {"id": s.id, "store_code": s.store_code, "name": s.name,
                 "type": s.type, "district": s.district, "score": 100}
                for s in stores
            ]
        finally:
            session.close()

    # ── 目录查询（槽位解析用）────────────────────────

    def all_products(self) -> list[dict]:
        session = get_session()
        try:
            return [self._product_dict(p, score=0) for p in session.query(Product).all()]
        finally:
            session.close()

    def all_stores(self) -> list[dict]:
        session = get_session()
        try:
            return [
                {"id": s.id, "store_code": s.store_code, "name": s.name,
                 "type": s.type, "district": s.district, "score": 0}
                for s in session.query(Store).all()
            ]
        finally:
            session.close()

    # ── 决策上下文构建 ─────────────────────────────

    def build_context(self, store_id: int, sku_id: int,
                      suggestion_id: int | None = None) -> dict | None:
        """构建决策上下文（建议快照 + 商品 + 最新库存 + 门店）.

        与阶段二 VectorRetriever 的语义等价：检索到的"证据块"。
        """
        session = get_session()
        try:
            result = {}

            # 1. 建议决策（优先指定，否则取该店该 SKU 最新一条）
            sug = None
            if suggestion_id:
                sug = session.query(Suggestion).filter(Suggestion.id == suggestion_id).first()
            if not sug:
                sug = (
                    session.query(Suggestion)
                    .filter(Suggestion.store_id == store_id, Suggestion.product_id == sku_id)
                    .order_by(Suggestion.generated_at.desc())
                    .first()
                )
            if sug:
                result["suggestion"] = {
                    "id": sug.id,
                    "suggested_qty": sug.suggested_qty,
                    "urgency": sug.urgency,
                    "confidence": sug.confidence,
                    "anomaly_type": sug.anomaly_type,
                    "reasoning": sug.reasoning,
                    "status": sug.status,
                    "generated_at": sug.generated_at,
                    "feedback_rating": sug.feedback_rating,
                }
                if sug.decision_factors:
                    try:
                        result["decision_factors"] = json.loads(sug.decision_factors)
                    except json.JSONDecodeError:
                        pass

            # 2. SKU 信息
            product = session.query(Product).filter(Product.id == sku_id).first()
            if product:
                result["product"] = {
                    "id": product.id, "sku_code": product.sku_code,
                    "name": product.name, "category": product.category,
                    "lead_time_days": product.lead_time_days,
                    "target_stock_days": product.target_stock_days,
                    "min_order_qty": product.min_order_qty,
                }

            # 3. 最新库存（对比决策时 vs 现在）
            latest_inv = (
                session.query(InventorySnapshot)
                .filter(
                    InventorySnapshot.store_id == store_id,
                    InventorySnapshot.product_id == sku_id,
                )
                .order_by(InventorySnapshot.snapshot_date.desc())
                .first()
            )
            if latest_inv:
                result["current_inventory"] = {
                    "stock_on_hand": latest_inv.stock_on_hand,
                    "in_transit": latest_inv.in_transit,
                    "safety_stock_line": latest_inv.safety_stock_line,
                    "snapshot_date": latest_inv.snapshot_date,
                }

            # 4. 门店
            store = session.query(Store).filter(Store.id == store_id).first()
            if store:
                result["store"] = {
                    "id": store.id, "name": store.name,
                    "type": store.type, "maturity": store.maturity,
                }

            return result if result else None
        finally:
            session.close()

    def store_summary(self, store_id: int, limit: int = 8) -> dict | None:
        """门店级汇总（summarize 意图）：最新一批建议按紧急度排序 + 统计.

        回答「这家店缺货的有哪些」「整体补货情况」类问题，
        单 SKU 上下文（build_context）覆盖不了的场景。
        """
        session = get_session()
        try:
            store = session.query(Store).filter(Store.id == store_id).first()
            if not store:
                return None

            latest = (
                session.query(Suggestion)
                .filter(Suggestion.store_id == store_id)
                .order_by(Suggestion.generated_at.desc())
                .first()
            )
            if not latest:
                return {
                    "type": "store_summary",
                    "store": {"id": store.id, "name": store.name,
                              "type": store.type, "maturity": store.maturity},
                    "suggestions": [],
                }

            batch = (
                session.query(Suggestion)
                .filter(
                    Suggestion.store_id == store_id,
                    Suggestion.generated_at == latest.generated_at,
                )
                .all()
            )
            severity = {"critical": 0, "high": 1, "medium": 2, "low": 3}
            items = sorted(batch, key=lambda s: severity.get(s.urgency or "", 9))

            product_map = {
                p.id: p for p in session.query(Product)
                .filter(Product.id.in_([s.product_id for s in batch])).all()
            }

            return {
                "type": "store_summary",
                "store": {"id": store.id, "name": store.name,
                          "type": store.type, "maturity": store.maturity},
                "generated_at": latest.generated_at,
                "stats": {
                    "total": len(batch),
                    "critical": sum(1 for s in batch if s.urgency == "critical"),
                    "high": sum(1 for s in batch if s.urgency == "high"),
                    "to_replenish": sum(1 for s in batch if (s.suggested_qty or 0) > 0),
                },
                "suggestions": [
                    {
                        "sku_code": product_map[s.product_id].sku_code,
                        "name": product_map[s.product_id].name,
                        "suggested_qty": s.suggested_qty,
                        "urgency": s.urgency,
                        "anomaly_type": s.anomaly_type,
                        "confidence": s.confidence,
                    }
                    for s in items[:limit]
                ],
            }
        finally:
            session.close()
