"""补货建议 API 路由."""

import json
import logging
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import func, case

from src.data.models import get_session, Store, Product, Suggestion, InventorySnapshot
from src.agent.decision.engine import DecisionEngine
from src.api.schemas import (
    SuggestionItem, SuggestionListResponse, SuggestionDetailResponse,
    ConfirmRequest, GenerateRequest, POSConnectRequest, POSConnectResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["suggestions"])


# ═══════════════════════════════════════════════
# POS 对接
# ═══════════════════════════════════════════════

@router.post("/pos/connect", response_model=POSConnectResponse)
def pos_connect(req: POSConnectRequest):
    """模拟 POS 对接."""
    return POSConnectResponse(
        connection_id=f"conn_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        status="syncing",
        sync_progress={"stores": 3, "products": 50, "orders": "13.5K"},
    )


# ═══════════════════════════════════════════════
# 建议生成
# ═══════════════════════════════════════════════

@router.post("/suggestions/generate")
def generate_suggestions(req: GenerateRequest):
    """触发补货建议生成."""
    engine = DecisionEngine()
    try:
        result = engine.generate(req.store_id)
        return {
            "status": "ok",
            "store_id": req.store_id,
            "mode": result["mode"],
            "total": result["total"],
            "llm_summary": result.get("llm_review", {}).get("summary", ""),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════
# 建议列表
# ═══════════════════════════════════════════════

@router.get("/suggestions/{store_id}", response_model=SuggestionListResponse)
def list_suggestions(
    store_id: int,
    urgency: str = Query(None, description="过滤紧急度"),
    category: str = Query(None, description="过滤品类"),
    anomaly: str = Query(None, description="过滤异常类型"),
    status: str = Query("pending", description="过滤状态"),
):
    """获取门店的补货建议列表."""
    session = get_session()
    try:
        store = session.query(Store).filter(Store.id == store_id).first()
        if not store:
            raise HTTPException(status_code=404, detail="Store not found")

        q = session.query(Suggestion).filter(
            Suggestion.store_id == store_id,
            Suggestion.status == status,
        )

        if urgency:
            q = q.filter(Suggestion.urgency == urgency)

        suggestions = q.order_by(
            case(
                (Suggestion.urgency == "critical", 0),
                (Suggestion.urgency == "high", 1),
                (Suggestion.urgency == "medium", 2),
                else_=3,
            )
        ).all()

        items = []
        summary = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for sug in suggestions:
            product = session.query(Product).filter(Product.id == sug.product_id).first()
            inv = (
                session.query(InventorySnapshot)
                .filter(
                    InventorySnapshot.store_id == store_id,
                    InventorySnapshot.product_id == sug.product_id,
                )
                .order_by(InventorySnapshot.snapshot_date.desc())
                .first()
            )

            # 品类过滤
            if category and product and product.category != category:
                continue
            # 异常过滤
            if anomaly and sug.anomaly_type and anomaly not in sug.anomaly_type:
                continue

            summary[sug.urgency] = summary.get(sug.urgency, 0) + 1

            items.append(SuggestionItem(
                id=sug.id,
                store_id=sug.store_id,
                product_id=sug.product_id,
                sku_code=product.sku_code if product else "?",
                sku_name=product.name if product else "?",
                category=product.category if product else "?",
                suggested_qty=sug.suggested_qty if sug.suggested_qty > 0 else None,
                final_qty=sug.final_qty,
                urgency=sug.urgency,
                confidence=sug.confidence or "medium",
                confidence_emoji=sug.confidence_level or "green",
                anomaly_flags=sug.anomaly_type.split(",") if sug.anomaly_type else [],
                reasoning=sug.reasoning or "",
                status=sug.status or "pending",
                current_stock=inv.stock_on_hand if inv else 0,
                safety_stock_line=inv.safety_stock_line if inv else 0,
            ))

        return SuggestionListResponse(
            store_id=store_id,
            store_name=store.name,
            total=len(items),
            summary=summary,
            suggestions=items,
        )
    finally:
        session.close()


# ═══════════════════════════════════════════════
# 建议详情
# ═══════════════════════════════════════════════

@router.get("/suggestions/detail/{suggestion_id}", response_model=SuggestionDetailResponse)
def suggestion_detail(suggestion_id: int):
    """获取单条建议的完整详情（含决策因子 + 产品信息 + 库存对比）."""
    session = get_session()
    try:
        sug = session.query(Suggestion).filter(Suggestion.id == suggestion_id).first()
        if not sug:
            raise HTTPException(status_code=404, detail="Suggestion not found")

        product = session.query(Product).filter(Product.id == sug.product_id).first()
        store = session.query(Store).filter(Store.id == sug.store_id).first()

        latest_inv = (
            session.query(InventorySnapshot)
            .filter(
                InventorySnapshot.store_id == sug.store_id,
                InventorySnapshot.product_id == sug.product_id,
            )
            .order_by(InventorySnapshot.snapshot_date.desc())
            .first()
        )

        # 解析决策因子 JSON
        decision_factors = {}
        if sug.decision_factors:
            try:
                decision_factors = json.loads(sug.decision_factors)
            except json.JSONDecodeError:
                pass

        return SuggestionDetailResponse(
            suggestion={
                "id": sug.id,
                "suggested_qty": sug.suggested_qty,
                "final_qty": sug.final_qty,
                "urgency": sug.urgency,
                "confidence": sug.confidence,
                "confidence_emoji": sug.confidence_level,
                "anomaly_flags": sug.anomaly_type.split(",") if sug.anomaly_type else [],
                "reasoning": sug.reasoning,
                "status": sug.status,
                "generated_at": sug.generated_at,
                "confirmed_at": sug.confirmed_at,
            },
            decision_factors=decision_factors,
            product={
                "id": product.id if product else None,
                "sku_code": product.sku_code if product else "?",
                "name": product.name if product else "?",
                "category": product.category if product else "?",
                "lead_time_days": product.lead_time_days if product else 3,
                "target_stock_days": product.target_stock_days if product else 7,
                "min_order_qty": product.min_order_qty if product else 1,
            },
            store={
                "id": store.id if store else None,
                "name": store.name if store else "?",
                "type": store.type if store else "?",
                "maturity": store.maturity if store else "?",
            },
            current_inventory={
                "stock_on_hand": latest_inv.stock_on_hand if latest_inv else 0,
                "in_transit": latest_inv.in_transit if latest_inv else 0,
                "safety_stock_line": latest_inv.safety_stock_line if latest_inv else 0,
            },
        )
    finally:
        session.close()


# ═══════════════════════════════════════════════
# 确认/修改/驳回
# ═══════════════════════════════════════════════

@router.put("/suggestions/{suggestion_id}/confirm")
def confirm_suggestion(suggestion_id: int, req: ConfirmRequest):
    """确认、修改或驳回补货建议."""
    session = get_session()
    try:
        sug = session.query(Suggestion).filter(Suggestion.id == suggestion_id).first()
        if not sug:
            raise HTTPException(status_code=404, detail="Suggestion not found")

        if sug.status != "pending":
            raise HTTPException(status_code=400, detail=f"Suggestion already {sug.status}")

        now = datetime.now().isoformat()
        if req.action == "confirm":
            sug.status = "confirmed"
            sug.final_qty = sug.suggested_qty
        elif req.action == "modify":
            sug.status = "modified"
            sug.final_qty = req.new_quantity
            sug.modify_reason = req.reason
        elif req.action == "reject":
            sug.status = "rejected"
            sug.final_qty = 0
            sug.modify_reason = req.reason
        else:
            raise HTTPException(status_code=400, detail=f"Invalid action: {req.action}")

        sug.confirmed_at = now
        sug.operator_id = req.operator_id
        session.commit()

        return {"status": "ok", "new_status": sug.status, "confirmed_at": now}
    except HTTPException:
        raise
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()
