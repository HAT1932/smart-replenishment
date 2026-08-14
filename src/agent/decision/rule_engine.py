"""规则引擎：纯 Python 实现 PRD 3.2-3.5 的补货公式.

LLM 不参与计算。作为主力(MVP) + LLM 失败的降级兜底。
"""

from datetime import datetime, timedelta
from sqlalchemy import func
from src.data.models import get_session, Store, Product, SalesDaily, InventorySnapshot


# ═══════════════════════════════════════════════════════════
# 数据查询
# ═══════════════════════════════════════════════════════════

class DataProvider:
    """封装数据库查询，RuleEngine 和 LLMAgent 共用."""

    def __init__(self):
        self._session = get_session()

    def close(self):
        """关闭数据库会话，释放连接."""
        if self._session:
            self._session.close()
            self._session = None

    @property
    def session(self):
        if self._session is None:
            self._session = get_session()
        return self._session

    def get_store(self, store_id: int) -> Store | None:
        return self.session.query(Store).filter(Store.id == store_id).first()

    def get_products(self, store_id: int = None) -> list[Product]:
        q = self.session.query(Product)
        return q.all()

    def get_current_inventory(self, product_id: int, store_id: int) -> dict:
        """获取最新库存快照."""
        snap = (
            self.session.query(InventorySnapshot)
            .filter(
                InventorySnapshot.product_id == product_id,
                InventorySnapshot.store_id == store_id,
            )
            .order_by(InventorySnapshot.snapshot_date.desc())
            .first()
        )
        if snap is None:
            return {"stock_on_hand": 0, "in_transit": 0, "safety_stock_line": 0}
        return {
            "stock_on_hand": snap.stock_on_hand,
            "in_transit": snap.in_transit,
            "safety_stock_line": snap.safety_stock_line or 0,
        }

    def get_sales_summary(self, product_id: int, store_id: int, store_open_date: str = None) -> dict:
        """获取加权日均销量和变化趋势."""
        today = datetime.now().strftime("%Y-%m-%d")

        # 近 7/14/30 天日均
        sums = {}
        avgs = {}
        for days_label, days in [("last_7d", 7), ("last_14d", 14), ("last_30d", 30)]:
            cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            total = (
                self.session.query(func.sum(SalesDaily.quantity))
                .filter(
                    SalesDaily.product_id == product_id,
                    SalesDaily.store_id == store_id,
                    SalesDaily.date >= cutoff,
                    SalesDaily.date <= today,
                )
                .scalar()
            ) or 0
            sums[days_label] = total
            avgs[days_label] = total / days

        # 加权日均 (PRD 3.2)
        weighted_avg = avgs["last_7d"] * 0.5 + avgs["last_14d"] * 0.3 + avgs["last_30d"] * 0.2

        # 近 3 天 vs 近 30 天对比（异常检测用）
        cutoff_3d = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
        recent_3d_sum = (
            self.session.query(func.sum(SalesDaily.quantity))
            .filter(
                SalesDaily.product_id == product_id,
                SalesDaily.store_id == store_id,
                SalesDaily.date >= cutoff_3d,
                SalesDaily.date <= today,
            )
            .scalar()
        ) or 0
        recent_3d_avg = recent_3d_sum / 3

        # 30 天总销量（截断约束用）
        total_30d = sums["last_30d"]

        # 缺货天数
        cutoff_30d = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        stockout_days = (
            self.session.query(SalesDaily)
            .filter(
                SalesDaily.product_id == product_id,
                SalesDaily.store_id == store_id,
                SalesDaily.date >= cutoff_30d,
                SalesDaily.quantity == 0,
            )
            .count()
        )

        # 是否有促销
        promo_days = (
            self.session.query(SalesDaily)
            .filter(
                SalesDaily.product_id == product_id,
                SalesDaily.store_id == store_id,
                SalesDaily.date >= cutoff_30d,
                SalesDaily.is_promo_day == 1,
            )
            .count()
        )

        # 店铺开业天数
        days_since_open = None
        if store_open_date:
            days_since_open = (datetime.now() - datetime.strptime(store_open_date, "%Y-%m-%d")).days

        return {
            "weighted_avg_daily": round(weighted_avg, 2),
            "avg_last_7d": round(avgs["last_7d"], 2),
            "avg_last_14d": round(avgs["last_14d"], 2),
            "avg_last_30d": round(avgs["last_30d"], 2),
            "recent_3d_avg": round(recent_3d_avg, 2),
            "total_30d": total_30d,
            "stockout_days_30d": stockout_days,
            "promo_days_30d": promo_days,
            "days_since_open": days_since_open,
        }


# ═══════════════════════════════════════════════════════════
# 规则引擎
# ═══════════════════════════════════════════════════════════

class RuleEngine:
    """补货决策规则引擎."""

    def __init__(self):
        self.data = DataProvider()

    def close(self):
        """释放数据库连接."""
        self.data.close()

    def decide_all(self, store_id: int) -> list[dict]:
        """为指定门店的所有 SKU 生成补货建议."""
        store = self.data.get_store(store_id)
        if not store:
            raise ValueError(f"Store {store_id} not found")

        products = self.data.get_products()
        decisions = []

        for product in products:
            decision = self._decide_one(product, store)
            decisions.append(decision)

        # 按紧急度排序: critical > high > medium > low
        urgency_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        decisions.sort(key=lambda d: urgency_order.get(d["urgency"], 99))

        return decisions

    def _decide_one(self, product: Product, store: Store) -> dict:
        """为单个 SKU 生成补货建议."""
        inv = self.data.get_current_inventory(product.id, store.id)
        sales = self.data.get_sales_summary(product.id, store.id, store.open_date)

        stock = inv["stock_on_hand"]
        in_transit = inv["in_transit"]
        avg_daily = sales["weighted_avg_daily"]
        lead_time = product.lead_time_days
        target_days = product.target_stock_days

        # ── 1. 补货公式 (PRD 3.2) ──
        gap = avg_daily * (lead_time + target_days) - stock - in_transit
        suggested = max(0, round(gap))

        # 约束: ≤0 → 不补
        if suggested <= 0 or avg_daily <= 0:
            suggested = None  # None = 无需补货
        else:
            # 约束: > 30天总销量×3 → 截断
            cap = sales["total_30d"] * 3
            if suggested > cap:
                suggested = cap
            # 约束: 低于起订量 → 标注但不强制
            if suggested < product.min_order_qty:
                suggested = product.min_order_qty  # 至少起订量
            # 整箱取整
            suggested = (suggested // product.min_order_qty) * product.min_order_qty
            if suggested == 0:
                suggested = product.min_order_qty

        # ── 2. 安全库存 + 紧急度 (PRD 3.3) ──
        safety_line = int(avg_daily * (lead_time + 3))  # 缓冲 3 天
        if stock <= safety_line * 0.5:
            urgency = "critical"
        elif stock <= safety_line * 1.0:
            urgency = "high"
        elif stock > safety_line * 1.5:
            urgency = "low"
        else:
            urgency = "medium"

        # ── 3. 异常标记 (PRD 3.4) ──
        anomalies = self._detect_anomalies(sales, avg_daily, stock, lead_time, target_days, product, store)

        # ── 4. 置信度 (PRD 3.5) ──
        days_open = sales.get("days_since_open")
        if days_open is None or days_open < 7:
            confidence = "low"
            confidence_emoji = "purple"
        elif days_open < 30:
            confidence = "medium"
            confidence_emoji = "yellow"
        else:
            confidence = "high"
            confidence_emoji = "green"

        # ── 5. 构建决策因子 JSON ──
        decision_factors = {
            "formula": "suggested_qty = avg_daily_sales * (lead_time + target_days) - stock - in_transit",
            "avg_daily_sales": round(avg_daily, 2),
            "weight_breakdown": {
                "last_7d": sales["avg_last_7d"],
                "last_14d": sales["avg_last_14d"],
                "last_30d": sales["avg_last_30d"],
                "weights": "0.5 / 0.3 / 0.2",
            },
            "lead_time_days": lead_time,
            "target_stock_days": target_days,
            "current_stock": stock,
            "in_transit": in_transit,
            "safety_stock_line": safety_line,
            "safety_ratio": round(stock / safety_line, 2) if safety_line > 0 else None,
            "gap_before_rounding": round(gap, 2),
            "constraint_cap": sales["total_30d"] * 3,
            "anomaly_flags": anomalies,
            "confidence_level": confidence,
            "store_maturity": store.maturity,
            "days_since_open": days_open,
        }

        # ── 6. 生成 reasoning 文本 (规则引擎版) ──
        reasoning = self._build_reasoning(product, suggested, urgency, confidence, decision_factors, anomalies)

        return {
            "product_id": product.id,
            "sku_code": product.sku_code,
            "sku_name": product.name,
            "category": product.category,
            "avg_daily_sales": round(avg_daily, 2),
            "current_stock": stock,
            "in_transit_qty": in_transit,
            "safety_stock_line": safety_line,
            "suggested_qty": suggested,
            "urgency": urgency,
            "confidence": confidence,
            "confidence_emoji": confidence_emoji,
            "anomaly_flags": anomalies,
            "decision_factors": decision_factors,
            "reasoning": reasoning,
        }

    def _detect_anomalies(self, sales: dict, avg_daily: float, stock: int,
                          lead_time: int, target_days: int,
                          product: Product, store: Store) -> list[str]:
        """异常检测 (PRD 3.4 的 7 种异常)."""
        anomalies = []
        recent = sales["recent_3d_avg"]
        avg_30d = sales["avg_last_30d"]

        # 销量飙升: 近3天 > 近30天×2.0
        if avg_30d > 0 and recent > avg_30d * 2.0:
            # 检查是否有促销解释
            if sales["promo_days_30d"] > 0:
                anomalies.append("sales_surge_promo")     # 🟡 促销可解释
            else:
                anomalies.append("sales_surge_unexplained")  # 🟡 无促销→异常

        # 销量骤降: 近3天 < 近30天×0.3
        if avg_30d > 0 and recent < avg_30d * 0.3:
            anomalies.append("sales_drop")

        # 缺货高风险: 可售天数 < 前置期
        if avg_daily > 0:
            days_available = stock / avg_daily if avg_daily > 0 else 999
            if days_available < lead_time:
                anomalies.append("stockout_risk")

        # 库存偏高: 可售天数 > 目标×2.0
        if avg_daily > 0 and stock / avg_daily > target_days * 2.0:
            anomalies.append("overstock")

        # 数据缺失: 近7天无销售
        if sales["avg_last_7d"] == 0:
            anomalies.append("data_missing")

        # 新品: 上架 ≤ 7 天 (用 store maturity 或 days_since_open)
        days_open = sales.get("days_since_open")
        if days_open is not None and days_open <= 7:
            anomalies.append("new_product")

        # 缺货反弹: 30天内有缺货天数
        if sales["stockout_days_30d"] > 0:
            anomalies.append("stockout_recovery")

        return anomalies

    def _build_reasoning(self, product: Product, suggested: int | None,
                         urgency: str, confidence: str,
                         factors: dict, anomalies: list[str]) -> str:
        """生成规则引擎版推理文本."""
        if suggested is None:
            return (
                f"SKU {product.sku_code}（{product.name}）当前库存 {factors['current_stock']} 件，"
                f"加权日均销量 {factors['avg_daily_sales']:.1f} 件，可售天数充足，无需补货。"
            )

        parts = [
            f"SKU {product.sku_code}（{product.name}）：",
            f"加权日均销量 {factors['avg_daily_sales']:.1f} 件 "
            f"(7天:{factors['weight_breakdown']['last_7d']:.1f}×0.5 + "
            f"14天:{factors['weight_breakdown']['last_14d']:.1f}×0.3 + "
            f"30天:{factors['weight_breakdown']['last_30d']:.1f}×0.2)，",
            f"当前库存 {factors['current_stock']} 件，在途 {factors['in_transit']} 件，",
            f"安全库存线 {factors['safety_stock_line']} 件。",
            f"缺口 = {factors['avg_daily_sales']:.1f} × ({factors['lead_time_days']}+{factors['target_stock_days']}) "
            f"- {factors['current_stock']} - {factors['in_transit']} "
            f"= {factors['gap_before_rounding']:.1f} → 建议补货 {suggested} 件。",
        ]

        if anomalies:
            anomaly_labels = {
                "sales_surge_promo": "近期促销导致的销量飙升",
                "sales_surge_unexplained": "近期销量异常飙升（无促销记录，建议运营确认）",
                "sales_drop": "近期销量骤降",
                "stockout_risk": "缺货高风险（可售天数不足前置期）",
                "overstock": "库存偏高（当前库存远超目标天数）",
                "data_missing": "近期无销售数据",
                "new_product": "新品/新店（数据不足，置信度低）",
                "stockout_recovery": "近期有缺货，恢复后销量可能反弹",
            }
            parts.append(f" ⚠️ 异常标记：{'；'.join(anomaly_labels.get(a, a) for a in anomalies)}。")

        return "".join(parts)
