"""决策引擎主控 —— RuleEngine + LLMAgent 混合编排 + 降级."""

import json
import logging
from datetime import datetime

from src.data.models import get_session, Suggestion
from src.agent.decision.rule_engine import RuleEngine
from src.agent.decision.llm_agent import LLMAgent

logger = logging.getLogger(__name__)


class DecisionEngine:
    """补货决策引擎.

    流程:
    1. RuleEngine 计算所有 SKU 的基准建议
    2. 如果有异常 SKU → LLMAgent 审查并给出修正
    3. 合并结果 → 写入 suggestions 表
    4. LLM 不可用时 → 规则引擎结果直接使用（降级）
    """

    def __init__(self):
        self.rule_engine = RuleEngine()
        self.llm_agent = LLMAgent()

    def generate(self, store_id: int) -> dict:
        """为指定门店生成补货建议."""
        try:
            # Step 1: 规则引擎跑全量
            logger.info(f"Step 1: RuleEngine for store {store_id}")
            decisions = self.rule_engine.decide_all(store_id)

            # Step 2: LLM 审查异常
            llm_result = None
            mode = "rule_only"

            try:
                llm_result = self.llm_agent.review(store_id, decisions)
                if llm_result and llm_result.get("reviews"):
                    mode = "rule_plus_llm"
                    decisions = self._apply_reviews(decisions, llm_result)
                    logger.info(f"LLM reviewed {len(llm_result['reviews'])} SKUs")
                else:
                    logger.info("LLM returned no reviews (no anomalies or API unavailable)")
            except Exception as e:
                logger.warning(f"LLM review failed, falling back to rules-only: {e}")
                mode = "rule_fallback"

            # Step 3: 写入数据库
            self._save_decisions(store_id, decisions, mode)

            return {
                "store_id": store_id,
                "total": len(decisions),
                "mode": mode,
                "decisions": decisions,
                "llm_review": llm_result,
            }
        finally:
            self.rule_engine.close()
            self.llm_agent.close()

    def _apply_reviews(self, decisions: list[dict], llm_result: dict) -> list[dict]:
        """将 LLM 修正意见应用到决策结果."""
        review_map = {r["sku_code"]: r for r in llm_result.get("reviews", [])}

        for d in decisions:
            review = review_map.get(d["sku_code"])
            if not review:
                continue

            action = review.get("action", "keep")

            if action == "modify" and review.get("adjusted_qty"):
                d["suggested_qty"] = review["adjusted_qty"]
                d["reasoning"] += f" [LLM修正: {review.get('review_reason', '')}]"
                d["decision_factors"]["llm_adjustment"] = {
                    "original_qty": review["rule_suggested_qty"],
                    "adjusted_qty": review["adjusted_qty"],
                    "reason": review.get("review_reason", ""),
                }
                if review.get("adjusted_urgency"):
                    d["urgency"] = review["adjusted_urgency"]

            elif action == "flag_for_review":
                d["anomaly_flags"].append("needs_manual_review")
                d["reasoning"] += f" [LLM标记需人工判断: {review.get('review_reason', '')}]"

            # 追加 LLM 发现的异常
            for extra in review.get("additional_anomalies", []):
                if extra not in d["anomaly_flags"]:
                    d["anomaly_flags"].append(extra)

        return decisions

    def _save_decisions(self, store_id: int, decisions: list[dict], mode: str):
        """保存决策结果到 suggestions 表."""
        session = get_session()
        try:
            now = datetime.now().isoformat()
            count = 0
            for d in decisions:
                sug = Suggestion(
                    store_id=store_id,
                    product_id=d["product_id"],
                    suggested_qty=d["suggested_qty"] or 0,
                    urgency=d["urgency"],
                    confidence=d["confidence"],
                    confidence_level=d["confidence_emoji"],
                    anomaly_type=",".join(d["anomaly_flags"]) if d["anomaly_flags"] else None,
                    decision_factors=json.dumps(d["decision_factors"], ensure_ascii=False),
                    reasoning=f"[{mode}] {d['reasoning']}",
                    status="pending",
                    generated_at=now,
                )
                session.add(sug)
                count += 1
            session.commit()
            logger.info(f"Saved {count} suggestions to DB (mode={mode})")
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
