"""LLM Agent 决策 —— DeepSeek 工具调用 + 结构化输出.

在规则引擎结果基础上，让 LLM 审查异常 SKU 并给出修正意见。
不重新计算所有 SKU（成本 + 时间），只聚焦异常 + 修正。
"""

import json
import logging
from openai import OpenAI
from src.config import config
from src.agent.decision.rule_engine import DataProvider

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """你是零售补货审核专家。你会收到规则引擎已经计算好的补货建议列表（含异常标记），你的任务是：

1. 审查异常标记是否合理——规则引擎可能误报或漏报
2. 对异常 SKU 给出修正建议（如果有）
3. 对库存极低（可售天数 < 前置期 × 0.5）的 SKU 给出紧急备注

## 你关注的维度
- **促销影响**：销量飙升 + 有促销 → 合理的，不需要修正；销量飙升 + 无促销 → 标注异常，建议运营确认
- **缺货反弹**：刚恢复供货的 SKU 销量会短期偏高，不是真实趋势
- **新店/新品**：数据不足，规则引擎已经标了低置信度，你不需要再修正数值，但可以给出使用建议
- **库存偏高**：如果库存高了但规则引擎仍然建议补货 → 这不对，检查是否计算有误

## 修正原则
- 只修正你确定规则引擎算错的，不要过度修正
- 如果数据不足以判断，保留规则引擎结果并标注"需人工判断"
- 修正理由必须引用具体数据

## 输出格式
对每个需要关注的 SKU 输出一个 JSON 对象，不需要关注的 SKU 不要输出。
{
  "reviews": [
    {
      "sku_code": "SKU-012",
      "rule_suggested_qty": 500,
      "rule_urgency": "high",
      "rule_anomalies": ["sales_surge_unexplained"],
      "action": "modify",           // "keep" | "modify" | "flag_for_review"
      "adjusted_qty": 400,          // 仅 action=modify 时
      "adjusted_urgency": "high",   // 仅修改时
      "review_reason": "近3天销量飙升(65→160)但无促销记录，可能是异常事件而非趋势。建议保守下单400件，运营确认后再追加。",
      "additional_anomalies": [],   // LLM 发现的规则引擎漏掉的异常
      "confidence": "medium"
    }
  ],
  "summary": "共审查 3 个异常 SKU：1 个建议修正(促销异常)，1 个确认规则引擎判断，1 个标记需人工判断。其余 47 个 SKU 无异常。"
}

## 严格规则
- 仅输出 JSON，不要 markdown 包裹，不要解释
- reviews 数组只包含需要关注的 SKU
- action 只能是 keep / modify / flag_for_review
"""


class LLMAgent:
    """LLM 审查 Agent —— 审查规则引擎结果中的异常 SKU."""

    def __init__(self):
        self.client = OpenAI(
            api_key=config.DEEPSEEK_API_KEY,
            base_url=config.DEEPSEEK_BASE_URL,
        )
        self.data = DataProvider()
        self.model = config.DECISION_LLM_MODEL
        self.temperature = config.DECISION_TEMPERATURE
        self.max_rounds = config.DECISION_MAX_TOOL_ROUNDS

    def close(self):
        """释放数据库连接."""
        self.data.close()

    def review(self, store_id: int, decisions: list[dict]) -> dict | None:
        """审查规则引擎的决策结果，返回修正建议.

        Args:
            store_id: 门店 ID
            decisions: RuleEngine.decide_all() 的返回值

        Returns:
            {"reviews": [...], "summary": "..."} 或 None (LLM 不可用时)
        """
        # 找出异常 SKU
        anomalies = [d for d in decisions if d["anomaly_flags"]]
        if not anomalies:
            logger.info("No anomalies found, skipping LLM review")
            return None

        # 只需处理异常 + 库存极低的 SKU
        criticals = [d for d in decisions if d["urgency"] == "critical" and not d["anomaly_flags"]]
        review_targets = anomalies + criticals

        # 构建消息
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self._build_user_message(store_id, decisions, review_targets)},
        ]

        # Agent Loop
        for round_num in range(self.max_rounds):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",
                    temperature=self.temperature,
                )
            except Exception as e:
                logger.error(f"LLM API error (round {round_num}): {e}")
                return None

            msg = response.choices[0].message

            if msg.tool_calls:
                # 执行工具调用
                messages.append({"role": "assistant", "tool_calls": [
                    {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in msg.tool_calls
                ]})
                for tc in msg.tool_calls:
                    result = self._execute_tool(tc.function.name, json.loads(tc.function.arguments))
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result, ensure_ascii=False),
                    })
            else:
                # LLM 完成审查，解析 JSON
                return self._parse_review(msg.content)

        logger.warning("LLM Agent exceeded max tool rounds")
        return None

    def _build_user_message(self, store_id: int, all_decisions: list[dict], review_targets: list[dict]) -> str:
        """构建发给 LLM 的用户消息."""
        # 门店信息
        store = self.data.get_store(store_id)

        total = len(all_decisions)
        anomaly_count = len([d for d in all_decisions if d["anomaly_flags"]])
        critical_count = len([d for d in all_decisions if d["urgency"] == "critical"])

        lines = [
            f"门店 {store.name}（{store.type}，{store.maturity} 阶段），共 {total} 个 SKU。",
            f"其中 {anomaly_count} 个有异常标记，{critical_count} 个库存极低。",
            "",
            "=== 需要审查的 SKU ===",
        ]

        for d in review_targets:
            lines.append(json.dumps({
                "sku_code": d["sku_code"],
                "sku_name": d["sku_name"],
                "category": d["category"],
                "avg_daily_sales": d["avg_daily_sales"],
                "current_stock": d["current_stock"],
                "safety_stock_line": d["safety_stock_line"],
                "suggested_qty": d["suggested_qty"],
                "urgency": d["urgency"],
                "confidence": d["confidence"],
                "anomaly_flags": d["anomaly_flags"],
                "lead_time_days": d["decision_factors"]["lead_time_days"],
                "target_stock_days": d["decision_factors"]["target_stock_days"],
                "stockout_days_30d": d["decision_factors"].get("stockout_days_30d", 0),
                "days_since_open": d["decision_factors"].get("days_since_open"),
            }, ensure_ascii=False))

        lines.append("")
        lines.append("=== 统计（全部 SKU）===")
        lines.append(f"总 SKU: {total}，正常: {total - anomaly_count}，异常: {anomaly_count}，缺货高风险: {critical_count}")
        lines.append("")
        lines.append("请审查以上异常 SKU，输出 JSON。")

        return "\n".join(lines)

    def _execute_tool(self, name: str, args: dict) -> dict:
        """执行工具调用."""
        if name == "get_sku_detail":
            product_id = args.get("product_id")
            store_id = args.get("store_id")
            # 找到对应的决策
            inv = self.data.get_current_inventory(product_id, store_id)
            # 需要 product 信息
            from src.data.models import get_session, Product
            session = get_session()
            product = session.query(Product).filter(Product.id == product_id).first()
            sales = self.data.get_sales_summary(product_id, store_id)
            return {
                "sku_name": product.name if product else "unknown",
                "category": product.category if product else "unknown",
                "current_stock": inv["stock_on_hand"],
                "in_transit": inv["in_transit"],
                "safety_stock_line": inv["safety_stock_line"],
                "weighted_avg_daily": sales["weighted_avg_daily"],
                "avg_last_7d": sales["avg_last_7d"],
                "avg_last_14d": sales["avg_last_14d"],
                "avg_last_30d": sales["avg_last_30d"],
                "recent_3d_avg": sales["recent_3d_avg"],
                "total_30d": sales["total_30d"],
                "stockout_days_30d": sales["stockout_days_30d"],
                "promo_days_30d": sales["promo_days_30d"],
                "days_since_open": sales["days_since_open"],
            }
        elif name == "get_promotion_detail":
            # 查未来促销计划（Mock 数据中没有，返回空）
            return {"upcoming_promos": [], "note": "No upcoming promotions on record"}
        elif name == "get_store_context":
            from src.data.models import get_session, Store
            session = get_session()
            store = session.query(Store).filter(Store.id == args.get("store_id")).first()
            return {
                "store_name": store.name if store else "unknown",
                "store_type": store.type if store else "unknown",
                "maturity": store.maturity if store else "unknown",
                "days_since_open": (
                    (__import__("datetime").datetime.now() - __import__("datetime").datetime.strptime(store.open_date, "%Y-%m-%d")).days
                    if store and store.open_date else None
                ),
            }
        return {"error": f"Unknown tool: {name}"}

    def _parse_review(self, content: str) -> dict | None:
        """解析 LLM 输出的 JSON."""
        try:
            # 去掉可能的 markdown 包裹
            text = content.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:-1])
            result = json.loads(text)
            if "reviews" in result:
                return result
            logger.warning(f"LLM output missing 'reviews' key: {text[:200]}")
            return None
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM JSON: {e}, content: {content[:300]}")
            return None


# ═══════════════════════════════════════════════
# 工具定义
# ═══════════════════════════════════════════════

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_sku_detail",
            "description": "获取某个 SKU 的完整库存和销售数据，用于深入审查异常",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {"type": "integer", "description": "SKU 的产品 ID"},
                    "store_id": {"type": "integer", "description": "门店 ID"},
                },
                "required": ["product_id", "store_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_promotion_detail",
            "description": "查询门店/SKU 是否有未来促销计划",
            "parameters": {
                "type": "object",
                "properties": {
                    "store_id": {"type": "integer"},
                    "product_id": {"type": "integer"},
                },
                "required": ["store_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_store_context",
            "description": "获取门店的上下文信息（类型、成熟度、开业天数）",
            "parameters": {
                "type": "object",
                "properties": {
                    "store_id": {"type": "integer"},
                },
                "required": ["store_id"],
            },
        },
    },
]
