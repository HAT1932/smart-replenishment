"""对话解释引擎 —— RAG 检索决策快照 + DeepSeek 生成自然语言解释.

核心设计：溯源而非再生。
- 解释来自 suggestions 表中存储的 decision_factors JSON（决策时的真实数据）
- 不是重新调用 LLM 编造理由
- 对比"决策时 vs 现在"的差异（库存可能已变化）
"""

import json
import logging
from datetime import datetime
from openai import OpenAI

from src.config import config
from src.data.models import get_session, Suggestion, Product, Store, InventorySnapshot, Conversation
from sqlalchemy import func

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """你是智能补货系统的对话助手，帮助采购/运营人员理解补货建议。

## 核心原则
1. **数据驱动**：每个结论都引用具体数字，不泛泛而谈
2. **公式透明**：展示"X × Y − Z = 结果"的推演过程
3. **不确定就承认**：数据不足或超出范围时，说"根据现有数据不能确定"
4. **主动提示风险**：低置信度、数据异常时主动标注
5. **可被挑战**：用户说"不对"，你重新分析而不是辩解

## 回答格式
📊 **数据基础** — 关键数字（库存/销量/交期等）
🧮 **推演过程** — 公式代入到结果
🎯 **建议结论** — 补多少 + 紧急度 + 置信度
⚠️ **风险提示** — 什么情况下建议会不准

## 你能做什么
- 解释某个 SKU 的补货逻辑
- What-if 分析："如果促销多 3 天呢？""如果我保守一点呢？"
- 对比两个方案："保守补 400 vs 积极补 650"
- 总结一个门店的整体补货状况

## 你不能做什么
- 修改补货建议（请去审核台操作）
- 预测未来销量（你只解释已有的决策，不替代预测模型）
- 回答和补货无关的问题（友好告知）

## 语气
专业但易懂，像和有经验的同事讨论。允许用户挑战你的判断。"""


class ConversationAgent:
    """对话解释 Agent."""

    def __init__(self):
        self.client = OpenAI(
            api_key=config.DEEPSEEK_API_KEY,
            base_url=config.DEEPSEEK_BASE_URL,
        )
        self.model = config.CONVERSATION_LLM_MODEL
        self.temperature = config.CONVERSATION_TEMPERATURE
        self.max_history = config.CONVERSATION_MAX_HISTORY

    def ask(self, question: str, context: dict = None, session_id: str = None) -> dict:
        """处理用户提问.

        Args:
            question: 用户问题
            context: {"store_id": int, "sku_id": int, "suggestion_id": int} 当前上下文
            session_id: 对话会话 ID（多轮对话）

        Returns:
            {"answer": "回答文本", "session_id": "..."}
        """
        session_id = session_id or f"conv_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        # 检索决策上下文
        decision_context = self._retrieve_context(context) if context else None

        # 构建消息
        messages = self._build_messages(session_id, question, decision_context)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                stream=False,
            )
            answer = response.choices[0].message.content

            # 保存对话记录
            self._save_conversation(session_id, question, answer, context, decision_context)

            return {"answer": answer, "session_id": session_id}
        except Exception as e:
            logger.error(f"Conversation API error: {e}")
            return {
                "answer": f"抱歉，当前无法处理你的问题。（系统错误：{str(e)[:100]}）",
                "session_id": session_id,
            }

    def ask_stream(self, question: str, context: dict = None, session_id: str = None):
        """流式对话（SSE）."""
        session_id = session_id or f"conv_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        decision_context = self._retrieve_context(context) if context else None
        messages = self._build_messages(session_id, question, decision_context)

        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                stream=True,
            )

            full_answer = ""
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    full_answer += token
                    yield token

            # 保存对话记录
            self._save_conversation(session_id, question, full_answer, context, decision_context)

        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"\n\n[系统错误: {str(e)[:100]}]"

    def _retrieve_context(self, context: dict) -> dict | None:
        """RAG 检索：从数据库拉取决策快照和实时数据."""
        session = get_session()
        try:
            result = {}

            # 1. 获取建议决策
            if context.get("suggestion_id"):
                sug = session.query(Suggestion).filter(Suggestion.id == context["suggestion_id"]).first()
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
                    }
                    if sug.decision_factors:
                        try:
                            result["decision_factors"] = json.loads(sug.decision_factors)
                        except json.JSONDecodeError:
                            pass

            # 2. 获取 SKU 信息
            if context.get("sku_id"):
                product = session.query(Product).filter(Product.id == context["sku_id"]).first()
                if product:
                    result["product"] = {
                        "id": product.id,
                        "sku_code": product.sku_code,
                        "name": product.name,
                        "category": product.category,
                        "lead_time_days": product.lead_time_days,
                        "target_stock_days": product.target_stock_days,
                        "min_order_qty": product.min_order_qty,
                    }

            # 3. 获取最新库存（对比决策时 vs 现在）
            if context.get("store_id") and context.get("sku_id"):
                latest_inv = (
                    session.query(InventorySnapshot)
                    .filter(
                        InventorySnapshot.store_id == context["store_id"],
                        InventorySnapshot.product_id == context["sku_id"],
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

            # 4. 获取门店信息
            if context.get("store_id"):
                store = session.query(Store).filter(Store.id == context["store_id"]).first()
                if store:
                    result["store"] = {
                        "id": store.id,
                        "name": store.name,
                        "type": store.type,
                        "maturity": store.maturity,
                    }

            return result if result else None
        finally:
            session.close()

    def _build_messages(self, session_id: str, question: str, decision_context: dict | None) -> list[dict]:
        """构建消息列表（含历史对话）."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # 注入决策上下文
        if decision_context:
            messages.append({
                "role": "system",
                "content": f"当前上下文数据（这就是你要解释的决策依据）：\n```json\n{json.dumps(decision_context, ensure_ascii=False, indent=2)}\n```\n请基于这些实际数据回答问题，不要编造。"
            })

        # 加载历史对话
        session = get_session()
        try:
            history = (
                session.query(Conversation)
                .filter(Conversation.session_id == session_id)
                .order_by(Conversation.created_at.desc())
                .limit(self.max_history * 2)  # user+assistant各算一条
                .all()
            )
            history.reverse()  # 时间正序
            for h in history:
                messages.append({"role": h.role, "content": h.content})
        finally:
            session.close()

        # 当前问题
        messages.append({"role": "user", "content": question})

        return messages

    def _save_conversation(self, session_id: str, question: str, answer: str,
                           context: dict | None, decision_context: dict | None):
        """保存对话记录."""
        session = get_session()
        try:
            ctx_json = json.dumps(decision_context, ensure_ascii=False) if decision_context else None
            now = datetime.now().isoformat()

            # 用户消息
            session.add(Conversation(
                session_id=session_id,
                store_id=context.get("store_id") if context else None,
                product_id=context.get("sku_id") if context else None,
                suggestion_id=context.get("suggestion_id") if context else None,
                role="user",
                content=question,
                context_json=ctx_json,
                created_at=now,
            ))
            # 助手消息
            session.add(Conversation(
                session_id=session_id,
                store_id=context.get("store_id") if context else None,
                product_id=context.get("sku_id") if context else None,
                suggestion_id=context.get("suggestion_id") if context else None,
                role="assistant",
                content=answer,
                context_json=ctx_json,
                created_at=now,
            ))
            session.commit()
        except Exception as e:
            session.rollback()
            logger.warning(f"Failed to save conversation: {e}")
        finally:
            session.close()

    def what_if(self, store_id: int, sku_id: int, scenario: str) -> dict:
        """What-if 场景分析.

        Args:
            scenario: 描述想改变的参数，如 "促销多3天" "交期变成5天" "保守方案"

        Returns:
            {"scenario": str, "analysis": str}
        """
        context = self._retrieve_context({"store_id": store_id, "sku_id": sku_id})
        if not context:
            return {"scenario": scenario, "analysis": "未找到该 SKU 的决策数据，无法进行 What-if 分析。"}

        prompt = f"""基于以下真实决策数据，分析"What-if"场景：

决策数据：
```json
{json.dumps(context, ensure_ascii=False, indent=2)}
```

用户的"What-if"场景："{scenario}"

请分析：
1. 这个变化会影响哪些因子？
2. 建议量会怎么变化？给出估算
3. 风险会变大还是变小？
4. 附带一句建议（是否值得采用这个方案）

如果场景描述不清晰，请求用户澄清具体想改变的参数。"""

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.5,
            )
            return {"scenario": scenario, "analysis": response.choices[0].message.content}
        except Exception as e:
            return {"scenario": scenario, "analysis": f"What-if 分析失败: {e}"}
