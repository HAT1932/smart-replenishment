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
from src.data.models import get_session, Suggestion, Conversation
from src.agent.conversation.vector_retriever import VectorRetriever
from src.agent.conversation.slots import SlotParser

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

## 检索依据说明
注入的「当前上下文数据」可能是：
- 用户显式选中的建议（来自审核台，最精确）
- 系统从自然语言问题中自动匹配到的（看「检索诊断」了解匹配到了哪家店、哪个商品）
回答时基于注入的数据即可；若注入数据为空，明确说明"未找到对应决策数据"。
「similar_historical」是语义检索到的历史相似决策（含反馈），作为参考佐证，
回答"上次同样情况结果如何"时用它；不要把它当成当前建议的推导依据。

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
        # 阶段二（完整版）：向量检索历史决策 + 槽位解析；未配 embedding Key 自动回退结构化
        self.retriever = VectorRetriever()
        self.parser = SlotParser(self.retriever)

    # ── 自然语言预处理（槽位填充 + 反问澄清，零 LLM）──

    def _preprocess(self, question: str, context: dict | None) -> tuple:
        """解析问题，返回 (context, diagnostics, rule_answer).

        - context 非空（显式传入）→ 直接走 LLM
        - rule_answer 非空 → 无需调 LLM（问候/无关/缺槽反问/无匹配）
        - 其余 → 槽位已解析，context 已组装，带 diagnostics 调 LLM
        """
        if context:
            return context, None, None

        parsed = self.parser.parse(question)

        if parsed["intent"] == "greeting":
            return None, None, (
                "你好！我是补货助手，可以帮你解释每条补货建议的逻辑。"
                "例如问「为什么可乐补这么多」「旗舰店缺货的有哪些」，或直接说 SKU 编号。"
            )
        if parsed["intent"] == "outofscope":
            return None, None, (
                "我是补货助手，只处理补货相关的问题（解释建议、What-if 分析、门店盘点）。"
                "补货之外的问题我帮不上忙。"
            )
        if parsed["needs_clarification"]:
            return None, None, (
                "为了准确回答，我需要确认一下：\n\n" + self.parser.build_clarification(parsed)
            )

        resolved_ctx = self._slots_to_context(parsed)
        if not resolved_ctx:
            return None, None, (
                "没找到对应的补货建议。试试换个说法，或直接说 SKU 编号（如 SKU-001）、商品名（如可乐）。"
            )

        diagnostics = {
            "intent": parsed["intent"],
            "store": parsed["slots"].get("store", {}).get("name") if parsed["slots"].get("store") else None,
            "sku": f"{parsed['slots']['sku']['name']}（{parsed['slots']['sku']['sku_code']}）"
            if parsed["slots"].get("sku") else None,
            "time_range": parsed["slots"].get("time_range"),
        }
        return resolved_ctx, diagnostics, None

    def _slots_to_context(self, parsed: dict) -> dict | None:
        """把已解析的槽位组装成 context dict."""
        slots = parsed["slots"]
        store = slots.get("store")
        sku = slots.get("sku")
        if not store or not sku:
            return None
        sug_id = self._latest_suggestion_id(store["id"], sku["id"])
        return {"store_id": store["id"], "sku_id": sku["id"], "suggestion_id": sug_id}

    def _latest_suggestion_id(self, store_id: int, sku_id: int) -> int | None:
        session = get_session()
        try:
            sug = (
                session.query(Suggestion)
                .filter(Suggestion.store_id == store_id, Suggestion.product_id == sku_id)
                .order_by(Suggestion.generated_at.desc())
                .first()
            )
            return sug.id if sug else None
        finally:
            session.close()

    # ── 主入口 ─────────────────────────────────────

    def ask(self, question: str, context: dict = None, session_id: str = None) -> dict:
        """处理用户提问.

        Args:
            question: 用户问题
            context: {"store_id": int, "sku_id": int, "suggestion_id": int} 当前上下文
            session_id: 对话会话 ID（多轮对话）

        Returns:
            {"answer": "回答文本", "session_id": "...", "intent"?: 意图, "rule"?: 是否规则应答}
        """
        session_id = session_id or f"conv_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        # 自然语言预处理：缺槽反问 / 问候 / 无关 → 规则应答（不烧 LLM）
        resolved_ctx, diagnostics, rule_answer = self._preprocess(question, context)
        if rule_answer:
            self._save_conversation(session_id, question, rule_answer, None, None)
            return {"answer": rule_answer, "session_id": session_id, "rule": True}

        context = resolved_ctx
        decision_context = self._retrieve_context(context) if context else None
        messages = self._build_messages(session_id, question, decision_context, diagnostics)

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

            result = {"answer": answer, "session_id": session_id}
            if diagnostics:
                result["context"] = context
                result["diagnostics"] = diagnostics
            return result
        except Exception as e:
            logger.error(f"Conversation API error: {e}")
            return {
                "answer": f"抱歉，当前无法处理你的问题。（系统错误：{str(e)[:100]}）",
                "session_id": session_id,
            }

    def ask_stream(self, question: str, context: dict = None, session_id: str = None):
        """流式对话（SSE）."""
        session_id = session_id or f"conv_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        # 规则应答路径：直接产出完整答案
        resolved_ctx, diagnostics, rule_answer = self._preprocess(question, context)
        if rule_answer:
            self._save_conversation(session_id, question, rule_answer, None, None)
            yield rule_answer
            return

        context = resolved_ctx
        decision_context = self._retrieve_context(context) if context else None
        messages = self._build_messages(session_id, question, decision_context, diagnostics)

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
        """检索决策上下文：委托给可插拔检索器（阶段一结构化 / 阶段二向量）."""
        if not context:
            return None
        return self.retriever.build_context(
            store_id=context.get("store_id"),
            sku_id=context.get("sku_id"),
            suggestion_id=context.get("suggestion_id"),
        )

    def _build_messages(self, session_id: str, question: str, decision_context: dict | None,
                        diagnostics: dict | None = None) -> list[dict]:
        """构建消息列表（含历史对话）."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # 注入检索诊断（自然语言自动匹配时可见）
        if diagnostics:
            messages.append({
                "role": "system",
                "content": (
                    "检索诊断（系统根据你的问题自动匹配到的对象）：\n"
                    f"```json\n{json.dumps(diagnostics, ensure_ascii=False, indent=2)}\n```\n"
                    "若与用户意图不符，请先指出你理解到的对象，再基于上下文数据回答。"
                ),
            })

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
