"""槽位解析 —— 从自然语言问题中提取 意图 + 门店/SKU/时间/场景.

参考智能客服的槽位填充（Slot Filling）：
1. 先判意图（explain/whatif/compare/summarize/feedback/greeting/outofscope）
2. 再抽槽位（门店、商品、时间范围、What-if 场景）
3. 判定缺哪些必填槽 → 返回「反问清单」，由上层生成澄清问题

纯规则实现（零 LLM 调用）→ 反问澄清不烧 token，符合"减少无效 LLM 调用"。
"""

import re

from src.agent.conversation.retriever import Retriever


# ── 意图关键词 ────────────────────────────────────

INTENT_KEYWORDS = {
    "explain": [
        "为什么", "怎么算", "怎么来", "解释", "依据", "原因", "逻辑",
        "凭什么", "是什么", "分析", "说明", "怎么得出", "根据",
    ],
    "whatif": [
        "如果", "要是", "假设", "假如", "保守", "激进", "方案", "会不会",
        "提升", "降低", "促销", "多3天", "少", "多补", "少补",
    ],
    "compare": ["对比", "相比", "区别", "哪个好", "还是", "选择", "哪种"],
    "summarize": ["总结", "整体", "哪些", "盘点", "汇总", "多少", "情况", "缺货", "高风险"],
    "feedback": ["上次", "结果", "效果", "准不准", "对不对", "验证", "学习", "准确吗"],
    "greeting": ["你好", "您好", "hi", "hello", "嗨", "在吗", "早上好", "下午好"],
    "outofscope": ["天气", "股票", "电影", "写诗", "唱歌", "笑话", "你是谁", "新闻", "菜谱"],
}

# 意图 → 必填槽位
REQUIRED_SLOTS = {
    "explain": ["store", "sku"],
    "whatif": ["store", "sku", "scenario"],
    "compare": ["store", "sku"],
    "summarize": ["store"],
    "feedback": ["store", "sku"],
    "greeting": [],
    "outofscope": [],
}

# 时间范围解析
TIME_PATTERNS = [
    (r"上周|上一周", "上周"),
    (r"这周|本周", "本周"),
    (r"上月|上个月", "上月"),
    (r"本月|这个月", "本月"),
    (r"最近\s*7\s*天|近一周|最近一周", "最近7天"),
    (r"最近\s*30\s*天|近一月", "最近30天"),
    (r"昨天|昨日", "昨天"),
    (r"今天|今日", "今天"),
]


class SlotParser:
    """意图分类 + 槽位提取."""

    def __init__(self, retriever: Retriever):
        self.retriever = retriever

    # ── 意图分类 ───────────────────────────────────

    def classify_intent(self, question: str) -> str:
        # 问候 / 无关优先（避免"为什么"类词误伤）
        for kw in INTENT_KEYWORDS["greeting"]:
            if kw in question and len(question) <= 12:
                return "greeting"
        for kw in INTENT_KEYWORDS["outofscope"]:
            if kw in question:
                return "outofscope"

        scores = {intent: 0 for intent in INTENT_KEYWORDS}
        for intent, kws in INTENT_KEYWORDS.items():
            for kw in kws:
                if kw in question:
                    scores[intent] += 1
        # 无命中默认 explain（最常用）
        best = max(scores, key=lambda k: scores[k])
        return best if scores[best] > 0 else "explain"

    # ── 槽位提取 ───────────────────────────────────

    def extract_slots(self, question: str) -> dict:
        """返回 {store, sku, time_range, scenario, sku_matches, store_matches}.

        对已知小目录（门店个位数、SKU 几十级）采用「目录式匹配」：
        把每个商品/门店的名称切成语义片段，看片段是否出现在问题中。
        比通用切词更稳，且天然支持"可乐""旗舰店的薯片"这类口语问法。
        """
        slots = {
            "store": None,          # {"id","store_code","name","score"}
            "sku": None,            # {"id","sku_code","name","score"}
            "time_range": None,
            "scenario": None,
            "store_matches": [],
            "sku_matches": [],
        }

        # 门店：名称片段(含2字窗口) / 编号 出现在问题中即命中
        for st in self.retriever.all_stores():
            tokens = self._name_tokens(st["name"]) + [st["store_code"]]
            if any(tok in question for tok in tokens if tok):
                slots["store_matches"].append(st)
        if len(slots["store_matches"]) == 1:
            slots["store"] = slots["store_matches"][0]

        # 商品：名称片段(含2字窗口) / SKU 编号 出现在问题中即命中
        sku_seen: dict[int, dict] = {}
        for p in self.retriever.all_products():
            tokens = self._name_tokens(p["name"]) + [p["sku_code"]]
            hits = sum(1 for tok in set(tokens) if tok and tok in question)
            if hits:
                p["_hits"] = hits
                sku_seen[p["id"]] = p
        sku_matches = sorted(
            sku_seen.values(), key=lambda x: (x["_hits"], x["score"]), reverse=True
        )[:5]
        slots["sku_matches"] = sku_matches
        if len(sku_matches) == 1:
            slots["sku"] = sku_matches[0]

        # 时间范围
        for pattern, label in TIME_PATTERNS:
            if re.search(pattern, question):
                slots["time_range"] = label
                break

        # What-if 场景：去掉意图词/实体词后的剩余
        if self.classify_intent(question) == "whatif":
            slots["scenario"] = self._extract_scenario(question)

        return slots

    def _name_tokens(self, text: str) -> list[str]:
        """把商品/门店名切成检索片段：完整 CJK 段 + 2 字滑动窗口.

        例「薯片原味 150g」→ ["薯片原味", "薯片", "片原", "原味"]，
        这样用户只说「薯片」也能命中。
        """
        tokens = []
        for run in re.findall(r"[一-鿿]{2,}", text):
            tokens.append(run)
            tokens.extend(run[i:i + 2] for i in range(len(run) - 1))
        return tokens

    def _extract_scenario(self, question: str) -> str:
        """whatif 场景 = 问题去掉意图关键词."""
        s = question
        for kws in INTENT_KEYWORDS.values():
            for kw in kws:
                s = s.replace(kw, "")
        s = s.replace("呢", "").replace("？", "").replace("?", "").strip("，。！! ")
        return s or question

    # ── 缺槽判定 ───────────────────────────────────

    def missing_slots(self, intent: str, slots: dict) -> list[str]:
        """返回缺失的必填槽名（store/sku/scenario）."""
        required = REQUIRED_SLOTS.get(intent, [])
        missing = []
        for slot in required:
            if slot == "store" and not slots.get("store"):
                missing.append("store")
            elif slot == "sku" and not slots.get("sku"):
                missing.append("sku")
            elif slot == "scenario" and not slots.get("scenario"):
                missing.append("scenario")
        return missing

    # ── 完整解析 ───────────────────────────────────

    def parse(self, question: str) -> dict:
        """解析问题 → {intent, slots, missing, needs_clarification}."""
        intent = self.classify_intent(question)
        slots = self.extract_slots(question)
        missing = self.missing_slots(intent, slots)
        return {
            "intent": intent,
            "slots": slots,
            "missing": missing,
            "needs_clarification": bool(missing),
        }

    # ── 澄清文案（反问，纯规则零 LLM）────────────────

    def build_clarification(self, parsed: dict) -> str:
        """根据缺失槽位生成反问话术，尽量给出可选项."""
        missing = parsed["missing"]
        if not missing:
            return ""
        lines = []
        for slot in missing:
            if slot == "store":
                matches = parsed["slots"]["store_matches"]
                if matches:
                    opts = "、".join(f"「{m['name']}」" for m in matches[:3])
                    lines.append(f"你是指 {opts} 里的哪一家门店？")
                else:
                    lines.append("请问是哪家门店？例如「旗舰店-朝阳区」「社区店-望京」")
            elif slot == "sku":
                matches = parsed["slots"]["sku_matches"]
                if matches:
                    opts = "、".join(f"「{m['name']}（{m['sku_code']}）」" for m in matches[:4])
                    lines.append(f"你指的是哪个商品？我找到了 {opts}，请确认。")
                else:
                    lines.append("请告诉我要查的商品名称或 SKU 编号（如 SKU-001、可乐）。")
            elif slot == "scenario":
                lines.append("请描述你想尝试的 What-if 场景，例如「促销多 3 天」「交期变成 5 天」「保守一点」。")
        return "\n".join(lines)
