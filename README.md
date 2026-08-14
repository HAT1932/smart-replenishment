# AI Agent 智能补货系统

> 面向中腰部连锁零售商的 AI 补货 Agent — Rules+LLM 混合决策，每条建议可追溯、可追问、可挑战。
>
> 从用户调研 → 竞品分析 → PRD → 原型 → **落地实现** 的完整 0→1 产品闭环。

---

## 为什么做这个

**行业痛点**：70.4% 的零售 SaaS 买了不用（ERP 闲置率）。现有补货系统给数字不解释，商家不敢信 → 退回 Excel。

**产品定位**：不拼"算法更准"，拼 **"商家敢用"**。
- AI 给出补货量的同时，展示完整推理链（📊数据 → 🧮推演 → 🎯结论 → ⚠️风险）
- 对话式 What-if："如果促销多 3 天呢？""保守方案怎么做？"
- AI 不可用时自动降级为规则引擎，系统不崩

---

## 系统架构

```
┌──────────────────────────────────────────────────────┐
│                  AI Agent 智能补货                     │
│                                                       │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐ │
│  │  数据层 (M1) │  │ 决策层 (M2) │  │ 对话层 (M3)  │ │
│  │             │  │             │  │              │ │
│  │ Mock POS    │  │ RuleEngine  │  │ RAG 检索     │ │
│  │   ↓         │  │ (Python公式)│  │ 决策快照→LLM │ │
│  │ SQLite DB   │  │   +         │  │ →自然语言    │ │
│  │ 6 tables    │  │ LLM Agent   │  │ + What-if    │ │
│  │ 27K rows    │  │ (DeepSeek)  │  │              │ │
│  └─────────────┘  └──────┬──────┘  └──────┬───────┘ │
│                          │                │          │
│                          └───────┬────────┘          │
│                                  ▼                    │
│  ┌──────────────────────────────────────────────┐    │
│  │            审核工作台 (M4)                     │    │
│  │  FastAPI + Alpine.js + Tailwind CSS          │    │
│  │  审核台 / 详情+对话 / 移动端确认              │    │
│  └──────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────┘
```

- **无 API Key 时**：系统以 `rule_only` 模式运行，所有 PRD 公式（加权销量、安全库存、异常检测）正常输出
- **接入 DeepSeek 后**：LLM 自动审查异常 SKU + 对话解释引擎全面激活

---

## 快速启动

### 环境要求
- Python >= 3.11
- （可选）DeepSeek API Key → [platform.deepseek.com](https://platform.deepseek.com)

### 一键启动

```bash
# Windows
scripts\start.bat

# macOS / Linux
bash scripts/start.sh
```

### 手动步骤

```bash
# 1. 安装依赖
pip install -e ".[dev]"

# 2. 配置（可选，不配也能跑）
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY=sk-xxx

# 3. 生成 Mock 数据
python -m src.data.seed -y

# 4. 生成补货建议
python -m src.agent.decision --store S001

# 5. 启动 Web
uvicorn src.api.main:app --reload
# 打开 http://localhost:8000
```

---

## CLI 演示

```bash
# 规则引擎（无需 API Key）
$ python -m src.agent.decision --store S001 --no-llm

Store: 旗舰店-朝阳区 (flagship, mature)
Mode: rule_only
Total SKUs: 50
Anomalies: 35
Urgent (critical/high): 20

--- CRITICAL (13) ---
  SKU-004 辣条大包装 200g: stock=0 safety=458 [stockout_risk]
  SKU-010 饼干夹心 150g: stock=26 safety=295 [stockout_risk]
  ...

# 对话引擎（需要 API Key）
$ python -m src.agent.conversation
/context S001 SKU-001
为什么建议补 60 件？

📊 数据基础：加权日均销量 42.7 件...
🧮 推演过程：42.7 × (3+7) − 646 − 0 = −219 → 无需补货
   但安全库存线 423，当前 646 高于安全线 → 库存充足
🎯 结论：当前库存可售 15 天，无需补货。
```

---

## 项目结构

```
smart-replenishment/
├── src/
│   ├── config.py                    # 配置（从 .env 读取）
│   ├── data/
│   │   ├── models.py                # SQLAlchemy 模型（6 表）
│   │   ├── generator.py             # Mock POS 数据模拟引擎
│   │   └── seed.py                  # ETL 管道 → 数据库
│   ├── agent/
│   │   ├── decision/                # M2: 决策引擎
│   │   │   ├── rule_engine.py       #   PRD 公式（加权销量/补货/安全库存/异常检测）
│   │   │   ├── llm_agent.py         #   DeepSeek Agent（3 tools + 结构化输出）
│   │   │   ├── engine.py            #   Rules+LLM 混合编排 + 降级切换
│   │   │   └── __main__.py          #   CLI 入口
│   │   └── conversation/            # M3: 对话引擎
│   │       ├── agent.py             #   RAG 检索 + 自然语言解释 + What-if
│   │       └── __main__.py          #   CLI 交互模式
│   ├── api/                         # M4: FastAPI 后端
│   │   ├── main.py                  #   应用入口 + 页面路由
│   │   ├── schemas.py               #   Pydantic 模型
│   │   └── routes/
│   │       ├── suggestions.py       #   补货建议 CRUD（7 个端点）
│   │       └── conversation.py      #   SSE 流式对话
│   └── ui/templates/                # M4: Web 前端（Alpine.js + Tailwind）
│       ├── index.html               #   门店列表
│       ├── workbench.html           #   审核工作台
│       ├── detail.html              #   详情 + 对话面板
│       └── mobile.html              #   门店确认（移动端）
├── scripts/
│   ├── start.sh                     # 一键启动（macOS/Linux）
│   └── start.bat                    # 一键启动（Windows）
├── tests/
├── notebooks/
├── pyproject.toml
└── README.md
```

---

## API 端点

| 方法 | 端点 | 用途 |
|------|------|------|
| POST | `/api/v1/pos/connect` | POS 对接（模拟） |
| POST | `/api/v1/suggestions/generate` | 触发补货建议生成 |
| GET | `/api/v1/suggestions/{store_id}` | 门店建议列表（支持过滤/排序） |
| GET | `/api/v1/suggestions/detail/{id}` | 单条建议详情（含决策因子 JSON） |
| PUT | `/api/v1/suggestions/{id}/confirm` | 确认/修改/驳回 |
| POST | `/api/v1/conversation/ask` | 对话追问（同步） |
| POST | `/api/v1/conversation/ask/stream` | 对话追问（SSE 流式） |

---

## 技术选型

| 组件 | 选型 | 理由 |
|------|------|------|
| LLM | DeepSeek (deepseek-chat) | 中文优异，成本 ~¥31/月（Claude 的 1/50） |
| Agent 框架 | 自建 Agent Loop (OpenAI SDK) | 轻量可控，无框架依赖 |
| 后端 | FastAPI + SQLAlchemy + SQLite | 异步原生，MVP 单文件部署 |
| 前端 | Alpine.js + Tailwind CSS + Chart.js | CDN 零编译，PM 可独立完成 |
| 数据 | SQLite（3 门店 × 50 SKU × 90 天 = 27K 行） | 面试 demo 无需 MySQL |

---

## 决策流程（Rules + LLM 混合）

```
RuleEngine 跑全量 50 SKU
  │
  ├── 正常 SKU (15/50) → 直接采纳，不调 LLM
  │
  └── 异常 SKU (35/50) → LLM Agent 审查
        │
        ├── 促销可解释 → keep（规则正确）
        ├── 无促销飙升 → flag_for_review（建议人工确认）
        ├── 缺货反弹 → modify（修正建议量）
        └── 新店数据不足 → keep（已标低置信度）
              │
              ▼
        合并结果 → suggestions 表 + reasoning
```

---

## 完整产品文档

本项目是 PM 全流程闭环项目，配套文档位于 `../草稿/`：

| 阶段 | 文档 | 说明 |
|------|------|------|
| 发现 | [用户调研报告最终版](../草稿/AI_Agent自动补货_用户调研报告_最终版.md) | 30+ 投诉聚类 + 3 Personas + 情感分析|
| 分析 | [竞品分析最终版](../草稿/AI_Agent自动补货_竞品分析_最终版.md) | 5 家竞品 × 10 维度矩阵 |
| 定义 | [PRD v3.0 精炼版](../草稿/AI_Agent自动补货_PRD_v3.0_精炼版.md) | 8 页面向开发，含 API/DB/Gherkin |
| 设计 | [Figma 原型](https://www.figma.com/design/1YNlZcFwg5hAA0DZYcXsS7/) | 6 页支付宝风格 UI |
| 规划 | [落地实施计划](../草稿/AI_Agent自动补货_落地实施计划_v1.0.md) | 技术选型 + 6 里程碑 |
| 战略 | [价值主张](../草稿/AI_Agent自动补货_价值主张_v1.0.md) | 面向老板/投资人的一页纸 |

---

## 里程碑进度

| M | 名称 | 状态 | 可演示 |
|:--:|------|:--:|:--:|
| M1 | 基础设施 + Mock 数据 | ✅ | — |
| M2 | Rules+LLM 决策引擎 | ✅ | CLI demo |
| M3 | 对话解释引擎 | ✅ | CLI demo（需 API Key） |
| M4 | 审核工作台 Web UI | ✅ | Web demo |
| M5 | 一键启动 + 集成 | ✅ | `scripts/start.bat` |
| M6 | 面试准备 | 🚧 | README done |

---

## License

MIT
