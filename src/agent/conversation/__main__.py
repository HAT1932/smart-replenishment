"""M3 CLI 入口: python -m src.agent.conversation

自然语言输入由 ConversationAgent 内部完成「意图分类 + 槽位填充 + 反问澄清」：
  - 缺信息 → 主动反问（零 LLM 成本）
  - 解析成功 → 检索决策快照 + LLM 解释
CLI 只负责命令（/list /context /reset /whatif）和展示。
"""

import logging

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

from src.data.models import get_session, Store, Product, Suggestion
from src.agent.conversation.agent import ConversationAgent


def print_banner():
    print("=" * 60)
    print("  AI 补货助手 — 对话模式（阶段一：槽位填充 + 反问澄清）")
    print("  直接提问即可，我会自动找到门店和 SKU，缺信息会反问")
    print("  输入 /list 查看数据, /help 帮助, /exit 退出")
    print("=" * 60)
    print()


def print_help():
    print("""
直接问（无需设置上下文，缺信息会自动反问）：
  为什么 SKU-001 补 60 件？        → 精确匹配
  薯片为什么补这么多？             → 命中多个口味 → 反问确认
  旗舰店的可乐为什么补这么多？    → 自动解析门店+SKU
  旗舰店缺货的有哪些？             → 门店盘点（意图：summarize）
  如果促销多 3 天呢？              → What-if（缺实体 → 反问）

命令：
  /list           → 查看所有门店和最近建议
  /context S001 SKU-001  → 手动锁定上下文（可选）
  /reset          → 清除上下文，恢复自动解析
  /whatif 交期变5天  → What-if（需要上下文）
  /exit           → 退出
""")


def main():
    # Windows 控制台默认 GBK，强制 UTF-8 避免中文乱码 / 代理对错误（Git Bash/Windows Terminal 均发 UTF-8）
    import sys
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print_banner()

    agent = ConversationAgent()
    current_context = None
    session_id = None

    while True:
        try:
            user_input = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue

        # 命令处理
        if user_input.startswith("/exit"):
            print("再见！")
            break
        elif user_input.startswith("/help"):
            print_help()
            continue
        elif user_input.startswith("/list"):
            list_context()
            continue
        elif user_input.startswith("/reset"):
            current_context = None
            session_id = None
            print("✓ 上下文已清除，下次提问会重新自动解析。")
            continue
        elif user_input.startswith("/context"):
            parts = user_input.split()
            if len(parts) >= 3:
                store_code, sku_code = parts[1], parts[2]
                session = get_session()
                store = session.query(Store).filter(Store.store_code == store_code).first()
                product = session.query(Product).filter(Product.sku_code == sku_code).first()
                if store and product:
                    sug = (
                        session.query(Suggestion)
                        .filter(Suggestion.store_id == store.id, Suggestion.product_id == product.id)
                        .order_by(Suggestion.generated_at.desc())
                        .first()
                    )
                    current_context = {
                        "store_id": store.id,
                        "sku_id": product.id,
                        "suggestion_id": sug.id if sug else None,
                    }
                    print(f"✓ 锁定: {store.name} / {product.name}")
                    if sug:
                        print(f"  建议: {sug.suggested_qty}pcs ({sug.urgency})")
                else:
                    print(f"✗ 未找到: {store_code} / {sku_code}")
                session.close()
            else:
                print("用法: /context <store_code> <sku_code>")
            continue
        elif user_input.startswith("/whatif"):
            scenario = user_input[len("/whatif"):].strip()
            if not current_context:
                print("请先让系统匹配一个 SKU（比如问'为什么可乐补这么多'），然后再 /whatif。")
                continue
            print("分析中...")
            result = agent.what_if(current_context["store_id"], current_context["sku_id"], scenario)
            print(f"\n助手: {result['analysis']}")
            continue

        # ── 对话：agent 内部完成 意图分类 + 槽位填充 + 反问 ──
        print("思考中...")
        result = agent.ask(user_input, context=current_context, session_id=session_id)
        session_id = result["session_id"]

        # 自动解析成功 → 展示匹配对象 + 记住上下文
        if result.get("context"):
            current_context = result["context"]
            d = result.get("diagnostics", {})
            shown = []
            if d.get("store"):
                shown.append(f"门店={d['store']}")
            if d.get("sku"):
                shown.append(f"SKU={d['sku']}")
            if shown:
                print(f"  → 自动解析: {' | '.join(shown)}")

        print(f"\n助手: {result['answer']}")


def list_context():
    """列出可用的门店和最近建议."""
    session = get_session()
    stores = session.query(Store).all()
    print("\n可用的门店：")
    for s in stores:
        sug_count = session.query(Suggestion).filter(Suggestion.store_id == s.id).count()
        critical = session.query(Suggestion).filter(
            Suggestion.store_id == s.id,
            Suggestion.urgency == "critical"
        ).count()
        print(f"  {s.store_code}: {s.name} ({s.type}, {sug_count} 建议, {critical} 缺货风险)")

    recent = session.query(Suggestion).order_by(Suggestion.generated_at.desc()).limit(8).all()
    if recent:
        print("\n最近建议（可以直接问 '为什么 SKU-xxx 补这么多'）：")
        for sug in recent:
            product = session.query(Product).filter(Product.id == sug.product_id).first()
            store = session.query(Store).filter(Store.id == sug.store_id).first()
            pname = product.name if product else "?"
            scode = store.store_code if store else "?"
            qty_str = f"{sug.suggested_qty}pcs" if sug.suggested_qty > 0 else "不补"
            print(f"  {scode} {product.sku_code if product else '?'} {pname}: {qty_str} ({sug.urgency})")
    session.close()
    print()


if __name__ == "__main__":
    main()
