"""M3 CLI 入口: python -m src.agent.conversation"""

import re
import logging

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

from src.data.models import get_session, Store, Product, Suggestion
from src.agent.conversation.agent import ConversationAgent


def print_banner():
    print("=" * 60)
    print("  AI 补货助手 — 对话模式")
    print("  直接提问即可，我会自动找到你关心的 SKU")
    print("  输入 /list 查看数据, /help 帮助, /exit 退出")
    print("=" * 60)
    print()


def print_help():
    print("""
直接问（无需设置上下文）：
  为什么 SKU-001 建议补 60 件？  → 自动匹配
  薯片为什么补这么多？           → 模糊搜索
  旗舰店缺货的有哪些？           → 跨门店搜索

命令：
  /list           → 查看所有门店和最近建议
  /context S001 SKU-001  → 手动锁定上下文（可选）
  /reset          → 清除上下文，恢复自动匹配
  /whatif 交期变5天  → What-if（需要上下文）
  /exit           → 退出
""")


def resolve_context(question: str, current_context: dict | None) -> dict | None:
    """智能解析上下文：从用户问题中搜索匹配的 SKU.

    如果已有上下文且用户的问题明显是追问（不含新的 SKU/门店引用），保留原上下文。
    否则从数据库搜索匹配。

    Returns: 找到的 context dict 或 None.
    """
    # 如果已有上下文且问题看起来是追问 → 保留
    if current_context:
        # 追问特征：代词、指代、简短、包含"为什么/怎么/如果"
        followup_patterns = [
            r'^(为什么|怎么|如果|那|这个|它|这|可以|能|帮我|再)',
            r'^.{1,15}$',  # 极短问题大概率是追问
        ]
        has_new_reference = bool(re.search(r'SKU-\d{3}|S00\d|哪个|哪些|门店', question))
        if not has_new_reference:
            for pat in followup_patterns:
                if re.search(pat, question):
                    return current_context  # 保留，不切换

    session = get_session()
    try:
        matches = []

        # 策略 1: 精确匹配 SKU 编号 如 SKU-001, SKU-012
        sku_codes = re.findall(r'SKU-(\d{1,3})', question, re.IGNORECASE)
        for code_num in sku_codes:
            code = f"SKU-{int(code_num):03d}"
            products = session.query(Product).filter(Product.sku_code == code).all()
            for p in products:
                sug = (
                    session.query(Suggestion)
                    .filter(Suggestion.product_id == p.id)
                    .order_by(Suggestion.generated_at.desc())
                    .first()
                )
                if sug:
                    store = session.query(Store).filter(Store.id == sug.store_id).first()
                    matches.append({
                        "context": {"store_id": sug.store_id, "sku_id": p.id, "suggestion_id": sug.id},
                        "label": f"[{store.store_code if store else '?'}] {p.name} ({p.sku_code}) → {sug.suggested_qty}pcs",
                        "score": 100,
                    })

        # 策略 2: 模糊搜索 SKU 名称中的关键词
        if len(matches) == 0:
            products = session.query(Product).all()
            for p in products:
                # 检查产品名中是否有词出现在问题里
                name_words = set(re.findall(r'[一-鿿]+|[a-zA-Z]+', p.name.lower()))
                question_words = set(re.findall(r'[一-鿿]+|[a-zA-Z]+', question.lower()))
                overlap = name_words & question_words
                if not overlap:
                    continue
                sug = (
                    session.query(Suggestion)
                    .filter(Suggestion.product_id == p.id)
                    .order_by(Suggestion.generated_at.desc())
                    .first()
                )
                if sug:
                    store = session.query(Store).filter(Store.id == sug.store_id).first()
                    matches.append({
                        "context": {"store_id": sug.store_id, "sku_id": p.id, "suggestion_id": sug.id},
                        "label": f"[{store.store_code if store else '?'}] {p.name} ({p.sku_code}) → {sug.suggested_qty}pcs",
                        "score": len(overlap) * 10,
                    })

        # 策略 3: 按问题中提到的数字搜索建议量
        if len(matches) == 0:
            numbers = re.findall(r'(\d{2,4})\s*件', question)
            for num_str in numbers:
                target = int(num_str)
                # ±20% 容差
                sugs = (
                    session.query(Suggestion)
                    .filter(
                        Suggestion.suggested_qty >= target * 0.8,
                        Suggestion.suggested_qty <= target * 1.2,
                        Suggestion.suggested_qty > 0,
                    )
                    .limit(10)
                    .all()
                )
                for sug in sugs:
                    product = session.query(Product).filter(Product.id == sug.product_id).first()
                    store = session.query(Store).filter(Store.id == sug.store_id).first()
                    matches.append({
                        "context": {"store_id": sug.store_id, "sku_id": sug.product_id, "suggestion_id": sug.id},
                        "label": f"[{store.store_code if store else '?'}] {product.name if product else '?'} → {sug.suggested_qty}pcs",
                        "score": 90 - abs(sug.suggested_qty - target),
                    })

        # 策略 4: 门店关键词匹配
        if len(matches) == 0:
            for keyword, store_code in [("旗舰", "S001"), ("社区", "S002"), ("望京", "S003"), ("新店", "S003")]:
                if keyword in question:
                    store = session.query(Store).filter(Store.store_code == store_code).first()
                    if store:
                        sugs = (
                            session.query(Suggestion)
                            .filter(Suggestion.store_id == store.id)
                            .order_by(
                                __import__('sqlalchemy').case(
                                    (Suggestion.urgency == "critical", 0),
                                    (Suggestion.urgency == "high", 1),
                                    else_=2,
                                )
                            )
                            .limit(5)
                            .all()
                        )
                        for sug in sugs:
                            product = session.query(Product).filter(Product.id == sug.product_id).first()
                            matches.append({
                                "context": {"store_id": sug.store_id, "sku_id": sug.product_id, "suggestion_id": sug.id},
                                "label": f"[{store_code}] {product.name if product else '?'} → {sug.suggested_qty}pcs ({sug.urgency})",
                                "score": 50,
                            })

        # 去重 + 按分数排序
        seen = set()
        unique = []
        for m in sorted(matches, key=lambda x: x["score"], reverse=True):
            key = (m["context"]["store_id"], m["context"]["sku_id"])
            if key not in seen:
                seen.add(key)
                unique.append(m)

        if len(unique) == 1:
            # 唯一匹配 → 自动设置
            ctx = unique[0]["context"]
            store = session.query(Store).filter(Store.id == ctx["store_id"]).first()
            product = session.query(Product).filter(Product.id == ctx["sku_id"]).first()
            print(f"  → 自动匹配: [{store.store_code if store else '?'}] {product.name if product else '?'}")
            session.close()
            return ctx

        if len(unique) > 1:
            # 多个匹配 → 让用户选（但选第一个最可能的作为默认）
            print(f"\n  找到 {len(unique)} 个可能的匹配，我猜你问的是第一个：")
            for i, m in enumerate(unique[:5]):
                marker = "→" if i == 0 else "  "
                print(f"  {marker} {m['label']}")
            if len(unique) > 5:
                print(f"  ... 还有 {len(unique)-5} 个匹配")
            print(f"  如果不对，请用 /context 指定或换个说法。\n")
            session.close()
            return unique[0]["context"]  # 默认选最可能的

        session.close()
        return None

    finally:
        # 确保 session 关闭
        if session:
            session.close()
    return None


def main():
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
            from src.agent.conversation.__main__ import list_context
            list_context()
            continue
        elif user_input.startswith("/reset"):
            current_context = None
            session_id = None
            print("✓ 上下文已清除，下次提问会重新匹配。")
            continue
        elif user_input.startswith("/context"):
            parts = user_input.split()
            if len(parts) >= 3:
                store_code = parts[1]
                sku_code = parts[2]

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
                print("请先让系统匹配一个 SKU（比如问'为什么SKU-001补这么多'），然后再 /whatif。")
                continue
            print("分析中...")
            result = agent.what_if(current_context["store_id"], current_context["sku_id"], scenario)
            print(f"\n助手: {result['analysis']}")
            continue

        # ── 智能上下文解析 ──
        resolved = resolve_context(user_input, current_context)
        if resolved:
            current_context = resolved
        elif not current_context:
            print("  没找到匹配的 SKU。试试 /list 看看有哪些数据，或者直接说 SKU 编号。")

        # ── 对话 ──
        print("思考中...")
        result = agent.ask(user_input, context=current_context, session_id=session_id)
        session_id = result["session_id"]

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
