"""M2 CLI 入口: python -m src.agent.decision --store S001"""

import argparse
import json
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

from src.data.models import get_session, Store
from src.agent.decision.engine import DecisionEngine


def main():
    parser = argparse.ArgumentParser(description="AI 补货决策引擎")
    parser.add_argument("--store", type=str, required=True, help="门店代码 (S001/S002/S003)")
    parser.add_argument("--no-llm", action="store_true", help="跳过 LLM 审查，仅用规则引擎")
    parser.add_argument("--output", type=str, help="输出 JSON 文件路径")
    args = parser.parse_args()

    # 查找门店
    session = get_session()
    store = session.query(Store).filter(Store.store_code == args.store).first()
    session.close()

    if not store:
        print(f"Error: Store '{args.store}' not found. Available: S001, S002, S003")
        sys.exit(1)

    print(f"Store: {store.name} ({store.type}, {store.maturity})")
    print(f"Generating replenishment suggestions...")
    print()

    engine = DecisionEngine()

    if args.no_llm:
        # 仅规则引擎
        decisions = engine.rule_engine.decide_all(store.id)
        result = {"store_id": store.id, "total": len(decisions), "mode": "rule_only", "decisions": decisions}
    else:
        result = engine.generate(store.id)

    # 输出摘要
    decisions = result["decisions"]
    mode = result["mode"]
    anomaly_count = len([d for d in decisions if d["anomaly_flags"]])
    urgent_count = len([d for d in decisions if d["urgency"] in ("critical", "high")])
    llm_reviewed = len(result.get("llm_review", {}).get("reviews", []))

    print(f"Mode: {mode}")
    print(f"Total SKUs: {len(decisions)}")
    print(f"Anomalies: {anomaly_count}")
    print(f"Urgent (critical/high): {urgent_count}")
    if llm_reviewed:
        print(f"LLM reviewed: {llm_reviewed} SKUs")
        print(f"LLM summary: {result['llm_review'].get('summary', 'N/A')}")
    print()

    # 按紧急度分组展示
    for urgency, label in [("critical", "CRITICAL"), ("high", "HIGH"), ("medium", "MEDIUM"), ("low", "LOW")]:
        items = [d for d in decisions if d["urgency"] == urgency]
        if items:
            print(f"--- {label} ({len(items)}) ---")
            for d in items[:5]:  # 每组最多 5 条
                qty_str = f"{d['suggested_qty']} pcs" if d["suggested_qty"] else "No order needed"
                flags = f" [{','.join(d['anomaly_flags'])}]" if d["anomaly_flags"] else ""
                print(f"  {d['sku_code']} {d['sku_name']}: {qty_str} | stock={d['current_stock']} safety={d['safety_stock_line']}{flags}")
            if len(items) > 5:
                print(f"  ... and {len(items) - 5} more")
            print()

    # 输出到文件
    if args.output:
        output = []
        for d in result["decisions"]:
            output.append({
                "sku_code": d["sku_code"],
                "sku_name": d["sku_name"],
                "suggested_qty": d["suggested_qty"],
                "urgency": d["urgency"],
                "confidence": d["confidence"],
                "anomaly_flags": d["anomaly_flags"],
                "reasoning": d["reasoning"],
                "decision_factors": d["decision_factors"],
            })
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({"mode": mode, "store_code": args.store, "decisions": output}, f, ensure_ascii=False, indent=2)
        print(f"Full results saved to: {args.output}")


if __name__ == "__main__":
    main()
