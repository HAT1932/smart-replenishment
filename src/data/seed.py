"""ETL 管道: Mock 数据 → 数据库."""

import json
from datetime import datetime
from sqlalchemy.orm import Session

from src.data.models import (
    init_db, get_session, Store, Product,
    SalesDaily, InventorySnapshot, Suggestion, Conversation
)
from src.data.generator import generate_all, SkuConfig, SalesRecord, InventoryRecord


def seed_database():
    """一键: 删旧数据 → 建表 → 生成 → 写入."""
    init_db()
    session = get_session()

    try:
        # 清空旧数据（按外键依赖顺序）
        for table in [InventorySnapshot, SalesDaily, Suggestion, Conversation, Product, Store]:
            session.query(table).delete()
        session.commit()

        # 生成 Mock 数据
        skus, stores, sales_records, inv_records = generate_all()

        # ── 写入门店 ──
        store_map = {}  # code → id
        for st in stores:
            obj = Store(
                store_code=st["code"], name=st["name"],
                type=st["type"], area_sqm=st["area"],
                district=st["district"], maturity=st["maturity"],
                open_date=st["open_date"],
            )
            session.add(obj)
            session.flush()
            store_map[st["code"]] = obj.id
        print(f"   [OK] Stores: {len(stores)} rows")

        # ── 写入 SKU ──
        sku_map = {}  # code → id
        for sku in skus:
            promo_json = json.dumps(sorted(sku.promo_dates), ensure_ascii=False) if sku.promo_dates else "[]"
            obj = Product(
                sku_code=sku.sku_code, name=sku.name,
                category=sku.category, base_daily_sales=sku.base_daily_sales,
                unit_price=sku.unit_price, min_order_qty=sku.min_order_qty,
                lead_time_days=sku.lead_time_days,
                target_stock_days=sku.target_stock_days,
                promo_dates_json=promo_json,
            )
            session.add(obj)
        session.flush()
        # 重新查询获取 ID
        products = session.query(Product).all()
        sku_map = {p.sku_code: p.id for p in products}
        print(f"   [OK] Products: {len(skus)} rows")

        # ── 批量写入销售记录 ──
        batch_size = 5000
        for i, rec in enumerate(sales_records):
            obj = SalesDaily(
                store_id=store_map[rec.store_code],
                product_id=sku_map[rec.sku_code],
                date=rec.date, quantity=rec.quantity,
                revenue=rec.revenue, is_promo_day=rec.is_promo_day,
            )
            session.add(obj)
            if (i + 1) % batch_size == 0:
                session.flush()
                print(f"   ... sales {i+1}/{len(sales_records)}")
        session.flush()
        print(f"   [OK] Sales: {len(sales_records):,} rows")

        # ── 批量写入库存快照 ──
        for i, rec in enumerate(inv_records):
            obj = InventorySnapshot(
                store_id=store_map[rec.store_code],
                product_id=sku_map[rec.sku_code],
                snapshot_date=rec.date, stock_on_hand=rec.stock_on_hand,
                in_transit=rec.in_transit, safety_stock_line=rec.safety_stock_line,
            )
            session.add(obj)
            if (i + 1) % batch_size == 0:
                session.flush()
                print(f"   ... inventory {i+1}/{len(inv_records)}")
        session.flush()
        print(f"   [OK] Inventory: {len(inv_records):,} rows")

        session.commit()
        print("\n[SUCCESS] Database initialized!")

        # ── 数据质量报告 ──
        _print_quality_report(session, store_map, sku_map)

    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _print_quality_report(session: Session, store_map: dict, sku_map: dict):
    """打印数据质量摘要."""
    print("\n" + "=" * 50)
    print("Data Quality Report")
    print("=" * 50)

    total_sales = session.query(SalesDaily).count()
    total_inv = session.query(InventorySnapshot).count()
    print(f"Total rows: Sales {total_sales:,} | Inventory {total_inv:,}")

    # 缺货天数（销量=0 的记录）
    stockout_count = session.query(SalesDaily).filter(SalesDaily.quantity == 0).count()
    print(f"Stockout days (qty=0): {stockout_count} ({100*stockout_count/total_sales:.1f}%)")

    # 周末 vs 工作日
    # SQLite date function
    from sqlalchemy import func
    weekend_sales = session.query(func.avg(SalesDaily.quantity)).filter(
        func.strftime('%w', SalesDaily.date).in_(['5', '6', '0'])
    ).scalar() or 0
    weekday_sales = session.query(func.avg(SalesDaily.quantity)).filter(
        func.strftime('%w', SalesDaily.date).in_(['1', '2', '3', '4'])
    ).scalar() or 0
    ratio = weekend_sales / weekday_sales if weekday_sales else 0
    print(f"Weekend avg sales: {weekend_sales:.1f} | Weekday: {weekday_sales:.1f} (ratio={ratio:.2f})")

    # 促销日销量
    promo_avg = session.query(func.avg(SalesDaily.quantity)).filter(SalesDaily.is_promo_day == 1).scalar() or 0
    normal_avg = session.query(func.avg(SalesDaily.quantity)).filter(SalesDaily.is_promo_day == 0).scalar() or 0
    print(f"Promo avg sales: {promo_avg:.1f} | Non-promo: {normal_avg:.1f}")

    # 新店 vs 成熟店数据对比
    for code, sid in store_map.items():
        avg = session.query(func.avg(SalesDaily.quantity)).filter(SalesDaily.store_id == sid).scalar() or 0
        print(f"Store {code} avg daily sales: {avg:.1f}")

    print("=" * 50)


# ── CLI 入口 ──

if __name__ == "__main__":
    import sys
    if "--yes" in sys.argv or "-y" in sys.argv:
        seed_database()
    else:
        confirm = input("WARNING: This will delete all data and regenerate. Continue? [y/N] ")
        if confirm.lower() == "y":
            seed_database()
        else:
            print("Cancelled.")
