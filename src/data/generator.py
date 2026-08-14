"""Mock POS 数据生成器.

生成 3 家门店 × 50 SKU × 90 天的模拟销售和库存数据。
不生成随机数——模拟真实零售业务模式：
- 周末效应（周五/六 +30%）
- 季节性趋势（饮料夏高冬低）
- 促销峰值（1.5-2.5x）
- 缺货断档（库存归零后销量=0）
- 补货后恢复（每周二到货）
- 新店衰减（前 7 天销量仅 40-60%）
"""

import random
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field


# ══════════════════════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════════════════════

START_DATE = datetime(2026, 4, 28)   # 模拟起始日
TOTAL_DAYS = 90                      # 总天数
RANDOM_SEED = 42                     # 可复现

CATEGORIES = ["零食", "饮料", "日化"]

STORE_CONFIGS = [
    {"code": "S001", "name": "旗舰店-朝阳区", "type": "flagship", "area": 500, "district": "核心商圈", "maturity": "mature", "open_date": "2025-01-15"},
    {"code": "S002", "name": "社区店-回龙观", "type": "community", "area": 200, "district": "居民区", "maturity": "transition", "open_date": "2026-06-01"},
    {"code": "S003", "name": "新店-望京SOHO", "type": "new", "area": 300, "district": "写字楼", "maturity": "new", "open_date": "2026-07-20"},
]


# ══════════════════════════════════════════════════════════════
# 数据类
# ══════════════════════════════════════════════════════════════

@dataclass
class SkuConfig:
    sku_code: str
    name: str
    category: str
    base_daily_sales: int          # 日均基准销量
    unit_price: float              # 单价
    min_order_qty: int             # 起订量
    lead_time_days: int            # 前置期
    target_stock_days: int         # 目标可售天数
    promo_dates: set[str] = field(default_factory=set)  # 促销日集合 "2026-07-01"


@dataclass
class SalesRecord:
    store_code: str
    sku_code: str
    date: str
    quantity: int
    revenue: float
    is_promo_day: int


@dataclass
class InventoryRecord:
    store_code: str
    sku_code: str
    date: str
    stock_on_hand: int
    in_transit: int
    safety_stock_line: int


# ══════════════════════════════════════════════════════════════
# SKU 清单
# ══════════════════════════════════════════════════════════════

def _build_sku_list() -> list[SkuConfig]:
    """构建 50 个 SKU 的定义."""
    skus = []
    idx = 0

    # ── 零食 20 个 ──
    snacks = [
        ("薯片原味 150g", 55, 8.9), ("薯片烧烤味 150g", 48, 8.9),
        ("薯片番茄味 150g", 42, 8.9), ("辣条大包装 200g", 60, 5.5),
        ("辣条小包装 80g", 75, 3.0), ("坚果混合装 500g", 30, 39.9),
        ("瓜子原味 300g", 35, 12.9), ("花生椒盐 250g", 25, 9.9),
        ("饼干苏打 200g", 40, 6.9), ("饼干夹心 150g", 38, 7.9),
        ("蛋糕瑞士卷 6枚", 28, 15.9), ("面包全麦切片 400g", 45, 12.9),
        ("巧克力排块 100g", 22, 18.9), ("糖果混合装 300g", 20, 22.9),
        ("果冻杯装 6杯", 33, 11.9), ("肉脯猪肉 200g", 26, 29.9),
        ("鱿鱼丝 150g", 18, 25.9), ("海苔夹心 100g", 30, 13.9),
        ("膨化虾条 120g", 50, 5.9), ("米饼蛋黄 200g", 36, 9.9),
    ]
    for name, base_sales, price in snacks:
        idx += 1
        skus.append(SkuConfig(
            sku_code=f"SKU-{idx:03d}", name=name, category="零食",
            base_daily_sales=base_sales, unit_price=price,
            min_order_qty=10, lead_time_days=3, target_stock_days=7,
            promo_dates=_random_promo_dates(3),
        ))

    # ── 饮料 15 个 ──
    drinks = [
        ("可乐 500ml", 80, 3.5), ("雪碧 500ml", 65, 3.5),
        ("芬达橙味 500ml", 50, 3.5), ("矿泉水 550ml", 100, 2.0),
        ("纯净水 1.5L", 60, 3.5), ("绿茶无糖 500ml", 55, 5.0),
        ("红茶冰红茶 500ml", 70, 4.5), ("果汁橙汁 1L", 30, 8.9),
        ("果汁苹果汁 1L", 28, 8.9), ("运动饮料 600ml", 45, 6.0),
        ("咖啡拿铁罐装 270ml", 25, 12.9), ("豆奶原味 250ml", 22, 4.5),
        ("酸奶原味 200g", 40, 5.9), ("酸奶草莓 200g", 35, 5.9),
        ("乳酸菌饮品 100ml×5", 32, 10.9),
    ]
    for name, base_sales, price in drinks:
        idx += 1
        skus.append(SkuConfig(
            sku_code=f"SKU-{idx:03d}", name=name, category="饮料",
            base_daily_sales=base_sales, unit_price=price,
            min_order_qty=24, lead_time_days=2, target_stock_days=5,
            promo_dates=_random_promo_dates(3),
        ))

    # ── 日化 15 个 ──
    daily = [
        ("洗发水去屑 500ml", 18, 39.9), ("洗发水柔顺 500ml", 15, 39.9),
        ("沐浴露清爽 1L", 20, 35.9), ("洗衣液薰衣草 2kg", 22, 29.9),
        ("洗衣液除菌 2kg", 19, 32.9), ("牙膏美白 180g", 30, 15.9),
        ("牙膏抗敏 120g", 22, 19.9), ("牙刷软毛 3支装", 28, 12.9),
        ("纸巾抽纸 6包装", 50, 24.9), ("纸巾卷纸 12卷", 40, 29.9),
        ("湿巾 80片×3", 25, 19.9), ("洗手液泡沫 250ml", 18, 13.9),
        ("洗洁精柠檬 1L", 16, 9.9), ("消毒液 500ml", 10, 22.9),
        ("衣物柔顺剂 1L", 13, 19.9),
    ]
    for name, base_sales, price in daily:
        idx += 1
        skus.append(SkuConfig(
            sku_code=f"SKU-{idx:03d}", name=name, category="日化",
            base_daily_sales=base_sales, unit_price=price,
            min_order_qty=6, lead_time_days=5, target_stock_days=14,
            promo_dates=_random_promo_dates(2),
        ))

    return skus


def _random_promo_dates(count: int) -> set[str]:
    """随机生成促销日期."""
    dates = set()
    for _ in range(count):
        offset = random.randint(10, TOTAL_DAYS - 5)
        d = START_DATE + timedelta(days=offset)
        dates.add(d.strftime("%Y-%m-%d"))
    return dates


# ══════════════════════════════════════════════════════════════
# 模拟引擎
# ══════════════════════════════════════════════════════════════

class SalesSimulator:
    """模拟零售销售：非随机，有业务规律."""

    def __init__(self, skus: list[SkuConfig], stores: list[dict], seed: int = RANDOM_SEED):
        self.skus = {s.sku_code: s for s in skus}
        self.stores = {s["code"]: s for s in stores}
        self.rng = np.random.default_rng(seed)

        # 品类季节性因子（按月）
        self._seasonal = {
            "饮料": {4: 0.9, 5: 1.1, 6: 1.3, 7: 1.4, 8: 1.4, 9: 1.1, 10: 0.9, 11: 0.7, 12: 0.6, 1: 0.6, 2: 0.7, 3: 0.8},
            "零食": {4: 1.0, 5: 1.0, 6: 1.2, 7: 1.3, 8: 1.3, 9: 1.1, 10: 1.0, 11: 1.0, 12: 1.2, 1: 1.2, 2: 1.1, 3: 1.0},
            "日化": {4: 1.0, 5: 1.0, 6: 1.1, 7: 1.1, 8: 1.0, 9: 1.0, 10: 1.0, 11: 0.9, 12: 1.0, 1: 1.0, 2: 0.9, 3: 1.0},
        }

    def generate(self) -> tuple[list[SalesRecord], list[InventoryRecord]]:
        sales_records = []
        inv_records = []

        # 追踪每个 store-sku 的库存状态
        stock_state: dict[str, dict[str, int]] = {}
        transit_state: dict[str, dict[str, int]] = {}  # 在途到货日映射
        in_transit_qty: dict[str, dict[str, int]] = {}

        # 初始化库存
        for store_code, st in self.stores.items():
            stock_state[store_code] = {}
            transit_state[store_code] = {}
            in_transit_qty[store_code] = {}
            for sku_code, sku in self.skus.items():
                # 初始库存 = 目标可售天数 × 基准日销 × (新店 0.5)
                mult = 0.5 if st["maturity"] == "new" else 1.2
                stock_state[store_code][sku_code] = int(sku.target_stock_days * sku.base_daily_sales * mult)
                in_transit_qty[store_code][sku_code] = 0

        # 逐日模拟
        for day_offset in range(TOTAL_DAYS):
            current_date = START_DATE + timedelta(days=day_offset)
            date_str = current_date.strftime("%Y-%m-%d")
            day_of_week = current_date.weekday()  # 0=Mon, 6=Sun
            is_tuesday = day_of_week == 1         # 周二到货日

            for store_code, st in self.stores.items():
                for sku_code, sku in self.skus.items():

                    # ── 到货处理（周二）──
                    if is_tuesday and in_transit_qty[store_code][sku_code] > 0:
                        arriving = in_transit_qty[store_code][sku_code]
                        stock_state[store_code][sku_code] += arriving
                        in_transit_qty[store_code][sku_code] = 0

                    # ── 计算当日销量 ──
                    current_stock = stock_state[store_code][sku_code]
                    if current_stock <= 0:
                        qty = 0  # 缺货：卖不了
                    else:
                        qty = self._simulate_day_sales(sku, st, day_offset, date_str)
                        qty = min(qty, current_stock)  # 不能卖出超过库存
                        stock_state[store_code][sku_code] -= qty

                    # ── 销售额 ──
                    revenue = round(qty * sku.unit_price, 2)
                    is_promo = 1 if date_str in sku.promo_dates else 0

                    sales_records.append(SalesRecord(
                        store_code=store_code, sku_code=sku_code,
                        date=date_str, quantity=qty, revenue=revenue,
                        is_promo_day=is_promo,
                    ))

                    # ── 库存快照 ──
                    stock_now = stock_state[store_code][sku_code]
                    in_trans = in_transit_qty[store_code][sku_code]
                    avg_sales = sku.base_daily_sales
                    safety = int(avg_sales * (sku.lead_time_days + 3))  # 安全线 = 日均 × (前置期+3天缓冲)

                    inv_records.append(InventoryRecord(
                        store_code=store_code, sku_code=sku_code,
                        date=date_str, stock_on_hand=stock_now,
                        in_transit=in_trans, safety_stock_line=safety,
                    ))

                    # ── 自动补货(简化: 库存低于安全线时触发) ──
                    if stock_now <= safety:
                        reorder_qty = int(sku.target_stock_days * sku.base_daily_sales)
                        reorder_qty = max(reorder_qty, sku.min_order_qty)
                        # 整箱取整
                        reorder_qty = ((reorder_qty + sku.min_order_qty - 1) // sku.min_order_qty) * sku.min_order_qty
                        in_transit_qty[store_code][sku_code] += reorder_qty

        return sales_records, inv_records

    def _simulate_day_sales(self, sku: SkuConfig, store: dict, day_offset: int, date_str: str) -> int:
        """模拟单日销量."""
        base = sku.base_daily_sales

        # 1. 周末效应
        current_date = START_DATE + timedelta(days=day_offset)
        dow = current_date.weekday()  # 0=Mon
        weekend_mult = {4: 1.15, 5: 1.30, 6: 1.10}.get(dow, 1.0)

        # 2. 季节因子
        month = current_date.month
        seasonal = self._seasonal.get(sku.category, {}).get(month, 1.0)

        # 3. 促销因子
        promo_mult = 2.0 if date_str in sku.promo_dates else 1.0

        # 4. 新店衰减
        store_mult = self._store_maturity_mult(store, day_offset)

        # 5. 预期销量
        expected = base * weekend_mult * seasonal * promo_mult * store_mult

        # 6. 泊松噪声（零售日销量近似泊松分布）
        qty = self.rng.poisson(max(0.5, expected))
        return max(0, int(qty))

    def _store_maturity_mult(self, store: dict, day_offset: int) -> float:
        """新店衰减：前7天 40-60%，7-30天逐渐恢复."""
        if store["maturity"] != "new":
            return 1.0
        open_date = datetime.strptime(store["open_date"], "%Y-%m-%d")
        days_since_open = (START_DATE + timedelta(days=day_offset) - open_date).days
        if days_since_open < 0:
            return 0.0   # 还没开业
        if days_since_open <= 7:
            return 0.4 + 0.03 * days_since_open   # 40%→60%
        if days_since_open <= 30:
            return 0.6 + 0.017 * (days_since_open - 7)  # 60%→100%
        return 1.0


# ══════════════════════════════════════════════════════════════
# 公共入口
# ══════════════════════════════════════════════════════════════

def generate_all() -> tuple[list[SkuConfig], list[dict], list[SalesRecord], list[InventoryRecord]]:
    """生成全部 Mock 数据."""
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    skus = _build_sku_list()
    stores = STORE_CONFIGS
    sim = SalesSimulator(skus, stores)
    sales, inv = sim.generate()

    print(f"[OK] Mock data generated: {len(skus)} SKU x {len(stores)} stores x {TOTAL_DAYS} days")
    print(f"     Sales records: {len(sales):,} rows")
    print(f"     Inventory snapshots: {len(inv):,} rows")
    return skus, stores, sales, inv
