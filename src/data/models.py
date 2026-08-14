"""SQLAlchemy 模型定义 —— 对齐 PRD v3.0 附录数据库设计."""

import logging
from datetime import datetime
from sqlalchemy import (
    Column, Integer, Float, String, Text, ForeignKey, UniqueConstraint,
    Index, create_engine, event, inspect, text
)
from sqlalchemy.orm import DeclarativeBase, relationship, Session
from src.config import config

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


# ── 引擎 & 会话工厂 ──────────────────────────────────

engine = create_engine(
    config.DATABASE_URL,
    echo=False,
    pool_size=10,
    max_overflow=20,
    pool_recycle=3600,
    pool_pre_ping=True,
    connect_args={"check_same_thread": False} if "sqlite" in config.DATABASE_URL else {},
)


@event.listens_for(engine, "connect")
def _sqlite_pragma(dbapi_connection, connection_record):
    """SQLite: 开启 WAL 模式 + 外键约束."""
    if "sqlite" in config.DATABASE_URL:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


def get_session() -> Session:
    return Session(engine)


# ── 1. 门店 ─────────────────────────────────────────

class Store(Base):
    __tablename__ = "stores"

    id = Column(Integer, primary_key=True)
    store_code = Column(String, unique=True, nullable=False)   # "S001"
    name = Column(String, nullable=False)                      # "旗舰店-朝阳区"
    type = Column(String, nullable=False)                      # flagship/community/new
    area_sqm = Column(Float)                                   # 面积(㎡)
    district = Column(String)                                  # 商圈
    maturity = Column(String, default="mature")                # new/transition/mature
    open_date = Column(String)                                 # "2026-06-01"

    sales = relationship("SalesDaily", back_populates="store")
    inventories = relationship("InventorySnapshot", back_populates="store")
    suggestions = relationship("Suggestion", back_populates="store")

    def __repr__(self):
        return f"<Store {self.store_code} {self.name}>"


# ── 2. 商品/SKU ─────────────────────────────────────

class Product(Base):
    __tablename__ = "products"

    id = Column(Integer, primary_key=True)
    sku_code = Column(String, unique=True, nullable=False)     # "SKU-001"
    name = Column(String, nullable=False)                      # "薯片原味 150g"
    category = Column(String, nullable=False)                  # 零食/饮料/日化
    unit = Column(String, default="件")
    unit_price = Column(Float, default=0.0)                    # 单价
    base_daily_sales = Column(Float)                           # 日均基准销量
    min_order_qty = Column(Integer, default=1)                # 起订量
    lead_time_days = Column(Integer, default=3)               # 前置期(天)
    target_stock_days = Column(Integer, default=7)            # 目标可售天数
    promo_dates_json = Column(Text, default="[]")             # 促销日期 JSON 数组
    created_at = Column(String, default=lambda: datetime.now().isoformat())

    sales = relationship("SalesDaily", back_populates="product")
    inventories = relationship("InventorySnapshot", back_populates="product")
    suggestions = relationship("Suggestion", back_populates="product")

    def __repr__(self):
        return f"<Product {self.sku_code} {self.name}>"


# ── 3. 日销售记录 ────────────────────────────────────

class SalesDaily(Base):
    __tablename__ = "sales_daily"
    __table_args__ = (
        UniqueConstraint("store_id", "product_id", "date", name="uq_sales_store_product_date"),
        Index("idx_sales_store_product", "store_id", "product_id"),
        Index("idx_sales_date", "date"),
    )

    id = Column(Integer, primary_key=True)
    store_id = Column(Integer, ForeignKey("stores.id"), nullable=False)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False)
    date = Column(String, nullable=False)                      # "2026-07-27"
    quantity = Column(Integer, nullable=False, default=0)
    revenue = Column(Float)                                    # 当日销售额
    is_promo_day = Column(Integer, default=0)                  # 0/1 是否促销日

    store = relationship("Store", back_populates="sales")
    product = relationship("Product", back_populates="sales")

    def __repr__(self):
        return f"<Sales {self.store_id}-{self.product_id} {self.date} qty={self.quantity}>"


# ── 4. 库存快照 ──────────────────────────────────────

class InventorySnapshot(Base):
    __tablename__ = "inventory_snapshots"
    __table_args__ = (
        UniqueConstraint("store_id", "product_id", "snapshot_date", name="uq_inv_store_product_date"),
    )

    id = Column(Integer, primary_key=True)
    store_id = Column(Integer, ForeignKey("stores.id"), nullable=False)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False)
    snapshot_date = Column(String, nullable=False)             # "2026-07-27"
    stock_on_hand = Column(Integer, nullable=False)            # 当前库存
    in_transit = Column(Integer, default=0)                    # 在途库存
    safety_stock_line = Column(Integer)                        # 安全库存线

    store = relationship("Store", back_populates="inventories")
    product = relationship("Product", back_populates="inventories")

    def __repr__(self):
        return f"<Inv {self.store_id}-{self.product_id} {self.snapshot_date} stock={self.stock_on_hand}>"


# ── 5. 补货建议 ──────────────────────────────────────

class Suggestion(Base):
    __tablename__ = "suggestions"
    __table_args__ = (
        Index("idx_suggestions_store", "store_id"),
        Index("idx_suggestions_status", "status"),
    )

    id = Column(Integer, primary_key=True)
    store_id = Column(Integer, ForeignKey("stores.id"), nullable=False)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False)
    suggested_qty = Column(Integer, nullable=False)
    final_qty = Column(Integer)                                # 人工修改后
    urgency = Column(String)                                   # low/medium/high/critical
    confidence = Column(String)                                # high/medium/low
    confidence_level = Column(String)                          # 🟢🟡🟣 emoji
    anomaly_type = Column(String)                              # 异常标记，逗号分隔
    decision_factors = Column(Text)                            # JSON: 所有计算因子
    reasoning = Column(Text)                                   # LLM 推理原文
    status = Column(String, default="pending")                 # pending/confirmed/modified/rejected
    generated_at = Column(String, default=lambda: datetime.now().isoformat())
    confirmed_at = Column(String)
    operator_id = Column(String)
    modify_reason = Column(String)

    # ── 闭环反馈（👍👎 写回，供后续 RAG 学习）──
    feedback_rating = Column(String)                           # like/dislike
    feedback_reason = Column(String)                           # 数量偏多/偏少/紧急度不准/其他
    feedback_comment = Column(Text)                            # 自由补充
    feedback_at = Column(String)                               # 反馈时间

    store = relationship("Store", back_populates="suggestions")
    product = relationship("Product", back_populates="suggestions")

    def __repr__(self):
        return f"<Suggestion {self.store_id}-{self.product_id} qty={self.suggested_qty} {self.status}>"


# ── 6. 对话记录 ──────────────────────────────────────

class Conversation(Base):
    __tablename__ = "conversations"
    __table_args__ = (
        Index("idx_conversations_session", "session_id"),
    )

    id = Column(Integer, primary_key=True)
    session_id = Column(String, nullable=False)
    store_id = Column(Integer, ForeignKey("stores.id"))
    product_id = Column(Integer, ForeignKey("products.id"))
    suggestion_id = Column(Integer, ForeignKey("suggestions.id"))
    role = Column(String, nullable=False)                      # user/assistant
    content = Column(Text, nullable=False)
    context_json = Column(Text)                                # 对话时的上下文快照(JSON)
    created_at = Column(String, default=lambda: datetime.now().isoformat())

    def __repr__(self):
        return f"<Conv {self.session_id} {self.role}>"


# ── 一键建表 + 轻量迁移 ────────────────────────────

def _migrate():
    """为已有 SQLite 库补充新增列（create_all 不会改已有表结构）."""
    if "sqlite" not in config.DATABASE_URL:
        return
    try:
        insp = inspect(engine)
        if "suggestions" not in insp.get_table_names():
            return  # 表还没建，交给 create_all
        existing = {c["name"] for c in insp.get_columns("suggestions")}
        additions = {
            "feedback_rating": "VARCHAR(10)",
            "feedback_reason": "VARCHAR(50)",
            "feedback_comment": "TEXT",
            "feedback_at": "VARCHAR(40)",
        }
        to_add = {k: v for k, v in additions.items() if k not in existing}
        if to_add:
            with engine.begin() as conn:
                for col, ddl in to_add.items():
                    conn.execute(text(f"ALTER TABLE suggestions ADD COLUMN {col} {ddl}"))
            logger.info(f"[migrate] suggestions 新增列: {list(to_add)}")
    except Exception as e:
        logger.warning(f"[migrate] 迁移跳过: {e}")


def init_db():
    """创建所有表（幂等：已存在的表不重建）."""
    Base.metadata.create_all(engine)
    _migrate()
