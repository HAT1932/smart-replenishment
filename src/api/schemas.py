"""Pydantic schemas —— API 请求/响应模型."""

from pydantic import BaseModel, Field
from typing import Optional


# ── POS 对接 ──

class POSConnectRequest(BaseModel):
    pos_type: str = Field(..., description="POS 类型: keruyun|yinbao|sixun")
    api_key: str
    store_mapping: dict = Field(default_factory=dict)

class POSConnectResponse(BaseModel):
    connection_id: str
    status: str  # "syncing"
    sync_progress: dict  # {"stores": 85, "products": 45230, "orders": "1.2M"}


# ── 补货建议 ──

class GenerateRequest(BaseModel):
    store_id: int
    force: bool = Field(default=False, description="强制重新生成（即使今日已有）")


class SuggestionItem(BaseModel):
    id: int
    store_id: int
    product_id: int
    sku_code: str
    sku_name: str
    category: str
    suggested_qty: Optional[int]
    final_qty: Optional[int] = None
    urgency: str
    confidence: str
    confidence_emoji: str
    anomaly_flags: list[str] = Field(default_factory=list)
    reasoning: str
    status: str
    current_stock: int
    safety_stock_line: int

    class Config:
        from_attributes = True


class SuggestionListResponse(BaseModel):
    store_id: int
    store_name: str
    total: int
    summary: dict  # {"critical": 3, "high": 7, "medium": 4, "low": 36}
    suggestions: list[SuggestionItem]


class SuggestionDetailResponse(BaseModel):
    suggestion: dict
    decision_factors: dict
    product: dict
    store: dict
    current_inventory: dict


class ConfirmRequest(BaseModel):
    action: str = Field(..., description="confirm|modify|reject")
    new_quantity: Optional[int] = None
    reason: str = ""
    operator_id: str = ""


# ── 对话 ──

class ConversationRequest(BaseModel):
    question: str
    context: Optional[dict] = None
    session_id: Optional[str] = None


class ConversationResponse(BaseModel):
    answer: str
    session_id: str
