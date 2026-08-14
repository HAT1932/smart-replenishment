"""FastAPI 应用入口."""

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from jinja2 import Environment, FileSystemLoader

from src.api.routes import suggestions, conversation
from src.data.models import get_session, Store, Suggestion, Product, init_db

# 启动时确保表结构就绪（含轻量迁移：老库自动补列）
init_db()

# ── App ──
app = FastAPI(title="AI Replenishment Agent", version="0.1.0")

app.include_router(suggestions.router)
app.include_router(conversation.router)

# ── Static & Templates ──
BASE = Path(__file__).parent.parent.parent  # smart-replenishment/
STATIC = BASE / "src" / "ui" / "static"
TEMPLATES_DIR = BASE / "src" / "ui" / "templates"

if STATIC.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

jinja_env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR))) if TEMPLATES_DIR.exists() else None


# ═══════════════════════════════════════════════
# 页面路由
# ═══════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    """首页：门店列表."""
    if not jinja_env:
        return HTMLResponse("<h1>Templates not found</h1>")

    session = get_session()
    stores = session.query(Store).all()
    store_data = []
    for s in stores:
        sug_count = session.query(Suggestion).filter(
            Suggestion.store_id == s.id, Suggestion.status == "pending"
        ).count()
        critical = session.query(Suggestion).filter(
            Suggestion.store_id == s.id, Suggestion.urgency == "critical"
        ).count()
        store_data.append({
            "id": s.id, "code": s.store_code, "name": s.name,
            "type": s.type, "maturity": s.maturity,
            "pending": sug_count, "critical": critical,
        })
    session.close()
    template = jinja_env.get_template("index.html")
    return HTMLResponse(template.render(stores=store_data))


@app.get("/stores/{store_id}", response_class=HTMLResponse)
def review_workbench(request: Request, store_id: int):
    """审核工作台."""
    if not jinja_env:
        return HTMLResponse("<h1>Templates not found</h1>")
    template = jinja_env.get_template("workbench.html")
    return HTMLResponse(template.render(store_id=store_id))


@app.get("/stores/{store_id}/sku/{sku_id}", response_class=HTMLResponse)
def detail_chat(request: Request, store_id: int, sku_id: int):
    """建议详情 + 对话面板."""
    if not jinja_env:
        return HTMLResponse("<h1>Templates not found</h1>")

    session = get_session()
    store = session.query(Store).filter(Store.id == store_id).first()
    product = session.query(Product).filter(Product.id == sku_id).first()
    sug = (
        session.query(Suggestion)
        .filter(Suggestion.store_id == store_id, Suggestion.product_id == sku_id)
        .order_by(Suggestion.generated_at.desc())
        .first()
    )
    session.close()

    template = jinja_env.get_template("detail.html")
    return HTMLResponse(template.render(
        store_id=store_id,
        sku_id=sku_id,
        suggestion_id=sug.id if sug else None,
        store_name=store.name if store else f"Store {store_id}",
        sku_name=product.name if product else f"SKU {sku_id}",
    ))


@app.get("/mobile/confirm/{store_id}", response_class=HTMLResponse)
def mobile_confirm(request: Request, store_id: int):
    """移动端门店确认."""
    if not jinja_env:
        return HTMLResponse("<h1>Templates not found</h1>")

    session = get_session()
    store = session.query(Store).filter(Store.id == store_id).first()
    session.close()
    template = jinja_env.get_template("mobile.html")
    return HTMLResponse(template.render(
        store_id=store_id,
        store_name=store.name if store else f"Store {store_id}",
    ))
