#!/bin/bash
# ─────────────────────────────────────────────
#  AI 智能补货 Agent — 一键启动脚本
#  macOS / Linux
# ─────────────────────────────────────────────
set -e

echo "========================================"
echo "  AI Replenishment Agent — Quick Start"
echo "========================================"
echo ""

# Check .env
if [ ! -f .env ]; then
    echo "[1/4] Creating .env from template..."
    cp .env.example .env
    echo "      WARNING: Edit .env and add your DEEPSEEK_API_KEY"
    echo "      Without it, the system runs in rule-only mode."
else
    echo "[1/4] .env exists"
fi

# Source .env
export $(grep -v '^#' .env | xargs)

# Install if needed
if ! python -c "import fastapi" 2>/dev/null; then
    echo "[*] Installing dependencies..."
    pip install -e ".[dev]" -q
fi

# Generate data
echo "[2/4] Generating mock data..."
python -m src.data.seed -y

# Generate suggestions
echo "[3/4] Generating replenishment suggestions..."
for store in S001 S002 S003; do
    echo "      Store $store..."
    python -m src.agent.decision --store "$store" --no-llm 2>&1 | grep "Mode:" || true
done

# Start server
echo "[4/4] Starting web server..."
echo ""
echo "========================================"
echo "  Open: http://localhost:8000"
echo "  Press Ctrl+C to stop"
echo "========================================"
echo ""

python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
