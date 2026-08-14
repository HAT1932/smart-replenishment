@echo off
REM ─────────────────────────────────────────────
REM  AI Replenishment Agent — One-Click Start
REM  Windows
REM ─────────────────────────────────────────────

echo ========================================
echo   AI Replenishment Agent — Quick Start
echo ========================================
echo.

REM Check .env
if not exist .env (
    echo [1/4] Creating .env from template...
    copy .env.example .env >nul
    echo       WARNING: Edit .env and add your DEEPSEEK_API_KEY
    echo       Without it, the system runs in rule-only mode.
) else (
    echo [1/4] .env exists
)

REM Generate data
echo [2/4] Generating mock data...
python -m src.data.seed -y

REM Generate suggestions
echo [3/4] Generating replenishment suggestions...
python -m src.agent.decision --store S001 --no-llm
python -m src.agent.decision --store S002 --no-llm
python -m src.agent.decision --store S003 --no-llm

REM Start server
echo [4/4] Starting web server...
echo.
echo ========================================
echo   Open: http://localhost:8000
echo   Press Ctrl+C to stop
echo ========================================
echo.

python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
