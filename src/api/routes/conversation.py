"""对话 API 路由."""

import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from src.agent.conversation.agent import ConversationAgent
from src.api.schemas import ConversationRequest

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/conversation", tags=["conversation"])

agent = ConversationAgent()


@router.post("/ask")
def ask(req: ConversationRequest):
    """对话追问（同步）."""
    try:
        result = agent.ask(
            question=req.question,
            context=req.context,
            session_id=req.session_id,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ask/stream")
async def ask_stream(req: ConversationRequest):
    """对话追问（SSE 流式）."""
    def generate():
        try:
            for token in agent.ask_stream(
                question=req.question,
                context=req.context,
                session_id=req.session_id,
            ):
                yield f"data: {token}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: [ERROR] {str(e)[:200]}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
