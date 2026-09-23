from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.core.config import MAX_HISTORY_MESSAGES
from app.core.rate_limit import rate_limit
from app.models.chat import ChatRequest, TitleRequest
from app.services.memory import get_chat, clear_chat
from app.memory.memory_service import clear_memory
from app.services.groq_service import (
    ask_groq,
    stream_groq,
    extract_memory,
    generate_chat_title,
    GroqServiceError,
)

router = APIRouter(dependencies=[Depends(rate_limit)])


def _trim_history(conversation):
    if len(conversation) > MAX_HISTORY_MESSAGES:
        del conversation[: len(conversation) - MAX_HISTORY_MESSAGES]


def _prepare_conversation(chat_id: str, message: str, regenerate: bool):
    conversation = get_chat(chat_id)

    if regenerate:
        # The user's turn is already the second-to-last entry from the
        # original request. Just drop the stale assistant reply we're
        # about to replace, instead of appending the same question again
        # (the old code re-appended it, so every regenerate duplicated the
        # question in the model's context and left the stale answer in
        # place too).
        if conversation and conversation[-1]["role"] == "assistant":
            conversation.pop()
    else:
        conversation.append({
            "role": "user",
            "content": message,
        })

    return conversation


@router.post("/chat")
async def chat(request: ChatRequest, background_tasks: BackgroundTasks):

    conversation = _prepare_conversation(
        request.chat_id, request.message, request.regenerate
    )

    # Fire-and-forget: this no longer blocks the reply on a second LLM call.
    background_tasks.add_task(extract_memory, request.chat_id, request.message)

    try:
        answer = await ask_groq(request.chat_id, conversation, web_search=request.web_search)
    except GroqServiceError:
        raise HTTPException(
            status_code=502,
            detail="VextAI couldn't reach the language model. Please try again in a moment.",
        )

    conversation.append({
        "role": "assistant",
        "content": answer,
    })

    _trim_history(conversation)

    return {
        "response": answer
    }


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest, background_tasks: BackgroundTasks):

    conversation = _prepare_conversation(
        request.chat_id, request.message, request.regenerate
    )

    background_tasks.add_task(extract_memory, request.chat_id, request.message)

    async def generate():

        full_answer = ""

        try:
            async for chunk in stream_groq(request.chat_id, conversation, web_search=request.web_search):
                full_answer += chunk
                yield chunk

        except GroqServiceError:
            fallback = "⚠ VextAI couldn't reach the language model. Please try again."
            full_answer = full_answer or fallback
            yield fallback

        conversation.append({
            "role": "assistant",
            "content": full_answer,
        })

        _trim_history(conversation)

    return StreamingResponse(
        generate(),
        media_type="text/plain",
        headers={
            # Without these, some reverse proxies / PaaS layers (nginx's
            # default proxy_buffering, for example) buffer the entire
            # streamed reply before sending anything to the browser. The
            # backend is streaming token-by-token the whole time -- it just
            # never reaches the client that way, so the reply looks like one
            # long pause followed by the whole answer appearing at once.
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache",
        },
    )


@router.post("/chat/title")
async def chat_title(request: TitleRequest):

    title = await generate_chat_title(request.message)

    return {
        "title": title
    }


@router.delete("/chat/{chat_id}")
async def delete_chat(chat_id: str):
    """Clears server-side history + remembered facts for a chat. Called
    when a chat is deleted in the sidebar so nothing lingers forever."""

    clear_chat(chat_id)
    await clear_memory(chat_id)

    return {
        "deleted": True
    }
