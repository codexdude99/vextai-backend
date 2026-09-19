from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)
    chat_id: str = Field(..., min_length=1, max_length=200)

    # True when this call is re-generating the last assistant reply.
    # Tells the backend to reuse the existing user turn instead of
    # appending it a second time (see routes/chat.py).
    regenerate: bool = False


class TitleRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
