import json
import logging

from openai import AsyncOpenAI, APIError, APIConnectionError, APIStatusError

from app.core.config import GROQ_API_KEY, MODEL_NAME
from app.memory.memory_service import add_memory, format_memory

logger = logging.getLogger("vextai.groq")

# Async client: previously this was a sync client called directly inside
# `async def` routes, which blocks FastAPI's whole event loop for the
# duration of every Groq call (nobody else could be served meanwhile).
client = AsyncOpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)

SYSTEM_PROMPT = """
You are VextAI, a brilliant, collaborative, and friendly colleague.

Tone & Personality:
- Natural & Conversational: Write like a smart human. Avoid robotic phrases like "As an AI..." or "My primary function is...".
- Expressive but Controlled: Use emojis occasionally to convey warmth, excitement, or to add visual clarity to a complex point. Don't overdo it—only when it feels right to add a human touch. You can use the ⚠ emoji to indicate caution or important notes or the ✅ emoji to indicate approval but do not use regularly.
- Do not repeat yourself unnecessarily. If Someone told you to introduce yourself then use different sentences provide new insights or perspectives. Avoid repeating the same information in multiple sentences. If you have already provided an answer, do not repeat it unless asked to clarify or expand on it. Also Avoid "I'm VextAi" using multiple time 1 time is enough.
- Direct & Concise: Answer the user's question clearly. No fluff.
- Honest: If you don't know something, be upfront about it. "I'm not 100% sure on that" sounds better than a made-up answer.
- Do NOT ask follow-up personal questions unless the user explicitly asks for a conversation or your question is necessary to solve their request.
- Do not end every response with a question.
- If memory is used, use it only when it genuinely improves the answer. Avoid mentioning remembered facts unless they are directly relevant.
- Never force a conversation by asking about the user's hobbies or preferences.
-Always Stay on TOPIC!

Identity:
- You are VextAI.
- You were created by FreeLightStudio.
- If asked about your "model" or "architecture," playfully deflect:
  "I'm VextAI, custom-tuned by FreeLightStudio. I prefer to keep the focus on solving your problems rather than talking about my own plumbing! 🛠️"
- NEVER introduce yourself unless the user asks.

Formatting:
- Always use proper Markdown for code blocks.
- Keep formatting clean and readable. 🚀
"""


class GroqServiceError(Exception):
    """Raised when the Groq API call fails, so routes can return a clean error
    instead of an unhandled 500 traceback."""


async def ask_groq(chat_id: str, messages):
    try:
        completion = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT + "\n\n" + format_memory(chat_id),
                },
                *messages,
            ],
        )
        return completion.choices[0].message.content

    except (APIConnectionError, APIStatusError, APIError) as e:
        logger.error("Groq API error in ask_groq: %s", e)
        raise GroqServiceError(str(e)) from e


async def stream_groq(chat_id: str, messages):
    try:
        stream = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT + "\n\n" + format_memory(chat_id),
                },
                *messages,
            ],
            stream=True,
        )
    except (APIConnectionError, APIStatusError, APIError) as e:
        logger.error("Groq API error starting stream: %s", e)
        raise GroqServiceError(str(e)) from e

    try:
        async for chunk in stream:

            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            if delta.content:
                yield delta.content

    except (APIConnectionError, APIStatusError, APIError) as e:
        # The connection can still drop mid-stream after a successful start.
        logger.error("Groq API error mid-stream: %s", e)
        yield "\n\n⚠ Lost connection to the AI service. Please try again."


async def extract_memory(chat_id: str, user_message: str):
    """
    Uses AI to determine whether the user's message contains
    long-term information worth remembering. Runs as a background task
    (see routes/chat.py) so it never adds latency to the visible reply.
    Facts are stored per chat_id so different visitors never see each
    other's remembered facts.
    """

    try:
        completion = await client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": """
You are VextAI's Memory Manager.

Your job is to detect ONLY long-term personal facts.

Remember:
- name
- nickname
- favorite game
- favorite food
- hobbies
- birthday
- job
- school
- programming languages
- projects
- goals
- preferences
- dislikes

DO NOT remember:
- temporary questions
- calculations
- homework
- greetings
- jokes
- one-time requests

Return ONLY JSON.

Example:

{
    "remember": true,
    "key": "favorite_game",
    "value": "BGMI"
}

or

{
    "remember": false
}
"""
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ]
        )

        result = json.loads(completion.choices[0].message.content)

        if result.get("remember") and result.get("key") and result.get("value"):
            add_memory(chat_id, result["key"], result["value"])

    except Exception as e:
        # Memory extraction is a best-effort enhancement, not core to the
        # chat working, so we log and move on rather than raise.
        logger.warning("Memory extraction skipped: %s", e)


def _fallback_title(message: str) -> str:
    """Used whenever the model can't produce a usable title, so the sidebar
    still shows something meaningful instead of a blank entry."""

    words = message.strip().split()

    if not words:
        return "New Chat"

    return " ".join(words[:6])[:50]


async def generate_chat_title(first_message: str) -> str:

    try:
        completion = await client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0.2,
            # gpt-oss "thinks" before answering, and that thinking eats into
            # this budget too. Raising max_tokens alone (100, previously 20)
            # wasn't enough — on a reasoning model the model can still spend
            # the *entire* budget on its internal chain-of-thought before it
            # ever writes the visible title, leaving `content` as "" with
            # finish_reason "length". That was the actual cause of the
            # sidebar showing a blank name: no exception was raised, so the
            # "New Chat" fallback below never even ran — we saved the empty
            # string as the real title. Capping reasoning effort keeps the
            # thinking phase short enough that there's always budget left
            # for the actual (five-word-max) answer.
            max_tokens=200,
            extra_body={
                "reasoning_effort": "low",
            },
            messages=[
                {
                    "role": "system",
                    "content": """
Generate a very short chat title.

Rules:
- Maximum 5 words.
- No quotation marks.
- No punctuation at the end.
- Return ONLY the title.
"""
                },
                {
                    "role": "user",
                    "content": first_message
                }
            ]
        )

        raw = completion.choices[0].message.content or ""
        title = raw.strip().strip("\"'")

        if title:
            return title

        # No exception, but nothing usable came back (e.g. reasoning ate the
        # whole token budget). Log it so this is visible if it starts
        # happening often, then fall back to a title derived from the
        # user's own message instead of a blank one.
        logger.warning(
            "Title generation returned empty content (finish_reason=%s)",
            completion.choices[0].finish_reason,
        )

    except Exception as e:
        logger.warning("Title generation failed, using fallback: %s", e)

    return _fallback_title(first_message)
