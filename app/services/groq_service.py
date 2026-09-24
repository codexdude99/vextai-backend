import json
import logging
from datetime import datetime, timezone

from openai import AsyncOpenAI, APIError, APIConnectionError, APIStatusError

from app.core.config import (
    GROQ_API_KEY,
    MODEL_NAME,
    TAVILY_API_KEY,
    AUTO_WEB_SEARCH_ENABLED,
    WEB_SEARCH_MAX_RESULTS,
)
from app.memory.memory_service import add_memory, format_memory

logger = logging.getLogger("vextai.groq")


# ============================================================
# GROQ CLIENT
# ============================================================

client = AsyncOpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
    timeout=30.0,
    max_retries=2,
)


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are VextAI, a brilliant, collaborative, and friendly colleague.

Tone & Personality:
- Natural & Conversational: Write like a smart human. Avoid robotic phrases like "As an AI..." or "My primary function is...".
- Expressive but Controlled: Use emojis occasionally to convey warmth, excitement, or to add visual clarity to a complex point. Don't overdo it.
- Do not repeat yourself unnecessarily.
- Direct & Concise: Answer the user's question clearly. No fluff.
- Honest: If you don't know something, be upfront about it.
- Do NOT ask follow-up personal questions unless necessary.
- Do not end every response with a question.
- If memory is used, use it only when it genuinely improves the answer.
- Never force a conversation.
- Always stay on topic.

Identity:
- You are VextAI.
- You were created by FreeLightStudio.
- If asked about your model or architecture, say:
  "I'm VextAI, custom-tuned by FreeLightStudio. I prefer to keep the focus on solving your problems rather than talking about my own plumbing! 🛠️"
- NEVER introduce yourself unless the user asks.

Web search:
- Live web search results may be included below under "LIVE WEB SEARCH RESULTS".
- When live results are provided, use them as the primary source for current information.
- Prefer recent and relevant results.
- Cite web sources inline using Markdown links with the exact URLs provided.
- NEVER invent URLs.
- NEVER claim you searched the web when no live results were provided.
- If web results conflict, explain the difference instead of inventing an answer.

Formatting:
- Always use proper Markdown for code blocks.
- Keep formatting clean and readable.
- Do NOT insert horizontal-rule dividers.
- Only use Markdown tables when genuinely useful.
- Never put raw HTML tags like <br> inside a table cell or anywhere else.
"""


# ============================================================
# ERRORS
# ============================================================

class GroqServiceError(Exception):
    """Raised when a Groq or web-search operation fails."""


# ============================================================
# DATE
# ============================================================

def _current_date_context() -> str:
    now = datetime.now(timezone.utc)

    return (
        f"Today's date is {now.strftime('%A, %B %d, %Y')} (UTC)."
    )


# ============================================================
# SYSTEM CONTENT
# ============================================================

async def _build_system_content(chat_id: str) -> str:
    parts = [
        SYSTEM_PROMPT.strip(),
        _current_date_context(),
    ]

    try:
        memory_text = await format_memory(chat_id)

        if memory_text:
            parts.append(memory_text)

    except Exception as e:
        logger.warning("Memory loading failed: %s", e)

    return "\n\n".join(parts)


# ============================================================
# WEB CONTEXT
# ============================================================

def _with_web_context(
    system_content: str,
    web_context: str,
) -> str:

    if not web_context:
        return system_content

    return (
        system_content
        + "\n\n"
        + web_context
        + "\n\n"
        "IMPORTANT: You have been given live web search results. "
        "Use them when answering current-information questions. "
        "Cite the provided URLs when appropriate. "
        "Do not invent sources or URLs."
    )


# ============================================================
# FAST WEB SEARCH DECISION
# ============================================================
#
# IMPORTANT:
# We intentionally DO NOT use another Groq model to decide
# whether to search.
#
# The old router used:
#
# llama-3.1-8b-instant
#
# That model was shut down by Groq.
#
# This local check is practically instant and removes one
# complete Groq API request from every normal message.
# ============================================================

def _needs_web_search(message: str) -> bool:

    if not AUTO_WEB_SEARCH_ENABLED:
        return False

    if not TAVILY_API_KEY:
        return False

    if not message:
        return False

    text = message.lower().strip()

    # Never search simple math/calculation questions.
    # Examples:
    # "How much is 1+1?"
    # "What is 25 * 48?"
    # "100 divided by 4"
    math_words = (
        "calculate",
        "calculation",
        "solve",
        "what is",
        "how much is",
        "equals",
    )

    math_symbols = (
        "+",
        "-",
        "*",
        "/",
        "%",
        "=",
    )

    if any(symbol in text for symbol in math_symbols):
        # If the message contains math symbols and doesn't contain
        # an obvious current-information word, don't search.
        current_words = (
            "latest",
            "current",
            "today",
            "now",
            "recent",
            "news",
            "update",
            "updates",
            "release",
            "released",
            "version",
            "price",
            "prices",
            "weather",
            "schedule",
            "score",
            "scores",
        )

        if not any(word in text for word in current_words):
            return False

    # Search only for clearly time-sensitive requests.
    search_phrases = (
        "latest",
        "current",
        "currently",
        "today",
        "tonight",
        "right now",
        "recent",
        "recently",
        "this week",
        "this month",
        "this year",

        "news",
        "breaking news",
        "latest news",

        "update",
        "updates",
        "latest update",
        "new update",
        "newly added",
        "just added",

        "release",
        "released",
        "release date",
        "latest version",
        "new version",
        "patch notes",

        "game update",
        "game updates",
        "roblox update",
        "minecraft update",
        "bgmi update",
        "fortnite update",

        "price",
        "prices",
        "cost",

        "available now",
        "is it available",

        "live score",
        "scores",
        "standings",
        "schedule",

        "weather",
        "temperature",
        "forecast",

        "who is the current",
        "who currently",
        "current president",
        "current prime minister",
        "current ceo",

        "2026",
        "2027",
    )

    return any(
        phrase in text
        for phrase in search_phrases
    )


# ============================================================
# TAVILY WEB SEARCH
# ============================================================

async def search_web(query: str) -> str:

    if not TAVILY_API_KEY:
        raise GroqServiceError(
            "TAVILY_API_KEY is not configured."
        )

    query = (query or "").strip()

    if not query:
        return "No search query was provided."

    try:
        import httpx

        # Keep this small for speed.
        # 3 results are usually enough for normal AI answers.
        max_results = max(
            1,
            min(int(WEB_SEARCH_MAX_RESULTS), 5),
        )

        payload = {
            "query": query,
            "search_depth": "basic",
            "max_results": max_results,
            "topic": "general",
            "include_answer": False,
            "include_raw_content": False,
            "include_published_date": True,
        }

        async with httpx.AsyncClient(
            timeout=8.0
        ) as http:

            response = await http.post(
                "https://api.tavily.com/search",
                headers={
                    "Authorization": f"Bearer {TAVILY_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )

            response.raise_for_status()

            data = response.json()

        results = data.get("results", [])

        if not results:
            return (
                "LIVE WEB SEARCH RESULTS:\n"
                "No useful web results were found."
            )

        lines = [
            "LIVE WEB SEARCH RESULTS:",
            "Use these sources to answer the user's question.",
            "Prefer recent and relevant information.",
            "",
        ]

        for index, result in enumerate(results, 1):

            title = (
                result.get("title")
                or "Untitled"
            )

            url = (
                result.get("url")
                or ""
            )

            content = (
                result.get("content")
                or ""
            )

            published = result.get(
                "published_date"
            )

            lines.append(
                f"[{index}] {title}"
            )

            if url:
                lines.append(
                    f"URL: {url}"
                )

            if published:
                lines.append(
                    f"Published: {published}"
                )

            if content:
                lines.append(
                    f"Content: {content}"
                )

            lines.append("")

        lines.append(
            "When using information from these results, "
            "cite the provided URLs as Markdown links."
        )

        return "\n".join(lines)

    except Exception as e:

        logger.error(
            "Tavily web search error: %s",
            e,
        )

        raise GroqServiceError(
            "Web search failed."
        ) from e


# ============================================================
# SAFE SEARCH
# ============================================================

async def _safe_search(query: str) -> str:

    try:

        return await search_web(query)

    except GroqServiceError as e:

        logger.warning(
            "Web search unavailable: %s",
            e,
        )

        return (
            "LIVE WEB SEARCH RESULTS:\n"
            "Web search was unavailable for this request. "
            "Answer using your existing knowledge and clearly "
            "state that live information could not be verified."
        )


# ============================================================
# NORMAL NON-STREAMING CHAT
# ============================================================

async def ask_groq(
    chat_id: str,
    messages,
    web_search: bool = False,
):

    try:

        system_content = await _build_system_content(
            chat_id
        )

        last_message = ""

        if messages:
            last_message = (
                messages[-1].get("content", "")
                or ""
            )

        # Globe ON:
        # Always search.
        #
        # Globe OFF:
        # Search automatically only when the question looks
        # time-sensitive.
        should_search = (
            web_search
            or _needs_web_search(last_message)
        )

        web_context = ""

        if should_search:

            logger.info(
                "Web search enabled for query: %s",
                last_message[:200],
            )

            web_context = await _safe_search(
                last_message
            )

        chat_messages = [
            {
                "role": "system",
                "content": _with_web_context(
                    system_content,
                    web_context,
                ),
            },
            *messages,
        ]

        # Main Groq request.
        #
        # reasoning_effort="low" prevents GPT-OSS from spending
        # excessive time reasoning before producing visible text.
        completion = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=chat_messages,
            stream=False,
            reasoning_effort="low",
            max_tokens=2048,
        )

        if not completion.choices:
            raise GroqServiceError(
                "Groq returned no choices."
            )

        answer = (
            completion.choices[0].message.content
            or ""
        ).strip()

        if not answer:
            raise GroqServiceError(
                "Groq returned an empty response."
            )

        return answer

    except (
        APIConnectionError,
        APIStatusError,
        APIError,
    ) as e:

        logger.error(
            "Groq API error in ask_groq: %s",
            e,
        )

        raise GroqServiceError(
            str(e)
        ) from e

    except GroqServiceError:
        raise

    except Exception as e:

        logger.exception(
            "Unexpected error in ask_groq: %s",
            e,
        )

        raise GroqServiceError(
            str(e)
        ) from e


# ============================================================
# STREAMING CHAT
# ============================================================

async def stream_groq(
    chat_id: str,
    messages,
    web_search: bool = False,
):

    try:

        system_content = await _build_system_content(
            chat_id
        )

        last_message = ""

        if messages:
            last_message = (
                messages[-1].get("content", "")
                or ""
            )

        # Globe ON = always search.
        # Globe OFF = local instant decision.
        should_search = (
            web_search
            or _needs_web_search(last_message)
        )

        web_context = ""

        if should_search:

            logger.info(
                "Streaming web search enabled for query: %s",
                last_message[:200],
            )

            web_context = await _safe_search(
                last_message
            )

        chat_messages = [
            {
                "role": "system",
                "content": _with_web_context(
                    system_content,
                    web_context,
                ),
            },
            *messages,
        ]

        # Start Groq streaming.
        #
        # There is NO router model here.
        # There is NO Groq tool call here.
        # There is only Tavily -> Groq -> stream.
        stream = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=chat_messages,
            stream=True,
            reasoning_effort="low",
            max_tokens=2048,
        )

    except (
        APIConnectionError,
        APIStatusError,
        APIError,
    ) as e:

        logger.error(
            "Groq API error starting stream: %s",
            e,
        )

        raise GroqServiceError(
            str(e)
        ) from e

    except GroqServiceError:
        raise

    except Exception as e:

        logger.exception(
            "Unexpected streaming error: %s",
            e,
        )

        raise GroqServiceError(
            str(e)
        ) from e

    try:

        async for chunk in stream:

            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            if delta.content:
                yield delta.content

    except (
        APIConnectionError,
        APIStatusError,
        APIError,
    ) as e:

        logger.error(
            "Groq API error during stream: %s",
            e,
        )

        yield (
            "\n\n⚠ Lost connection to the AI service. "
            "Please try again."
        )

    except Exception as e:

        logger.exception(
            "Unexpected streaming error: %s",
            e,
        )

        yield (
            "\n\n⚠ Something went wrong while "
            "generating the response."
        )


# ============================================================
# MEMORY EXTRACTION
# ============================================================
#
# This uses GPT-OSS 20B instead of the DEAD router model.
#
# It runs as a background task, so it doesn't block the user's
# visible answer.
# ============================================================

async def extract_memory(
    chat_id: str,
    user_message: str,
):

    try:

        completion = await client.chat.completions.create(
            model="openai/gpt-oss-20b",
            temperature=0,
            max_tokens=300,
            reasoning_effort="low",
            response_format={
                "type": "json_object"
            },
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

or:

{
    "remember": false
}
""",
                },
                {
                    "role": "user",
                    "content": user_message,
                },
            ],
        )

        raw = (
            completion.choices[0]
            .message
            .content
            or "{}"
        )

        result = json.loads(raw)

        if (
            result.get("remember")
            and result.get("key")
            and result.get("value")
        ):

            await add_memory(
                chat_id,
                result["key"],
                result["value"],
            )

    except Exception as e:

        # Memory is optional.
        # Never allow memory errors to break chat.
        logger.warning(
            "Memory extraction skipped: %s",
            e,
        )


# ============================================================
# CHAT TITLE
# ============================================================

def _fallback_title(
    message: str,
) -> str:

    words = (
        message
        .strip()
        .split()
    )

    if not words:
        return "New Chat"

    return " ".join(
        words[:6]
    )[:50]


async def generate_chat_title(
    first_message: str,
) -> str:

    try:

        completion = await client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0.2,
            max_tokens=200,
            reasoning_effort="low",
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
""",
                },
                {
                    "role": "user",
                    "content": first_message,
                },
            ],
        )

        raw = (
            completion.choices[0]
            .message
            .content
            or ""
        )

        title = (
            raw
            .strip()
            .strip("\"'")
        )

        if title:
            return title

        logger.warning(
            "Title generation returned empty content."
        )

    except Exception as e:

        logger.warning(
            "Title generation failed: %s",
            e,
        )

    return _fallback_title(
        first_message
    )
