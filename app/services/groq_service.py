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

Web search:
- You have a `search_web` tool that fetches live results from the internet. It costs a couple of extra seconds, so use it with judgment.
- Call it automatically, without asking the user for permission first, whenever the answer depends on something that can change over time or that you can't be fully confident is still accurate as of today: news, current events, prices, scores, weather, schedules, software/product versions, who currently holds a role or title, or anything about "today," "now," "latest," "current," "this year," etc.
- Do NOT call it for stable facts, definitions, math, code, or general knowledge that doesn't change — answer those directly.
- When you do search, base your answer on the returned results, prefer the most recent and relevant ones, and cite them inline as Markdown links using the URLs provided. Never claim you searched the web if you did not, and never invent a source URL.

Formatting:
- Always use proper Markdown for code blocks.
- Keep formatting clean and readable. 🚀
- Do NOT insert horizontal-rule dividers (---) between sections of an answer. Use a heading, a bold label, or just a blank line to separate ideas instead.
- Only reach for a Markdown table when the content is genuinely tabular (the same few attributes repeated across rows, like comparing options). Don't use a table to summarize a file list, a set of steps, or a project structure — a short list, or a fenced code block showing a tree, reads better for that.
- Never put raw HTML tags like <br> inside a table cell or anywhere else in a reply — use plain Markdown (a new line, a separate bullet, or a shorter cell) instead.
"""

# Tool schema handed to the model so it can decide, on its own, when a
# question needs live web results. gpt-oss-120b supports tool calling but
# not *parallel* tool calls, so it will request this one tool at a time.
WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_web",
        "description": (
            "Search the live web for up-to-date information. Use this for anything "
            "that can change over time or that you can't be fully sure is still "
            "accurate today: news, current events, prices, sports scores, weather, "
            "release/version info, schedules, who currently holds a role, or "
            "anything about 'today', 'now', 'latest', 'current', 'this year'. Do "
            "not use it for stable facts, definitions, math, or code."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A short, focused search query for what needs to be looked up.",
                }
            },
            "required": ["query"],
        },
    },
}

# Safety cap on how many times we let the model call search_web in a single
# reply before forcing a final answer. Keeps a confused/looping model from
# turning one chat message into a chain of slow tool calls.
MAX_TOOL_ROUNDS = 2


class GroqServiceError(Exception):
    """Raised when the Groq API call fails, so routes can return a clean error
    instead of an unhandled 500 traceback."""


def _current_date_context() -> str:
    """Gives the model today's date so it can actually judge what counts as
    'current' or 'latest' -- without this it has no way to know if its own
    knowledge is stale, which is what automatic web search decisions depend on."""

    now = datetime.now(timezone.utc)
    return f"Today's date is {now.strftime('%A, %B %d, %Y')} (UTC)."


def _build_system_content(chat_id: str) -> str:
    parts = [SYSTEM_PROMPT.strip(), _current_date_context()]

    memory_text = format_memory(chat_id)
    if memory_text:
        parts.append(memory_text)

    return "\n\n".join(parts)


def _with_web_context(system_content: str, web_context: str) -> str:
    if not web_context:
        return system_content

    return (
        system_content
        + "\n\n" + web_context +
        "\n\nIMPORTANT: You are answering with live web research. "
        "Do not claim you searched the web if no results were provided. "
        "Do not invent source URLs."
    )


def _extract_query_from_args(arguments: str, fallback: str) -> str:
    """Pulls the `query` argument out of a tool call's (possibly malformed)
    JSON arguments string. Falls back to the user's own message if parsing
    fails -- a tool call should never be allowed to hard-crash a reply."""

    try:
        args = json.loads(arguments or "{}")
        query = (args.get("query") or "").strip()
        return query or fallback
    except (json.JSONDecodeError, AttributeError):
        return fallback


def _extract_query(tool_call, fallback: str) -> str:
    """Same as above, but for a tool_call object as returned by the SDK
    (non-streaming path) rather than a raw arguments string."""

    try:
        return _extract_query_from_args(tool_call.function.arguments, fallback)
    except AttributeError:
        return fallback


async def search_web(query: str) -> str:
    """Search Tavily and turn the results into compact context for Groq."""
    if not TAVILY_API_KEY:
        raise GroqServiceError("TAVILY_API_KEY is not configured.")

    try:
        import httpx

        payload = {
            "query": query,
            "search_depth": "basic",
            "max_results": WEB_SEARCH_MAX_RESULTS,
            "topic": "general",
            "include_answer": False,
            "include_raw_content": False,
            "include_published_date": True,
        }

        async with httpx.AsyncClient(timeout=15.0) as http:
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
            return "No useful web results were found."

        lines = [
            "LIVE WEB SEARCH RESULTS:",
            "Use these sources to answer the user's question. Prefer the most relevant and recent information.",
            ""
        ]

        for i, result in enumerate(results, 1):
            title = result.get("title", "Untitled")
            url = result.get("url", "")
            content = result.get("content", "")
            published = result.get("published_date")
            lines.append(f"[{i}] {title}")
            lines.append(f"URL: {url}")
            if published:
                lines.append(f"Published: {published}")
            lines.append(f"Content: {content}")
            lines.append("")

        lines.append("When using information from these results, cite them inline as Markdown links using the provided URLs.")
        return "\n".join(lines)

    except GroqServiceError:
        raise
    except Exception as e:
        logger.error("Tavily web search error: %s", e)
        raise GroqServiceError("Web search failed.") from e


async def _safe_search(query: str) -> str:
    """Wraps search_web so a Tavily hiccup (missing key, timeout, rate limit)
    degrades the reply instead of crashing it -- same philosophy as the rest
    of this file: a broken dependency should never 500 the whole chat."""

    try:
        return await search_web(query)
    except GroqServiceError as e:
        logger.warning("Web search unavailable, continuing without it: %s", e)
        return (
            "Web search is currently unavailable. Answer from what you already "
            "know, and briefly tell the user you weren't able to verify this "
            "with a live search right now."
        )


async def ask_groq(chat_id: str, messages, web_search: bool = False):
    try:
        system_content = _build_system_content(chat_id)

        # Manual override (e.g. a "search the web" toggle in the UI): always
        # search once up front using the user's own message as the query,
        # then answer with that context. Skips letting the model decide.
        if web_search:
            web_context = await _safe_search(messages[-1]["content"])
            chat_messages = [
                {"role": "system", "content": _with_web_context(system_content, web_context)},
                *messages,
            ]
            completion = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=chat_messages,
            )
            return completion.choices[0].message.content

        # Automatic mode (default): offer the model the search_web tool and
        # let it decide, on its own, whether this particular question needs
        # live results. If Tavily isn't configured, the tool simply isn't
        # offered -- the model just answers normally, no error.
        chat_messages = [{"role": "system", "content": system_content}, *messages]
        tools_enabled = AUTO_WEB_SEARCH_ENABLED and bool(TAVILY_API_KEY)

        for _ in range(MAX_TOOL_ROUNDS):
            completion = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=chat_messages,
                tools=[WEB_SEARCH_TOOL] if tools_enabled else None,
                tool_choice="auto" if tools_enabled else None,
            )
            choice = completion.choices[0]

            if not choice.message.tool_calls:
                return choice.message.content

            # Record the assistant's tool-call turn, then feed each tool
            # result back in as its own "tool" message, per the standard
            # OpenAI/Groq function-calling flow.
            chat_messages.append(choice.message)

            for tool_call in choice.message.tool_calls:
                query = _extract_query(tool_call, messages[-1]["content"])
                web_context = await _safe_search(query)
                chat_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": web_context,
                })

        # Used up MAX_TOOL_ROUNDS and the model still wants to call tools --
        # ask one last time without offering the tool so it's forced to answer.
        completion = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=chat_messages,
        )
        return completion.choices[0].message.content

    except (APIConnectionError, APIStatusError, APIError) as e:
        logger.error("Groq API error in ask_groq: %s", e)
        raise GroqServiceError(str(e)) from e


async def stream_groq(chat_id: str, messages, web_search: bool = False):
    system_content = _build_system_content(chat_id)

    # Manual override: same idea as ask_groq -- search once up front, then
    # stream a single normal completion with that context baked in.
    if web_search:
        web_context = await _safe_search(messages[-1]["content"])
        chat_messages = [
            {"role": "system", "content": _with_web_context(system_content, web_context)},
            *messages,
        ]

        try:
            stream = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=chat_messages,
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
            logger.error("Groq API error mid-stream: %s", e)
            yield "\n\n⚠ Lost connection to the AI service. Please try again."

        return

    # Automatic mode: stream normally, but watch for the model asking to
    # call search_web. Tool-call requests arrive as fragments across many
    # chunks (id, then the function name, then the arguments piece by piece)
    # instead of as visible text, so we accumulate them separately from
    # regular content and only act once the model finishes requesting them.
    chat_messages = [{"role": "system", "content": system_content}, *messages]
    tools_enabled = AUTO_WEB_SEARCH_ENABLED and bool(TAVILY_API_KEY)

    for round_num in range(MAX_TOOL_ROUNDS + 1):
        offer_tools = tools_enabled and round_num < MAX_TOOL_ROUNDS

        try:
            stream = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=chat_messages,
                tools=[WEB_SEARCH_TOOL] if offer_tools else None,
                tool_choice="auto" if offer_tools else None,
                stream=True,
            )
        except (APIConnectionError, APIStatusError, APIError) as e:
            logger.error("Groq API error starting stream: %s", e)
            raise GroqServiceError(str(e)) from e

        tool_calls_acc = {}
        assistant_content = ""

        try:
            async for chunk in stream:
                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta

                if delta.content:
                    assistant_content += delta.content
                    yield delta.content

                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        slot = tool_calls_acc.setdefault(
                            tc.index, {"id": None, "name": None, "arguments": ""}
                        )
                        if tc.id:
                            slot["id"] = tc.id
                        if tc.function and tc.function.name:
                            slot["name"] = tc.function.name
                        if tc.function and tc.function.arguments:
                            slot["arguments"] += tc.function.arguments

        except (APIConnectionError, APIStatusError, APIError) as e:
            logger.error("Groq API error mid-stream: %s", e)
            yield "\n\n⚠ Lost connection to the AI service. Please try again."
            return

        if not tool_calls_acc:
            # Normal reply, no search needed -- done.
            return

        # The model asked to search. Replay its tool-call turn into the
        # running message list, run each search, feed the results back as
        # tool messages, then loop to start a fresh stream that continues
        # with that context in hand.
        tool_calls_list = [
            {
                "id": slot["id"],
                "type": "function",
                "function": {
                    "name": slot["name"],
                    "arguments": slot["arguments"],
                },
            }
            for slot in tool_calls_acc.values()
            if slot["id"] and slot["name"]
        ]

        if not tool_calls_list:
            # Malformed/empty tool call fragments -- nothing usable to run,
            # so just stop here rather than looping forever.
            return

        chat_messages.append({
            "role": "assistant",
            "content": assistant_content or None,
            "tool_calls": tool_calls_list,
        })

        for slot in tool_calls_acc.values():
            if not slot["id"]:
                continue
            query = _extract_query_from_args(slot["arguments"], messages[-1]["content"])
            web_context = await _safe_search(query)
            chat_messages.append({
                "role": "tool",
                "tool_call_id": slot["id"],
                "content": web_context,
            })

    # Fallback safety net: if we somehow exit the loop without returning
    # (shouldn't happen given the round cap above forces tools off on the
    # last pass), there's nothing more to yield.


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
