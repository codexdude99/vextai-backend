import asyncio
import json
from pathlib import Path

MEMORY_FILE = Path(__file__).parent / "memory.json"

# add_memory() and format_memory() run on (or ahead of) the request path
# for every single chat message -- format_memory() is called on every
# reply, add_memory() on every background memory-extraction task. All the
# actual file I/O below is synchronous. Left as plain `def`s called
# directly from async code, each read/write blocks the *entire* FastAPI
# event loop for its duration -- not just the one request, every
# concurrently connected user stalls too. Wrapping the sync work in
# asyncio.to_thread() moves it off the event loop onto a worker thread, so
# one chat's memory read/write can no longer stall everyone else's request.
# The synchronous helpers (_*_sync) keep the original logic untouched;
# only how they're called has changed.


def _load_all_sync():
    if not MEMORY_FILE.exists():
        return {}

    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            return {}

    # Old version stored one global list for every visitor. If that file
    # is still around, treat it as empty instead of leaking it into a
    # random chat.
    if isinstance(data, list):
        return {}

    return data


def _save_all_sync(data):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def _add_memory_sync(chat_id: str, key: str, value: str):
    data = _load_all_sync()
    memories = data.setdefault(chat_id, [])

    for memory in memories:
        if memory["key"] == key:
            memory["value"] = value
            _save_all_sync(data)
            return

    memories.append({
        "key": key,
        "value": value,
    })

    _save_all_sync(data)


def _format_memory_sync(chat_id: str) -> str:
    data = _load_all_sync()
    memories = data.get(chat_id, [])

    if not memories:
        return ""

    text = "Known facts about the user in this conversation:\n\n"

    for memory in memories:
        text += f"- {memory['key']}: {memory['value']}\n"

    return text


def _clear_memory_sync(chat_id: str):
    data = _load_all_sync()
    if chat_id in data:
        del data[chat_id]
        _save_all_sync(data)


async def add_memory(chat_id: str, key: str, value: str):
    await asyncio.to_thread(_add_memory_sync, chat_id, key, value)


async def format_memory(chat_id: str) -> str:
    return await asyncio.to_thread(_format_memory_sync, chat_id)


async def clear_memory(chat_id: str):
    await asyncio.to_thread(_clear_memory_sync, chat_id)
