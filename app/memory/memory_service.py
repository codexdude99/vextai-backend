import json
from pathlib import Path

MEMORY_FILE = Path(__file__).parent / "memory.json"


def _load_all():
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


def _save_all(data):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def add_memory(chat_id: str, key: str, value: str):
    data = _load_all()
    memories = data.setdefault(chat_id, [])

    for memory in memories:
        if memory["key"] == key:
            memory["value"] = value
            _save_all(data)
            return

    memories.append({
        "key": key,
        "value": value,
    })

    _save_all(data)


def format_memory(chat_id: str) -> str:
    data = _load_all()
    memories = data.get(chat_id, [])

    if not memories:
        return ""

    text = "Known facts about the user in this conversation:\n\n"

    for memory in memories:
        text += f"- {memory['key']}: {memory['value']}\n"

    return text


def clear_memory(chat_id: str):
    data = _load_all()
    if chat_id in data:
        del data[chat_id]
        _save_all(data)
