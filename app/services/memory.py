from collections import defaultdict

conversations = defaultdict(list)


def get_chat(chat_id: str):
    return conversations[chat_id]


def clear_chat(chat_id: str):
    conversations[chat_id] = []