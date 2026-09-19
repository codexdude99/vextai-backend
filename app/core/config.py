from dotenv import load_dotenv
import os

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is not set. Copy backend/.env.example to backend/.env "
        "and add your key (get one at https://console.groq.com/keys)."
    )

# Model + behavior are now overridable via env vars instead of hardcoded,
# so you can swap models or tune history length without touching code.
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")

MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "20"))

# Public deployment protection: max requests per IP per window on the
# chat endpoints, so one visitor (or a bot) can't burn through your
# entire Groq quota.
RATE_LIMIT_MAX_REQUESTS = int(os.getenv("RATE_LIMIT_MAX_REQUESTS", "20"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))

ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]
