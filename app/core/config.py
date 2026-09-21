from dotenv import load_dotenv
import os

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is not set. Copy backend/.env.example to backend/.env "
        "and add your key (get one at https://console.groq.com/keys)."
    )

# Model + behavior are overridable via environment variables.
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-oss-120b")

MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "20"))

# Public deployment protection.
RATE_LIMIT_MAX_REQUESTS = int(os.getenv("RATE_LIMIT_MAX_REQUESTS", "20"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))

# Allowed frontend origins.
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,https://vextai.netlify.app",
    ).split(",")
    if origin.strip()
]

# Web search (Tavily). AUTO_WEB_SEARCH_ENABLED is a kill switch: set it to
# "false" to stop the model from ever calling the search tool on its own,
# without touching code. It's a no-op either way if TAVILY_API_KEY isn't set
# -- the model just answers from its own knowledge, no error.
AUTO_WEB_SEARCH_ENABLED = os.getenv("AUTO_WEB_SEARCH_ENABLED", "true").lower() == "true"
WEB_SEARCH_MAX_RESULTS = int(os.getenv("WEB_SEARCH_MAX_RESULTS", "5"))
