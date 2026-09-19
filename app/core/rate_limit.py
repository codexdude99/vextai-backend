import time
from collections import defaultdict, deque

from fastapi import Request, HTTPException

from app.core.config import RATE_LIMIT_MAX_REQUESTS, RATE_LIMIT_WINDOW_SECONDS

# Simple in-memory sliding window, per IP address.
# Good enough for a single backend process. If you later run multiple
# server processes/instances behind a load balancer, move this to Redis
# so all instances share the same counts.
_hits = defaultdict(deque)


def rate_limit(request: Request):
    ip = request.client.host if request.client else "unknown"
    now = time.time()
    hits = _hits[ip]

    while hits and now - hits[0] > RATE_LIMIT_WINDOW_SECONDS:
        hits.popleft()

    if len(hits) >= RATE_LIMIT_MAX_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please slow down and try again shortly.",
        )

    hits.append(now)
