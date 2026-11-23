"""Guardrails for a web server that runs on your laptop.

"It only listens on 127.0.0.1" stops other computers, but not other *web pages*: any site open
in your browser can send a form POST to http://127.0.0.1:8000. That's cross-site request
forgery (CSRF), and here it could approve a topic removal or release a flagged item. So every
request that changes something must:

1. carry the `HX-Request` header. htmx adds it; a plain cross-site form can't add custom
   headers, and a script on another site would need a CORS preflight, which this app never
   allows.
2. come from this app's own origin, when the browser says where it comes from (`Origin`).

GET requests only read, so they need neither. A separate `TrustedHostMiddleware` (in app.py)
refuses requests whose Host isn't localhost, which blocks DNS-rebinding tricks.
"""

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import PlainTextResponse

SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
ALLOWED_HOSTS = ["127.0.0.1", "localhost"]


class SameOriginWrites(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.method not in SAFE_METHODS:
            if request.headers.get("HX-Request") != "true":
                return PlainTextResponse("Writes must come from the Radar page itself.", 403)
            origin = request.headers.get("Origin")
            if origin is not None and origin != f"{request.url.scheme}://{request.url.netloc}":
                return PlainTextResponse("Cross-site request refused.", 403)
        return await call_next(request)
