from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import socket


BLOCKED_MARKERS = (
    "please enable js",
    "disable any ad blocker",
    "captcha",
    "captcha-delivery",
    "datadome",
    "access denied",
)


@dataclass
class FetchResult:
    ok: bool
    text: str = ""
    status_code: int | None = None
    final_url: str | None = None
    error: str | None = None
    blocked: bool = False
    provider: str | None = None


def fetch_property_text(url: str, timeout_seconds: int = 12) -> FetchResult:
    request = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
            ),
            "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
        },
    )

    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read()
            charset = response.headers.get_content_charset() or "utf-8"
            html = raw.decode(charset, errors="replace")
            status_code = response.status
            final_url = response.url
    except HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        provider = detect_block_provider(body, dict(error.headers.items()))
        blocked = is_blocked_response(body) or error.code in {401, 403, 429}
        return FetchResult(
            ok=False,
            text=html_to_text(body),
            status_code=error.code,
            final_url=url,
            error=build_http_error(error.code, provider),
            blocked=blocked,
            provider=provider,
        )
    except URLError as error:
        return FetchResult(ok=False, final_url=url, error=str(error.reason))
    except (TimeoutError, socket.timeout):
        return FetchResult(ok=False, final_url=url, error="Timeout al leer la URL")

    blocked = is_blocked_response(html)
    provider = detect_block_provider(html, dict(response.headers.items()))
    text = html_to_text(html)
    return FetchResult(
        ok=not blocked and bool(text.strip()),
        text=text,
        status_code=status_code,
        final_url=final_url,
        error=build_http_error(status_code, provider) if blocked else None,
        blocked=blocked,
        provider=provider,
    )


def is_blocked_response(html: str) -> bool:
    normalized = html.lower()
    return any(marker in normalized for marker in BLOCKED_MARKERS)


def detect_block_provider(html: str, headers: dict[str, str] | None = None) -> str | None:
    headers = headers or {}
    header_blob = " ".join(f"{key}: {value}" for key, value in headers.items()).lower()
    normalized = f"{html.lower()} {header_blob}"
    if "datadome" in normalized or "captcha-delivery" in normalized:
        return "DataDome"
    if "cloudflare" in normalized:
        return "Cloudflare"
    return None


def build_http_error(status_code: int, provider: str | None) -> str:
    if provider:
        return f"HTTP {status_code}: bloqueo anti-bot ({provider})"
    if status_code in {401, 403, 429}:
        return f"HTTP {status_code}: acceso bloqueado o limitado por la web"
    return f"HTTP {status_code}"


def html_to_text(html: str) -> str:
    parser = VisibleTextParser()
    parser.feed(html)
    return "\n".join(line for line in parser.lines if line)


class VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.lines: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"} and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        cleaned = " ".join(data.split())
        if cleaned:
            self.lines.append(cleaned)
