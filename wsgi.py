"""wsgi.py — WSGI adapter for the SDG Paper Matcher.

Lets the zero-dependency web app run on WSGI-only hosts (PythonAnywhere
free tier, gunicorn, uvicorn, ...) without touching web/app.py.

The app's handler is a http.server.BaseHTTPRequestHandler, which only needs
a socket-like object with makefile(). We feed it a fully assembled raw
HTTP request built from the WSGI environ and capture the response it writes.

Run locally to check:

    python3 -c "from wsgi import application; print(application)"

Deploy on PythonAnywhere:
    1. git clone this repo into your home dir
    2. python3 engine/sdg2sqlite.py          (build the query DB)
    3. Web tab -> Add a new web app -> Manual configuration -> Python 3.11
       Source dir: /home/<user>/sdg-paper-matcher
    4. In the WSGI configuration file, keep only the sys.path line and add:
         from wsgi import application
    5. Reload -> https://<user>.pythonanywhere.com
"""

from __future__ import annotations

import io
import sys
import traceback
from pathlib import Path

# repo root (engine/ is imported by web/app.py from there)
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "web"))

from web.app import Handler, sentry_report  # noqa: E402


class _Socket(io.BytesIO):
    """Minimal socket stand-in. BaseHTTPRequestHandler only calls
    makefile('rb') (request input) and makefile('wb') (response output)
    on it; StreamRequestHandler.setup() also reads .connection."""

    def __init__(self, data: bytes):
        super().__init__(data)
        self._wbuf = io.BytesIO()

    def makefile(self, mode: str = "r", *args, **kwargs):  # noqa: ARG002
        if "w" in mode:
            return self._wbuf
        self.seek(0)
        return self

    def sendall(self, data: bytes) -> None:  # never used, but harmless
        self._wbuf.write(data)


def application(environ, start_response):
    """Standard WSGI callable -> runs one request through the app's handler."""
    try:
        body = (environ.get("wsgi.input") or io.BytesIO(b"")).read()
        method = environ.get("REQUEST_METHOD", "GET")
        target = environ.get("PATH_INFO", "/")
        qs = environ.get("QUERY_STRING", "")
        if qs:
            target += "?" + qs
        proto = environ.get("SERVER_PROTOCOL", "HTTP/1.1")

        headers = []
        for key, value in environ.items():
            if key.startswith("HTTP_"):
                headers.append((key[5:].replace("_", "-"), value))
            elif key in ("CONTENT_TYPE", "CONTENT_LENGTH"):
                headers.append((key.replace("_", "-"), str(value)))
        raw = ("%s %s %s\r\n%s\r\n" % (method, target, proto,
               "".join("%s: %s\r\n" % h for h in headers))).encode("utf-8", "replace") + body

        sock = _Socket(raw)
        Handler(sock, ("127.0.0.1", 0), None)  # constructs -> handles -> responds
        out = sock._wbuf.getvalue()
    except Exception as exc:  # noqa: BLE001 — surface any failure + report it
        sentry_report("error", "wsgi.adapter", f"WSGI adapter error: {exc}", exc=exc,
                      extra={"traceback": traceback.format_exc()})
        start_response("500 Internal Server Error",
                       [("Content-Type", "text/plain; charset=utf-8")])
        return [("WSGI adapter error: %s" % exc).encode("utf-8")]

    head, _, resp_body = out.partition(b"\r\n\r\n")
    lines = head.split(b"\r\n")
    parts = lines[0].split(b" ", 2)
    status = ("%s %s" % (parts[1].decode(), parts[2].decode())) if len(parts) == 3 else "200 OK"
    resp_headers = []
    for line in lines[1:]:
        if b":" in line:
            name, _, value = line.partition(b":")
            resp_headers.append((name.decode().strip(), value.decode().strip()))
    start_response(status, resp_headers)
    return [resp_body]


if __name__ == "__main__":
    print("WSGI adapter OK — use 'from wsgi import application' in your WSGI file.")
