# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the agent-switch authors. See LICENSE.

"""A scriptable fake model server for provider tests."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


class FakeServer:
    """Routes map (method, path) to (status, body) or a callable(request_body) -> (status, body)."""

    def __init__(self):
        self.routes = {}
        self.requests = []
        self.header_logs = []
        handler = self._handler()
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.base = f"http://127.0.0.1:{self._server.server_address[1]}"
        threading.Thread(target = self._server.serve_forever, args = (0.05,), daemon = True).start()

    def route(self, method, path, status = 200, body = None):
        self.routes[(method, path)] = (status, body) if not callable(status) else status

    def _handler(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            def _serve(self, method):
                path = self.path.split("?", 1)[0]
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                try:
                    payload = json.loads(raw) if raw else None
                except ValueError:
                    payload = raw.decode()
                server.requests.append((method, self.path, payload, self.headers.get("Authorization")))
                server.header_logs.append((method, path, dict(self.headers)))
                answer = server.routes.get((method, path))
                if answer is None:
                    status, body = 404, {"error": "not found"}
                elif callable(answer):
                    status, body = answer(payload)
                else:
                    status, body = answer
                data = body.encode() if isinstance(body, str) else json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "text/plain" if isinstance(body, str) else "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                self._serve("GET")

            def do_POST(self):
                self._serve("POST")

            def do_DELETE(self):
                self._serve("DELETE")

            def log_message(self, *args):
                pass

        return Handler

    def close(self):
        self._server.shutdown()


@pytest.fixture
def fake_server():
    server = FakeServer()
    yield server
    server.close()
