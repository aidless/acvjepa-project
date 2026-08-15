"""Local-only Prometheus exposition server for the Compose corruption demo.

The server runs one deterministic in-memory corruption/fallback demo at startup
and exposes only its sanitized low-cardinality metrics and report. It has no
network clients, no credentials, no mutable checkpoint backend, and no cluster
control code.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from checkpoint_integrity_corruption_demo import run_demo


class DemoState:
    def __init__(self) -> None:
        report, exposition = run_demo()
        self.report = json.dumps({"mode": "local-offline-synthetic", "report": asdict(report)}, sort_keys=True).encode("utf-8")
        self.exposition = exposition.encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    state: DemoState

    def log_message(self, format: str, *args: object) -> None:  # Keep container logs free of request payloads.
        return

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/healthz":
            self._send(HTTPStatus.OK, b"ok\n", "text/plain; charset=utf-8")
        elif self.path == "/metrics":
            self._send(HTTPStatus.OK, self.state.exposition, "text/plain; version=0.0.4; charset=utf-8")
        elif self.path == "/report":
            self._send(HTTPStatus.OK, self.state.report, "application/json; charset=utf-8")
        else:
            self._send(HTTPStatus.NOT_FOUND, b"not found\n", "text/plain; charset=utf-8")

    def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve local offline chaos-demo metrics")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    Handler.state = DemoState()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(json.dumps({"local_demo": "ready", "port": args.port, "safety": "offline_synthetic_only"}, sort_keys=True), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
