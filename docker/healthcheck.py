"""Container-local health probe for the offline demo metrics server."""
from __future__ import annotations

from urllib.request import urlopen


with urlopen("http://127.0.0.1:8000/healthz", timeout=2) as response:
    if response.status != 200 or response.read().strip() != b"ok":
        raise SystemExit(1)
