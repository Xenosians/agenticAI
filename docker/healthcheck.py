"""Read real AI readiness without importing the model runtime."""
from __future__ import annotations

import json
import sys
from urllib.request import urlopen


def main() -> int:
    try:
        with urlopen("http://127.0.0.1:8000/ready", timeout=3) as response:
            if response.status != 200:
                return 1
            body = json.loads(response.read(4096))
        return 0 if body == {"status": "ready", "service": "itsm-ai"} else 1
    except (OSError, ValueError, TypeError):
        return 1


if __name__ == "__main__":
    sys.exit(main())
