"""Start one AI process; configuration remains environment-owned."""
from __future__ import annotations

import os


def main() -> None:
    # Startup retains the existing Hub warm-up, provider policy and admission.
    # --workers is deliberately fixed at one until mutable stores coordinate.
    os.execvp("python", ["python", "-m", "uvicorn", "api.app:app", "--host", "0.0.0.0",
                         "--port", "8000", "--workers", "1"])


if __name__ == "__main__":
    main()
