from __future__ import annotations

import json
import os
import sys

from localrag.mcp.app import handle_stdio_message


def main() -> None:
    api_key = os.environ.get("MCP_API_KEY")
    for line in sys.stdin:
        if not line.strip():
            continue
        response = handle_stdio_message(json.loads(line), api_key)
        if response is not None:
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
