"""Keep test runs quiet unless a test explicitly asserts on logs."""

from __future__ import annotations

import os

os.environ.setdefault("LOG_LEVEL", "ERROR")
