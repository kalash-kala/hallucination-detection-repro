#!/usr/bin/env python
"""Stage 24 -- validate store entries. Thin wrapper over `csx-store verify`.

Exists so the extraction pipeline has a numbered final stage that matches the
others' interface; the real implementation is csx_extract/verify.py, which is
importable without torch so it runs anywhere.

Usage:
    python cli_extract/24_verify.py --pairs qwen25vl_advqa
    python cli_extract/24_verify.py --all
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from csx_extract.verify import main as verify_main  # noqa: E402


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Translate this stage's --pairs to verify's --pair, one call per pair, so
    # both spellings work and the exit code is the worst of them.
    if "--pairs" in argv:
        i = argv.index("--pairs")
        keys = [k.strip() for k in argv[i + 1].split(",") if k.strip()]
        rest = argv[:i] + argv[i + 2:]
        return max(verify_main(["verify", "--pair", k, *rest]) for k in keys)
    return verify_main(["verify", *argv])


if __name__ == "__main__":
    raise SystemExit(main())
