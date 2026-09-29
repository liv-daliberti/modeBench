#!/usr/bin/env python3
"""Verify one ModeBench Python-factor response through the external worker."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modebench.python_modebench_process import (  # noqa: E402
    validate_python_factor_function_external,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--reference", required=True)
    args = parser.parse_args()

    try:
        spec = json.loads(args.reference)
    except json.JSONDecodeError as error:
        raise SystemExit(f"invalid --reference JSON: {error}") from error
    validation = validate_python_factor_function_external(args.candidate, spec)
    if validation is None:
        print(json.dumps({"valid": False}, sort_keys=True))
        raise SystemExit(1)
    print(
        json.dumps(
            {
                "valid": True,
                "canonical_key": validation.canonical_key,
                "outputs": list(validation.outputs),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
