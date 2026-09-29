"""JSON-lines worker that executes restricted ModeBench Python functions."""

from __future__ import annotations

import json
import signal
import sys

from .python_modebench import execute_python_factor_candidate


def _timeout(_signum, _frame) -> None:
    raise TimeoutError("Python ModeBench candidate timed out")


def main() -> None:
    signal.signal(signal.SIGALRM, _timeout)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            signal.setitimer(signal.ITIMER_REAL, 0.25)
            validation = execute_python_factor_candidate(
                request["candidate"],
                request["spec"],
            )
            payload = {
                "valid": True,
                "canonical_key": validation.canonical_key,
                "outputs": list(validation.outputs),
            }
        except Exception as error:
            payload = {
                "valid": False,
                "error": type(error).__name__,
            }
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
        sys.stdout.write(json.dumps(payload, allow_nan=False) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
