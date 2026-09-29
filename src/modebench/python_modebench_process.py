"""Killable external execution boundary for ModeBench Python functions."""

from __future__ import annotations

import json
import os
from pathlib import Path
import select
import subprocess
import sys
import threading
from typing import Any, Mapping

from .python_modebench import PythonFactorValidation


class PythonFactorVerifierProcess:
    """Execute restricted candidate functions outside the trainer process."""

    def __init__(self, *, timeout_seconds: float = 1.0) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.timeout_seconds = float(timeout_seconds)
        self._process: subprocess.Popen[str] | None = None
        self._owner_pid = os.getpid()
        self._lock = threading.Lock()

    def _start(self) -> subprocess.Popen[str]:
        if self._owner_pid != os.getpid():
            self._process = None
            self._owner_pid = os.getpid()
        process = self._process
        if process is not None and process.poll() is None:
            return process
        worker_env = os.environ.copy()
        source_root = str(Path(__file__).resolve().parents[1])
        inherited_pythonpath = worker_env.get("PYTHONPATH")
        worker_env["PYTHONPATH"] = (
            source_root
            if not inherited_pythonpath
            else source_root + os.pathsep + inherited_pythonpath
        )
        isolated_entrypoint = (
            "import runpy,sys;"
            f"sys.path.insert(0,{source_root!r});"
            "runpy.run_module('modebench.python_modebench_worker',run_name='__main__')"
        )
        self._process = subprocess.Popen(
            [sys.executable, "-I", "-c", isolated_entrypoint],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            encoding="utf-8",
            bufsize=1,
            start_new_session=True,
            env=worker_env,
        )
        return self._process

    def _stop(self) -> None:
        process, self._process = self._process, None
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1)
        if process.stdin is not None:
            process.stdin.close()
        if process.stdout is not None:
            process.stdout.close()

    def close(self) -> None:
        with self._lock:
            self._stop()

    def validate(
        self,
        candidate: str,
        spec: Mapping[str, Any],
    ) -> PythonFactorValidation | None:
        request = json.dumps(
            {"candidate": str(candidate), "spec": dict(spec)},
            allow_nan=False,
        )
        with self._lock:
            for attempt in range(2):
                process = self._start()
                try:
                    assert process.stdin is not None and process.stdout is not None
                    process.stdin.write(request + "\n")
                    process.stdin.flush()
                    readable, _, _ = select.select(
                        [process.stdout], [], [], self.timeout_seconds
                    )
                    if not readable:
                        self._stop()
                        return None
                    line = process.stdout.readline()
                    if not line:
                        raise BrokenPipeError("Python ModeBench worker closed stdout")
                    payload = json.loads(line)
                    if not payload.get("valid"):
                        return None
                    return PythonFactorValidation(
                        canonical_key=str(payload["canonical_key"]),
                        outputs=tuple(int(value) for value in payload["outputs"]),
                    )
                except (BrokenPipeError, OSError, ValueError, json.JSONDecodeError):
                    self._stop()
                    if attempt:
                        return None
        return None

    def __del__(self) -> None:
        try:
            self._stop()
        except Exception:
            pass


_SHARED_VERIFIER = PythonFactorVerifierProcess()


def validate_python_factor_function_external(
    candidate: str,
    spec: Mapping[str, Any],
) -> PythonFactorValidation | None:
    """Validate a candidate through the shared external Python worker."""

    return _SHARED_VERIFIER.validate(candidate, spec)
