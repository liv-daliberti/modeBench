"""Deadline-bounded subprocess RPC, including partial reads and blocked writes."""
from __future__ import annotations
import json
import math
import os
import selectors
import signal
import subprocess
import sys
import threading
import time

from .diagnostics import MAX_REQUEST_BYTES, MAX_REPLY_BYTES, STATUSES, result


class BoundedWorker:
    def __init__(self, *, timeout_seconds=5.0, module='modebench.verifier_worker'):
        if isinstance(timeout_seconds, bool) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError('timeout_seconds must be positive and finite')
        self.timeout_seconds = float(timeout_seconds)
        self.module = module
        self._process = None
        self._owner_pid = os.getpid()
        self._lock = threading.Lock()

    def _ensure_owner(self):
        if self._owner_pid != os.getpid():
            process, self._process = self._process, None
            self._owner_pid = os.getpid()
            self._lock = threading.Lock()
            if process is not None:
                for handle in (process.stdin, process.stdout):
                    if handle is not None:
                        handle.close()

    def worker_command(self):
        return [sys.executable, '-I', '-m', self.module]

    def _start(self):
        if self._process is not None:
            if self._process.poll() is None:
                return self._process
            self._stop()
            raise OSError('worker exited between requests')
        self._process = subprocess.Popen(self.worker_command(), stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                        start_new_session=True, bufsize=0)
        os.set_blocking(self._process.stdin.fileno(), False)
        os.set_blocking(self._process.stdout.fileno(), False)
        return self._process

    def _stop(self):
        self._ensure_owner()
        process, self._process = self._process, None
        if process is None:
            return
        # Terminate the process group even if its leader has already exited.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=0.5)
        for handle in (process.stdin, process.stdout):
            if handle is not None:
                handle.close()

    def close(self):
        self._ensure_owner()
        with self._lock:
            self._stop()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()

    def __del__(self):
        try:
            self._stop()
        except Exception:
            pass

    def request(self, payload):
        try:
            raw = (json.dumps(payload, allow_nan=False) + '\n').encode()
        except (TypeError, ValueError, RecursionError):
            return result('invalid_reference', detail='request is not finite JSON')
        if len(raw) > MAX_REQUEST_BYTES:
            return result('resource_limit', detail='request byte limit exceeded')
        self._ensure_owner()
        with self._lock:
            try:
                deadline = time.monotonic() + self.timeout_seconds
                process = self._start()
                received, sent = bytearray(), 0
                with selectors.DefaultSelector() as selector:
                    selector.register(process.stdin, selectors.EVENT_WRITE)
                    selector.register(process.stdout, selectors.EVENT_READ)
                    while True:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise TimeoutError('parent request deadline exceeded')
                        events = selector.select(remaining)
                        for key, _mask in events:
                            try:
                                if key.fileobj is process.stdin:
                                    sent += os.write(process.stdin.fileno(), raw[sent:sent+4096])
                                    if sent == len(raw):
                                        selector.unregister(process.stdin)
                                else:
                                    chunk = os.read(process.stdout.fileno(), 65536)
                                    if not chunk:
                                        raise OSError('worker closed stdout')
                                    received.extend(chunk)
                            except BlockingIOError:
                                continue
                        if len(received) > MAX_REPLY_BYTES:
                            self._stop()
                            return result('worker_failure', detail='worker reply byte limit exceeded')
                        if b'\n' in received:
                            line, extra = received.split(b'\n', 1)
                            if extra.strip():
                                raise ValueError('unsolicited worker output')
                            reply = json.loads(line)
                            if not isinstance(reply, dict) or reply.get('status') not in STATUSES:
                                raise ValueError('invalid worker status')
                            correct = reply['status'] == 'correct'
                            if type(reply.get('verified')) is not bool or reply['verified'] != correct:
                                raise ValueError('invalid worker verification flag')
                            if correct and payload.get('operation') != 'reference':
                                if not isinstance(reply.get('canonical_key'), str) or not reply['canonical_key']:
                                    raise ValueError('missing worker canonical key')
                            elif reply.get('canonical_key') is not None:
                                raise ValueError('unexpected worker canonical key')
                            if not isinstance(reply.get('graded_text'), str):
                                raise ValueError('invalid worker graded text')
                            if reply['status'] in ('timeout', 'worker_failure', 'resource_limit'):
                                self._stop()
                            return reply
            except TimeoutError as error:
                self._stop()
                return result('timeout', detail=str(error))
            except (OSError, ValueError, TypeError, KeyError) as error:
                self._stop()
                return result('worker_failure', detail=f'{type(error).__name__}: {error}')
            except BaseException:
                self._stop()
                raise
