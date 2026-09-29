"""Bounded, ordered prompt concurrency over isolated verifier processes."""

import threading
from collections import deque
from concurrent.futures import CancelledError, ThreadPoolExecutor

from .evaluation import grade_prepared
from .verifier import grade_with_worker
from .worker_process import BoundedWorker

MAX_WORKERS = 8


class ParallelGrader:
    def __init__(self, workers):
        self.workers = workers
        self.local = threading.local()
        self.processes = []
        self.lock = threading.Lock()
        self.cancelled = threading.Event()
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="modebench")

    def _grade(self, prepared):
        if not hasattr(self.local, "worker"):
            self.local.worker = BoundedWorker()
            with self.lock:
                self.processes.append(self.local.worker)

        def grader(level, domain, row, text):
            if self.cancelled.is_set():
                raise CancelledError("evaluation interrupted")
            return grade_with_worker(self.local.worker, level, domain, row, text)

        return grade_prepared(prepared, grader=grader)

    def results(self, prompts):
        """At most 2*workers prompts are pending; yield results in input order."""
        iterator, pending = iter(prompts), deque()
        for _ in range(2 * self.workers):
            try:
                seq, prepared = next(iterator)
            except StopIteration:
                break
            pending.append((seq, prepared, self.executor.submit(self._grade, prepared)))
        while pending:
            seq, prepared, future = pending.popleft()
            yield seq, prepared, future.result()
            try:
                seq, prepared = next(iterator)
            except StopIteration:
                continue
            pending.append((seq, prepared, self.executor.submit(self._grade, prepared)))

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.cancelled.set()
        self.executor.shutdown(wait=True, cancel_futures=True)
        for worker in self.processes:
            worker.close()
