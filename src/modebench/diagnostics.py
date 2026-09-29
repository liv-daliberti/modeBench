"""Verifier statuses and explicit failure policy shared by workers and callers."""
STATUSES = frozenset({'correct', 'incorrect', 'malformed', 'timeout', 'invalid_reference', 'worker_failure', 'resource_limit'})
UNSCORABLE = frozenset({'timeout', 'invalid_reference', 'worker_failure', 'resource_limit'})
MAX_REQUEST_BYTES = 1_048_576
MAX_REPLY_BYTES = 262_144
MAX_REFERENCE_BYTES = 65_536
MAX_RESPONSE_CHARS = 131_072
WORKER_MEMORY_BYTES = 1_073_741_824
PYTHON_DEADLINE_SECONDS = 0.25
DOMAIN_DEADLINE_SECONDS = 2.0


def result(status, *, key=None, text='', detail=None, **extra):
    assert status in STATUSES
    return {'status': status, 'verified': status == 'correct', 'canonical_key': key,
            'graded_text': text, 'detail': detail, **extra}


class VerifierExecutionError(RuntimeError):
    """Legacy key-only APIs raise rather than disguise infrastructure failure."""
    def __init__(self, diagnostic):
        self.diagnostic = diagnostic
        super().__init__(f"verifier {diagnostic['status']}: {diagnostic.get('detail')}")
