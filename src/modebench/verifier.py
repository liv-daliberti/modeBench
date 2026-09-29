"""Public structured grading and isolated reference validation."""
import atexit
from collections.abc import Mapping
from .diagnostics import MAX_RESPONSE_CHARS, UNSCORABLE, VerifierExecutionError, result
from .worker_process import BoundedWorker

_SHARED = BoundedWorker()
atexit.register(_SHARED.close)


def grade_response(level, domain, row, text):
    """Return a structured status; infrastructure errors are never wrong answers."""
    if isinstance(text, str) and len(text) > MAX_RESPONSE_CHARS:
        return result('resource_limit', detail='response character limit exceeded')
    if not isinstance(row, Mapping) or 'answer' not in row:
        return result('invalid_reference', detail='row requires answer')
    return _SHARED.request({'operation': 'grade', 'level': level, 'domain': domain,
                            'answer': row['answer'], 'text': text})


def reference_check(level, domain, answer):
    from .validation import InputError
    diagnostic = _SHARED.request({'operation': 'reference', 'level': level, 'domain': domain, 'answer': answer})
    if diagnostic['status'] == 'invalid_reference':
        raise InputError(diagnostic['detail'])
    if diagnostic['status'] in UNSCORABLE:
        raise VerifierExecutionError(diagnostic)
    reference = diagnostic.get('reference')
    if not isinstance(reference, dict):
        raise VerifierExecutionError(result('worker_failure', detail='missing worker reference'))
    return reference


def close_worker():
    """Reap the shared verifier process; the next request starts a fresh worker."""
    _SHARED.close()
