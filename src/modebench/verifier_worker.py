"""Bounded JSON-lines worker for reference validation and all five domains."""
import json
import resource
import signal
import sys

from .diagnostics import (DOMAIN_DEADLINE_SECONDS, MAX_REQUEST_BYTES, PYTHON_DEADLINE_SECONDS,
                          WORKER_MEMORY_BYTES, result)


def _timeout(*_args):
    raise TimeoutError('worker execution deadline exceeded')


def handle(request):
    from .verifier_core import grade_local, python_result
    from .validation import _validate_reference_local
    operation = request.get('operation')
    if operation == 'python':
        return python_result(request['candidate'], request['spec'])
    if operation == 'grade':
        return grade_local(request['level'], request['domain'], request['answer'], request['text'])
    if operation == 'reference':
        try:
            reference = _validate_reference_local(request['level'], request['domain'], request['answer'])
        except (ValueError, TypeError, KeyError) as error:
            return result('invalid_reference', detail=str(error))
        return result('correct', reference=reference)
    raise ValueError('unknown worker operation')


def guarded_handle(request):
    seconds = PYTHON_DEADLINE_SECONDS if (request.get('operation') == 'python' or (request.get('operation') == 'grade' and request.get('domain') == 'python_factors')) else DOMAIN_DEADLINE_SECONDS
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        return handle(request)
    except TimeoutError as error:
        return result('timeout', detail=str(error))
    except MemoryError:
        return result('resource_limit', detail='worker memory limit exceeded')
    except Exception as error:
        return result('worker_failure', detail=f'unexpected {type(error).__name__}')
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def main():
    resource.setrlimit(resource.RLIMIT_AS, (WORKER_MEMORY_BYTES, WORKER_MEMORY_BYTES))
    signal.signal(signal.SIGALRM, _timeout)
    while True:
        line = sys.stdin.buffer.readline(MAX_REQUEST_BYTES + 1)
        if not line:
            return
        if len(line) > MAX_REQUEST_BYTES or not line.endswith(b'\n'):
            payload = result('resource_limit', detail='request byte limit exceeded')
        else:
            try:
                request = json.loads(line)
                if not isinstance(request, dict):
                    raise ValueError('request must be an object')
            except (ValueError, UnicodeError):
                payload = result('worker_failure', detail='invalid worker request')
            else:
                payload = guarded_handle(request)
        sys.stdout.write(json.dumps(payload, allow_nan=False) + '\n')
        sys.stdout.flush()
        if payload['status'] in ('timeout', 'resource_limit', 'worker_failure'):
            return


if __name__ == '__main__':
    main()
