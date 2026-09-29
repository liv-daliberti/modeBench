"""Killable external execution boundary for ModeBench Python functions."""
import atexit
from modebench.diagnostics import UNSCORABLE, VerifierExecutionError
from modebench.domains.python_factors.verifier import PythonFactorValidation
from modebench.worker_process import BoundedWorker


class PythonFactorVerifierProcess(BoundedWorker):
    """Detailed checks plus a compatible validation/key interface."""
    def check(self, candidate, spec):
        return self.request({'operation': 'python', 'candidate': candidate, 'spec': spec})

    def validate(self, candidate, spec):
        diagnostic = self.check(candidate, spec)
        if diagnostic['status'] in UNSCORABLE:
            raise VerifierExecutionError(diagnostic)
        if not diagnostic['verified']:
            return None
        outputs = diagnostic.get('outputs')
        if not isinstance(outputs, list) or any(type(n) is not int for n in outputs):
            from modebench.diagnostics import result
            self.close()
            raise VerifierExecutionError(result('worker_failure', detail='invalid Python worker outputs'))
        return PythonFactorValidation(diagnostic['canonical_key'], tuple(outputs))


_SHARED_VERIFIER = PythonFactorVerifierProcess()
atexit.register(_SHARED_VERIFIER.close)


def validate_python_factor_function_external(candidate, spec):
    """Return validation/None for answers; raise on verifier failures."""
    return _SHARED_VERIFIER.validate(candidate, spec)
