"""ModeBench: execution-bound correctness and solution mode evaluation."""
from .grading import validated_modebench_outcome_key
from .metrics import mode_diversity, cell_summary
__version__ = "0.4.0"
__all__ = ["validated_modebench_outcome_key", "mode_diversity", "cell_summary"]

# Compatibility imports for existing ModeBench / Re:Max integrations.
from importlib import import_module as _import_module
import sys as _sys

_LEGACY_MODULES = {
    'historical_prompts': 'modebench.registered_prompts',
    'mathir': 'modebench.domains.mathir.verifier',
    'pantry_plan': 'modebench.domains.pantry_plan.verifier',
    'pantry_support_action': 'modebench.domains.pantry_plan.support',
    'python_modebench': 'modebench.domains.python_factors.verifier',
    'python_modebench_process': 'modebench.domains.python_factors.process',
}
for _old, _new in _LEGACY_MODULES.items():
    _module = _import_module(_new)
    _sys.modules[f"{__name__}.{_old}"] = _module
    globals()[_old] = _module
del _old, _new, _module, _import_module, _sys
