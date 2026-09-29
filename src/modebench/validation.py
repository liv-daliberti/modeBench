"""Input contracts for public evaluation; model answers remain fail-closed."""
from __future__ import annotations

import json
import math
from numbers import Integral


class InputError(ValueError):
    """Invalid evaluation input, distinct from an incorrect model response."""


def positive_integer(value, label):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise InputError(f'{label} must be a positive integer')
    return int(value)


def json_value(value, label='value'):
    """Reject non-JSON values, non-string object keys, and nonfinite numbers."""
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise InputError(f'{label} must not contain NaN or infinity')
    elif isinstance(value, list):
        for item in value:
            json_value(item, label)
    elif isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise InputError(f'{label} object keys must be strings')
            json_value(item, label)
    else:
        raise InputError(f'{label} must contain only JSON values')


def loads(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise InputError(f'duplicate JSON field: {key}')
            result[key] = value
        return result

    try:
        value = json.loads(text, object_pairs_hook=pairs)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise InputError(f'invalid JSON: {error}') from error
    json_value(value)
    return value


def _validate_reference_local(level, domain, answer):
    """Validate a task reference before any generated answers are scored."""
    from .historical_prompts import _identity
    from modebench.domains.mathir.verifier import MATHIR_VERIFIER, MATHIR_MENU_VERIFIER
    from modebench.domains.python_factors.verifier import PYTHON_FACTOR_VERIFIER
    from .domains.countdown.verifier import validate_reference as validate_countdown
    from .domains.graph_coloring.verifier import validate_reference as validate_graph_coloring
    from .domains.python_factors.verifier import validate_reference as validate_python_factors
    from .domains.pantry_plan.verifier import validate_reference as validate_pantry_plan
    from .domains.mathir.verifier import validate_reference as validate_mathir

    from .diagnostics import MAX_REFERENCE_BYTES
    encoded = json.dumps(answer, allow_nan=False)
    if len(encoded.encode()) > MAX_REFERENCE_BYTES:
        raise InputError('reference byte limit exceeded')
    level, domain = _identity(level, domain)
    spec = loads(answer) if isinstance(answer, str) else answer
    if not isinstance(spec, dict):
        raise InputError('answer must be an object or a JSON-encoded object')
    json_value(spec, 'answer')
    allowed = {
        'countdown': {'countdown'},
        'graph_coloring': {'graph_coloring'},
        'mathir': {MATHIR_VERIFIER, MATHIR_MENU_VERIFIER},
        'pantry_plan': {'pantry_plan'},
        'python_factors': {PYTHON_FACTOR_VERIFIER},
    }
    verifier = spec.get('verifier')
    if not isinstance(verifier, str) or verifier not in allowed[domain]:
        raise InputError(f'answer.verifier {verifier!r} does not match domain {domain!r}')
    try:
        validators = {
            'countdown': validate_countdown,
            'graph_coloring': validate_graph_coloring,
            'python_factors': validate_python_factors,
            'pantry_plan': validate_pantry_plan,
            'mathir': validate_mathir,
        }
        validators[domain](spec, level)
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError) as error:
        raise InputError(f'invalid {domain} reference: {error}') from error
    return spec


def validate_run(run):
    """One declared model/generation/prompt condition shared by the entire run."""
    from .prompts import CONDITIONS
    if not isinstance(run, dict):
        raise InputError('run metadata must be an object')
    json_value(run, 'run')
    required = {'model', 'generation', 'prompt_condition'}
    missing = required - run.keys()
    if missing:
        raise InputError(f'run metadata missing fields: {", ".join(sorted(missing))}')
    model = run['model']
    if not isinstance(model, dict):
        raise InputError('run.model must be an object with id and revision')
    for field in ('id', 'revision'):
        if not isinstance(model.get(field), str) or not model[field].strip():
            raise InputError(f'run.model.{field} must be a nonempty string')
    if not isinstance(run['generation'], dict) or not run['generation']:
        raise InputError('run.generation must be a nonempty object of generation settings')
    if 'draws_per_prompt' in run['generation']:
        positive_integer(run['generation']['draws_per_prompt'], 'run.generation.draws_per_prompt')
    if run['prompt_condition'] not in CONDITIONS:
        raise InputError('run.prompt_condition is not a supported prompt condition')
    return loads(json.dumps(run, allow_nan=False))


def validate_reference(level, domain, answer):
    """Validate references in the same bounded worker used for grading."""
    from .verifier import reference_check
    return reference_check(level, domain, answer)
