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


def validate_reference(level, domain, answer):
    """Validate a task reference before any generated answers are scored."""
    from .historical_prompts import _identity
    from .mathir import MATHIR_VERIFIER, MATHIR_MENU_VERIFIER
    from .mathir import _validated_reference, _validated_menu_reference
    from .pantry_plan import parse_pantry_plan_spec
    from .python_modebench import PYTHON_FACTOR_VERIFIER, parse_python_factor_spec

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
        if domain == 'countdown':
            numbers = spec.get('numbers')
            if not isinstance(numbers, list) or not numbers:
                raise InputError('answer.numbers must be a nonempty list of integers')
            for label, value in [('target', spec.get('target'))] + [('numbers', n) for n in numbers]:
                if isinstance(value, bool) or not isinstance(value, int):
                    raise InputError(f'answer.{label} must contain integers')
        elif domain == 'graph_coloring':
            n = positive_integer(spec.get('n'), 'answer.n')
            edges = spec.get('edges')
            if not isinstance(edges, list):
                raise InputError('answer.edges must be a list of vertex pairs')
            for edge in edges:
                if not isinstance(edge, list) or len(edge) != 2:
                    raise InputError('each graph edge must be a pair')
                if any(isinstance(v, bool) or not isinstance(v, int) or not 1 <= v <= n for v in edge):
                    raise InputError('graph edge endpoints must be integers in 1..n')
                if edge[0] == edge[1]:
                    raise InputError('graph edges must not be self-loops')
            colors = spec.get('partial_colors')
            if colors is not None:
                if not isinstance(colors, list) or len(colors) != n:
                    raise InputError('answer.partial_colors must have n entries')
                if any(c is not None and (type(c) is not int or c not in (1, 2, 3)) for c in colors):
                    raise InputError('partial colors must be null or integers 1, 2, 3')
        elif domain == 'python_factors':
            parse_python_factor_spec(spec)
        elif domain == 'pantry_plan':
            parsed = parse_pantry_plan_spec(spec)
            if level == 1 and len(parsed.ingredients) != 6:
                raise InputError('Level 1 Pantry requires exactly six ingredients for its support mask')
        else:
            if 'max_steps' in spec:
                positive_integer(spec['max_steps'], 'answer.max_steps')
            for field in ('initial_lhs', 'initial_rhs'):
                if not isinstance(spec.get(field), str) or not spec[field].strip():
                    raise InputError(f'answer.{field} must be a nonempty string')
            parser = _validated_menu_reference if verifier == MATHIR_MENU_VERIFIER else _validated_reference
            parser(spec)
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
