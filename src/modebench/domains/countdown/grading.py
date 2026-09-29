"""Typed exact-arithmetic grading."""
import ast
import re
from collections import Counter
from fractions import Fraction
from modebench.diagnostics import result


def grade(candidate, spec, text):
    from .verifier import _normalize_countdown_expression, _countdown_eval_and_numbers, _canonical_countdown_ast
    target = Fraction(int(spec['target']), 1)
    numbers = Counter(int(n) for n in spec['numbers'])
    parts = [p.strip() for p in _normalize_countdown_expression(candidate).split('=') if p.strip()]
    evaluable = False
    for part in parts:
        if not re.fullmatch(r'[0-9+\-*/().\s*]+', part):
            continue
        try:
            parsed = ast.parse(part, mode='eval')
            value, used = _countdown_eval_and_numbers(parsed)
        except SyntaxError:
            continue
        except ValueError as error:
            if 'division by zero' in str(error):
                evaluable = True
            continue
        evaluable = True
        if value == target and Counter(used) == numbers:
            return result('correct', key='countdown:' + _canonical_countdown_ast(parsed), text=text)
    return result('incorrect' if evaluable else 'malformed', text=text,
                  detail='arithmetic or operand mismatch' if evaluable else 'invalid expression syntax')
