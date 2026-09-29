"""Typed Python-factor grading."""
from modebench.diagnostics import result


def grade(candidate, spec):
    from modebench.domains.python_factors.verifier import (PythonModeBenchError, execute_python_factor_candidate,
                                   parse_python_factor_candidate, parse_python_factor_spec)
    try:
        parse_python_factor_spec(spec)
    except (ValueError, TypeError, KeyError) as error:
        return result('invalid_reference', detail=str(error))
    if not isinstance(candidate, str):
        return result('malformed', detail='candidate must be a string')
    try:
        parse_python_factor_candidate(candidate)
    except PythonModeBenchError as error:
        return result('malformed', text=candidate, detail=str(error))
    try:
        validation = execute_python_factor_candidate(candidate, spec)
    except (PythonModeBenchError, ArithmeticError) as error:
        return result('incorrect', text=candidate, detail=str(error))
    return result('correct', key=validation.canonical_key, text=candidate, outputs=list(validation.outputs))

