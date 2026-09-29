"""Worker-only reference validation and dispatch to domain-owned grading."""
from .diagnostics import MAX_RESPONSE_CHARS, result
from .domains.python_factors.grading import grade as python_result
from .domains.countdown.grading import grade as countdown_result


def grade_local(level, domain, answer, text):
    from .validation import _validate_reference_local
    from .historical_prompts import _identity
    from .grading import _extract_modebench_candidate
    from .domains.graph_coloring.grading import grade as graph_grade
    from .domains.mathir.grading import grade as mathir_grade
    from .domains.pantry_plan.grading import grade as pantry_grade
    try:
        level, domain = _identity(level, domain)
        spec = _validate_reference_local(level, domain, answer)
    except (ValueError, TypeError, KeyError) as error:
        return result('invalid_reference', text=text if isinstance(text, str) else '', detail=str(error))
    if not isinstance(text, str):
        return result('malformed', detail='response text must be a string')
    if len(text) > MAX_RESPONSE_CHARS:
        return result('resource_limit', detail='response character limit exceeded')
    if domain == 'pantry_plan' and level == 1:
        return pantry_grade(level, None, spec, text)
    candidate = _extract_modebench_candidate(text, spec)
    if candidate is None:
        return result('malformed', text=text, detail='invalid response surface')
    if domain == 'python_factors':
        diagnostic = python_result(candidate, spec)
        diagnostic['graded_text'] = text
        return diagnostic
    if domain == 'pantry_plan':
        return pantry_grade(level, candidate, spec, text)
    graders = {'countdown': countdown_result, 'graph_coloring': graph_grade, 'mathir': mathir_grade}
    return graders[domain](candidate, spec, text)
