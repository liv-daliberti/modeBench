"""Typed allocation grading and registered Level 1 support projection."""
from modebench.diagnostics import result


def grade(level, candidate, spec, text):
    from .verifier import PantryPlanError, parse_pantry_plan_candidate, validate_pantry_plan
    from .support import pantry_support_from_mask, project_pantry_support, PANTRY_SUPPORT_MASK_INVALID
    if level == 1:
        mask = text.strip()
        if len(mask) != 6 or any(bit not in '01' for bit in mask):
            return result('malformed', text=PANTRY_SUPPORT_MASK_INVALID, detail='expected a six-bit support mask')
        support = pantry_support_from_mask(mask, spec)
        validation = project_pantry_support(support, spec) if support is not None else None
        if validation is None:
            return result('incorrect', text=PANTRY_SUPPORT_MASK_INVALID, detail='support has no feasible allocation')
        graded = ';'.join(f'{name}={grams}' for name,grams in validation.allocations_g)
        return result('correct', key=validation.canonical_key, text=graded)
    try:
        parse_pantry_plan_candidate(candidate)
    except PantryPlanError as error:
        return result('malformed', text=text, detail=str(error))
    validation = validate_pantry_plan(candidate, spec)
    return result('correct' if validation else 'incorrect', text=text,
                  key=validation.canonical_key if validation else None,
                  detail=None if validation else 'allocation violates feasibility constraints')
