"""Typed equation-action grading."""
from modebench.diagnostics import result


def grade(candidate, spec, text):
    from modebench.domains.mathir.verifier import (MathIRError, MATHIR_MENU_VERIFIER, MATHIR_VERSION, MATHIR_MENU_VERSION,
                         _validated_reference, _validated_menu_reference, parse_mathir_program,
                         parse_mathir_action_program, _execute_mathir_commands)
    menu = spec['verifier'] == MATHIR_MENU_VERIFIER
    if menu:
        state, bindings, max_steps, actions = _validated_menu_reference(spec)
    else:
        state, bindings, max_steps = _validated_reference(spec)
    try:
        if menu:
            ids = parse_mathir_action_program(candidate, action_ids=actions, max_steps=max_steps)
            commands = tuple(actions[a] for a in ids)
        else:
            ids = ()
            commands = parse_mathir_program(candidate, allowed_symbols=frozenset(bindings)|{'x'}, max_steps=max_steps)
    except MathIRError as error:
        return result('malformed', text=text, detail=str(error))
    try:
        validation = _execute_mathir_commands(initial_state=state, bindings=bindings, commands=commands,
                                             key_version=MATHIR_MENU_VERSION if menu else MATHIR_VERSION,
                                             action_ids=ids)
    except MathIRError as error:
        return result('incorrect', text=text, detail=str(error))
    return result('correct', key=validation.canonical_key, text=text)
