"""Prompt templates and response-token mask helpers."""

from __future__ import annotations

from collections.abc import Callable, Sequence


# Chat surfaces keep one benchmark contract portable across model families.
# A surface supplies only the role markers; the system instruction and the user
# body below are shared verbatim, so a Qwen prompt and its Falcon counterpart
# differ in nothing but the tokens the base model was instruction-tuned on.
# The Falcon markers reproduce tiiuae/Falcon3-*-Instruct's published chat
# template byte for byte (``<|system|>\n...\n<|user|>\n...\n<|assistant|>\n``);
# unlike Qwen's single-token ``<|im_start|>`` they encode as ordinary tokens,
# which costs a few prompt tokens but leaves the response surface untouched.
CHAT_SURFACES: dict[str, tuple[str, str, str]] = {
    "qwen": (
        "<|im_start|>system\n",
        "<|im_end|>\n<|im_start|>user\n",
        "<|im_end|>\n<|im_start|>assistant\n",
    ),
    "falcon": ("<|system|>\n", "\n<|user|>\n", "\n<|assistant|>\n"),
}


def render_chat_prompt(surface: str, system: str, user: str) -> str:
    """Render one system/user turn onto a named chat surface."""

    system_marker, user_marker, assistant_marker = CHAT_SURFACES[surface]
    return system_marker + system + user_marker + user + assistant_marker


_MATH_SYSTEM = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


def apply_qwen_math_template(question: str) -> str:
    return render_chat_prompt("qwen", _MATH_SYSTEM, question)


def apply_falcon_math_template(question: str) -> str:
    return render_chat_prompt("falcon", _MATH_SYSTEM, question)


def apply_qwen_math_route_json_v1_template(question: str) -> str:
    """Reproduce the frozen JSON-v1 Gate 1 prompt exactly."""

    return (
        "<|im_start|>system\n"
        "Solve the problem step by step and put the final answer within "
        "\\boxed{}. After the boxed answer, also emit exactly one compact "
        "<route>JSON</route> block when the computation fits this schema: "
        '{"version":"math-route-v1","steps":[...],"final":"sN"}. '
        "Step IDs are contiguous s1,s2,... . A source step is "
        '{"id":"s1","op":"source","value":"a number copied from the problem"}. '
        "Other steps contain only id, op, and earlier args; never write a "
        "result field. Allowed ops are neg, abs, square, cube, sqrt, factorial, "
        "percent, add, mul, sub, div, pow, mod, choose, permute, gcd, lcm, min, "
        "max, and average. Every step must contribute to final. If this exact "
        "numeric trace language cannot express the solution, omit the route "
        "block but still give the boxed answer."
        "<|im_end|>\n<|im_start|>user\n"
        + question
        + "<|im_end|>\n<|im_start|>assistant\n"
    )


_MATH_ROUTE_SYSTEM = (
        "Solve the problem step by step and put the final answer within "
        "\\boxed{}. When the answer is computed from numbers in the problem, "
        "end with exactly one compact reverse-Polish program between "
        "<route> and </route>. Start the program with v2. Write copied input "
        "numbers and operation words separated only by spaces. For example, "
        "'<route>v2 2 5 mul 1 sub square 1 add</route>' computes "
        "(2*5-1)^2+1. Unary operations are neg abs square cube sqrt factorial "
        "percent. Binary operations are add mul sub div pow mod choose permute "
        "gcd lcm min max average. A binary operation consumes the previous two "
        "values in left-to-right order. Copy every numeric input from the "
        "problem; never put a computed result, JSON, equations, labels, or "
        "prose inside <route>. If this numeric stack language cannot express "
        "the solution, omit <route> but still give the boxed answer."
)


def apply_qwen_math_route_template(question: str) -> str:
    """Request a boxed answer plus a compact executable numeric route."""

    return render_chat_prompt("qwen", _MATH_ROUTE_SYSTEM, question)


def apply_falcon_math_route_template(question: str) -> str:
    """Falcon-surface twin of :func:`apply_qwen_math_route_template`."""

    return render_chat_prompt("falcon", _MATH_ROUTE_SYSTEM, question)


_BOXED_SYSTEM = "Return only the final answer inside \\boxed{}. Do not explain."


def apply_qwen_boxed_template(question: str) -> str:
    return render_chat_prompt("qwen", _BOXED_SYSTEM, question)


def apply_falcon_boxed_template(question: str) -> str:
    return render_chat_prompt("falcon", _BOXED_SYSTEM, question)


_LEVEL2_HYBRID_SYSTEMS = {
    "countdown": (
        "Solve the executable constraint problem carefully. You may reason briefly, "
        "but end with exactly one final answer inside \\boxed{}. Systematically combine "
        "every supplied number exactly once using +, -, *, /, and parentheses. Check "
        "the exact target before answering. Output exactly the boxed expression."
    ),
    "python_factors": (
        "Solve the executable constraint problem carefully. You may reason briefly, "
        "but end with exactly one final answer inside \\boxed{}. Construct one allowed "
        "lambda expression. Test small divisors with nested conditional expressions, "
        "for example 2 if n % 2 == 0 else 3 if n % 3 == 0 else 5, but adapt the "
        "tests to every listed case. You may instead dispatch on each listed value. "
        "Output exactly the boxed lambda."
    ),
    "mathir": (
        "Solve the executable constraint problem carefully. You may reason briefly, "
        "but end with exactly one final answer inside \\boxed{}. Use algebraic isolation: "
        "move the right-side x term left, remove the left constant, then divide by the "
        "combined coefficient. Match those operations to the shuffled menu IDs."
    ),
    "pantry": (
        "Solve the executable constraint problem carefully. You may reason briefly, "
        "but end with exactly one final answer inside \\boxed{}. Prefer allowed "
        "high-energy/protein, very-low-sodium ingredients, especially seeds or oats. "
        "Choose stepped amounts, check every bound, and output 2 to 4 "
        "ingredient_id=grams pairs."
    ),
}


def apply_qwen_level2_countdown_template(question: str) -> str:
    return render_chat_prompt("qwen", _LEVEL2_HYBRID_SYSTEMS["countdown"], question)


def apply_qwen_level2_python_factors_template(question: str) -> str:
    return render_chat_prompt("qwen", _LEVEL2_HYBRID_SYSTEMS["python_factors"], question)



# Prospective Python Level 3 default, chosen after the September 11 prompt
# sensitivity analysis. Keep the existing Level 2/historical template intact.
_LEVEL3_PYTHON_NEUTRAL_SYSTEM = (
    "Solve the executable constraint problem carefully. You may reason briefly, "
    "but end with exactly one final answer inside \\boxed{}. Construct one allowed "
    "lambda expression. Output exactly the boxed lambda."
)


def apply_qwen_level3_python_factors_template(question: str) -> str:
    return render_chat_prompt("qwen", _LEVEL3_PYTHON_NEUTRAL_SYSTEM, question)


def apply_qwen_level2_mathir_template(question: str) -> str:
    return render_chat_prompt("qwen", _LEVEL2_HYBRID_SYSTEMS["mathir"], question)


def apply_qwen_level2_pantry_template(question: str) -> str:
    return render_chat_prompt("qwen", _LEVEL2_HYBRID_SYSTEMS["pantry"], question)


_GRAPH_DIGITS_SYSTEM = (
    "Return only the requested bare sequence of digits. Do not explain, "
    "add punctuation, or use LaTeX."
)


def _graph_digits_user(question: str, template_name: str) -> str:
    boxed_suffix = "inside \\boxed{}."
    if question.count(boxed_suffix) != 1 or not question.endswith(boxed_suffix):
        raise ValueError(
            f"{template_name} requires a graph prompt ending in a boxed answer"
        )
    return (
        question[: -len(boxed_suffix)]
        + "as one bare digit string with no spaces, punctuation, or other text."
    )


def apply_qwen_graph_digits_template(question: str) -> str:
    """Request the benchmark's canonical bare graph-color action vector."""

    return render_chat_prompt(
        "qwen",
        _GRAPH_DIGITS_SYSTEM,
        _graph_digits_user(question, "qwen_graph_digits"),
    )


def apply_falcon_graph_digits_template(question: str) -> str:
    """Falcon-surface twin of :func:`apply_qwen_graph_digits_template`."""

    return render_chat_prompt(
        "falcon",
        _GRAPH_DIGITS_SYSTEM,
        _graph_digits_user(question, "falcon_graph_digits"),
    )


_COUNTDOWN_DIGITS_SYSTEM = (
    "Return only three digits and no other text. The given numbers are "
    "n1,n2,n3 in dataset order. Digit 2 selects singleton s: 1=n1, "
    "2=n2, 3=n3; a,b are the other two numbers in their original order. "
    "Digit 3 makes pair: 1=a+b, 2=a*b, 3=a-b, 4=b-a, 5=a/b, 6=b/a. "
    "Digit 1 combines pair and s: 1=pair+s, 2=pair*s, 3=pair-s, "
    "4=s-pair, 5=pair/s, 6=s/pair."
)


def _countdown_digits_user(question: str, template_name: str) -> str:
    boxed_suffix = "the expression inside \\boxed{}."
    if question.count(boxed_suffix) != 1 or not question.endswith(boxed_suffix):
        raise ValueError(
            f"{template_name} requires an easy3 Countdown prompt ending "
            "in the boxed-expression instruction"
        )
    return (
        question[: -len(boxed_suffix)]
        + "a three-digit canonical action code using the scheme above."
    )


def apply_qwen_countdown_digits_template(question: str) -> str:
    """Request the audited three-digit Countdown expression code."""

    return render_chat_prompt(
        "qwen",
        _COUNTDOWN_DIGITS_SYSTEM,
        _countdown_digits_user(question, "qwen_countdown_digits"),
    )


def apply_falcon_countdown_digits_template(question: str) -> str:
    """Falcon-surface twin of :func:`apply_qwen_countdown_digits_template`."""

    return render_chat_prompt(
        "falcon",
        _COUNTDOWN_DIGITS_SYSTEM,
        _countdown_digits_user(question, "falcon_countdown_digits"),
    )


_PANTRY_SUPPORT_MASK_SYSTEM = (
    "Return only six binary digits and no other text. Each digit maps to "
    "the corresponding Pantry row in printed order. A trusted environment "
    "will choose exact quantities on precisely the selected rows."
)


def _pantry_support_mask_user(question: str, template_name: str) -> str:
    suffix = (
        "Return only ingredient_id=grams pairs separated by semicolons inside "
        "\\boxed{}. Do not add a recipe name or preparation prose."
    )
    if question.count(suffix) != 1 or not question.endswith(suffix):
        raise ValueError(
            f"{template_name} requires the frozen PantryPlan answer suffix"
        )
    return (
        question[: -len(suffix)]
        + "Choose the ingredient support as a six-bit mask in the exact Pantry "
        "row order. Bit 1 includes that row and bit 0 excludes it."
    )


def apply_qwen_pantry_support_mask_template(question: str) -> str:
    """Request a six-bit ingredient-support action in printed pantry order."""

    return render_chat_prompt(
        "qwen",
        _PANTRY_SUPPORT_MASK_SYSTEM,
        _pantry_support_mask_user(question, "qwen_pantry_support_mask"),
    )


def apply_falcon_pantry_support_mask_template(question: str) -> str:
    """Falcon-surface twin of :func:`apply_qwen_pantry_support_mask_template`."""

    return render_chat_prompt(
        "falcon",
        _PANTRY_SUPPORT_MASK_SYSTEM,
        _pantry_support_mask_user(question, "falcon_pantry_support_mask"),
    )


# Every canonical-action task admits exactly one prompt template per chat
# surface.  Binding the task to that closed set keeps a Falcon run from
# silently training against Qwen role markers the base model never saw.
CANONICAL_TASK_PROMPT_TEMPLATES: dict[str, frozenset[str]] = {
    "graph_coloring": frozenset({"qwen_graph_digits", "falcon_graph_digits"}),
    "countdown": frozenset(
        {"qwen_countdown_digits", "falcon_countdown_digits"}
    ),
    "pantry_support_mask": frozenset(
        {"qwen_pantry_support_mask", "falcon_pantry_support_mask"}
    ),
}


def validate_canonical_prompt_materialization(
    prompt_template: str,
    raw_questions: Sequence[str],
    formatted_prompts: Sequence[str],
) -> None:
    """Fail closed if dataset mapping did not render the canonical template.

    The exact string comparison deliberately detects stale Hugging Face map
    caches from older prompt templates as well as dropped or reordered rows.
    Re-applying ``TEMPLATE_FACTORY[prompt_template]`` rather than a hardcoded
    function makes the check family-agnostic: a Falcon-surface run is held to
    its own template and cannot pass on Qwen-rendered rows.
    """

    if len(raw_questions) != len(formatted_prompts):
        raise RuntimeError(
            "canonical prompt materialization changed the row count: "
            f"raw={len(raw_questions)} formatted={len(formatted_prompts)}"
        )
    apply_template = TEMPLATE_FACTORY[prompt_template]
    for row_index, (question, observed) in enumerate(
        zip(raw_questions, formatted_prompts)
    ):
        if observed != apply_template(question):
            raise RuntimeError(
                f"canonical {prompt_template} prompt materialization mismatch "
                f"at row {row_index}; stale or foreign template cache detected"
            )


def validate_qwen_graph_digits_materialization(
    raw_questions: Sequence[str], formatted_prompts: Sequence[str]
) -> None:
    """Fail closed on stale or foreign canonical graph prompt rows."""

    validate_canonical_prompt_materialization(
        "qwen_graph_digits", raw_questions, formatted_prompts
    )


def validate_qwen_countdown_digits_materialization(
    raw_questions: Sequence[str], formatted_prompts: Sequence[str]
) -> None:
    """Fail closed on stale or foreign canonical Countdown prompt rows."""

    validate_canonical_prompt_materialization(
        "qwen_countdown_digits", raw_questions, formatted_prompts
    )


def validate_qwen_pantry_support_mask_materialization(
    raw_questions: Sequence[str], formatted_prompts: Sequence[str]
) -> None:
    """Fail closed on stale or foreign Pantry support-mask prompt rows."""

    validate_canonical_prompt_materialization(
        "qwen_pantry_support_mask", raw_questions, formatted_prompts
    )


def apply_r1_template(question: str) -> str:
    return (
        "A conversation between User and Assistant. The User asks a question, "
        "and the Assistant solves it. The Assistant first thinks about the "
        "reasoning process in the mind and then provides the User with the "
        "answer. The reasoning process is enclosed within <think> </think> "
        "and answer is enclosed within <answer> </answer> tags, respectively, "
        "i.e., <think> reasoning process here </think> <answer> answer here "
        "</answer>.\nUser: " + question + "\nAssistant: <think>"
    )


def apply_no_template(question: str) -> str:
    return question


TEMPLATE_FACTORY: dict[str, Callable[[str], str]] = {
    "qwen_boxed": apply_qwen_boxed_template,
    "qwen_countdown_digits": apply_qwen_countdown_digits_template,
    "qwen_graph_digits": apply_qwen_graph_digits_template,
    "qwen_pantry_support_mask": apply_qwen_pantry_support_mask_template,
    "qwen_math": apply_qwen_math_template,
    "qwen_math_route": apply_qwen_math_route_template,
    "qwen_level2_countdown": apply_qwen_level2_countdown_template,
    "qwen_level2_python_factors": apply_qwen_level2_python_factors_template,
    "qwen_level3_python_factors": apply_qwen_level3_python_factors_template,
    "qwen_level3_python_factors_neutral_v1": apply_qwen_level3_python_factors_template,
    "qwen_level2_mathir": apply_qwen_level2_mathir_template,
    "qwen_level2_pantry": apply_qwen_level2_pantry_template,
    "falcon_boxed": apply_falcon_boxed_template,
    "falcon_countdown_digits": apply_falcon_countdown_digits_template,
    "falcon_graph_digits": apply_falcon_graph_digits_template,
    "falcon_pantry_support_mask": apply_falcon_pantry_support_mask_template,
    "falcon_math": apply_falcon_math_template,
    "falcon_math_route": apply_falcon_math_route_template,
    "r1": apply_r1_template,
    "no": apply_no_template,
}

# The Qwen and Falcon families must stay a pure surface swap of one contract.
PROMPT_TEMPLATE_FAMILY_TWINS: dict[str, str] = {
    "qwen_boxed": "falcon_boxed",
    "qwen_countdown_digits": "falcon_countdown_digits",
    "qwen_graph_digits": "falcon_graph_digits",
    "qwen_pantry_support_mask": "falcon_pantry_support_mask",
    "qwen_math": "falcon_math",
    "qwen_math_route": "falcon_math_route",
}

# A template's *role* is the training contract it encodes; the chat surface is
# an implementation detail of the base model. Argument validation keys off the
# role so a Falcon run is held to exactly the Qwen run's contract.
PROMPT_TEMPLATE_ROLES: dict[str, str] = {
    "qwen_boxed": "boxed",
    "falcon_boxed": "boxed",
    "qwen_countdown_digits": "countdown_digits",
    "falcon_countdown_digits": "countdown_digits",
    "qwen_graph_digits": "graph_digits",
    "falcon_graph_digits": "graph_digits",
    "qwen_pantry_support_mask": "pantry_support_mask",
    "falcon_pantry_support_mask": "pantry_support_mask",
    "qwen_math": "math",
    "falcon_math": "math",
    "qwen_math_route": "math_route",
    "qwen_level2_countdown": "boxed",
    "qwen_level2_python_factors": "boxed",
    "qwen_level3_python_factors": "boxed",
    "qwen_level3_python_factors_neutral_v1": "boxed",
    "qwen_level2_mathir": "boxed",
    "qwen_level2_pantry": "boxed",
    "falcon_math_route": "math_route",
    "r1": "r1",
    "no": "no",
}

CANONICAL_DIGIT_TEMPLATE_ROLES: frozenset[str] = frozenset(
    {"graph_digits", "countdown_digits", "pantry_support_mask"}
)


def prompt_template_role(prompt_template: str) -> str:
    """Return the surface-independent training contract a template encodes."""

    return PROMPT_TEMPLATE_ROLES[prompt_template]


def prompt_template_surface(prompt_template: str) -> str:
    """Return the chat surface family a template renders onto."""

    for surface in CHAT_SURFACES:
        if prompt_template.startswith(f"{surface}_"):
            return surface
    return "none"


def apply_prompt_template_to_example(
    example: dict,
    *,
    input_key: str,
    prompt_template: str,
) -> dict:
    """Apply a configured prompt template to one dataset row."""

    problem = example[input_key]
    example[input_key] = TEMPLATE_FACTORY[prompt_template](problem)
    return example


def collate_eval_prompt_items(
    item_list: list[dict],
    *,
    prompt_template: str,
) -> tuple[list[str], list[str], list[str]]:
    """Collate eval rows into templated prompts, raw problems, and answers."""

    problems = []
    formatted_problems = []
    answers = []
    for item in item_list:
        problems.append(item["problem"])
        formatted_problems.append(TEMPLATE_FACTORY[prompt_template](item["problem"]))
        answers.append(item["answer"])
    return formatted_problems, problems, answers


def build_response_token_prefix_mask(
    response_masks: torch.Tensor,
    token_counts: torch.Tensor,
) -> torch.Tensor:
    """Select the first ``token_counts`` response tokens in each row."""

    import torch
    if response_masks.ndim != 2:
        raise ValueError("response_masks must have shape [batch, seq].")
    if token_counts.ndim != 1 or token_counts.shape[0] != response_masks.shape[0]:
        raise ValueError(
            "token_counts must have shape [batch] matching response_masks."
        )
    safe_counts = token_counts.to(
        device=response_masks.device, dtype=torch.int64
    ).clamp(min=0)
    response_positions = response_masks.to(torch.int64).cumsum(dim=1)
    return response_masks.to(torch.bool) & (response_positions <= safe_counts[:, None])
