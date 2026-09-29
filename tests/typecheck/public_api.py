"""Static consumer contract; checked by mypy, never executed as a benchmark."""

from modebench.api import EvaluationOptions, Task, grade, make_prompt, read_report
from modebench.api_types import ChatMessage, GradeResult, Report


def consume(task: Task) -> tuple[list[ChatMessage], GradeResult]:
    messages: list[ChatMessage] = make_prompt(task)
    result: GradeResult = grade(task, "saved response")
    return messages, result


options = EvaluationOptions(min_defined_prompts=30, config="level1_countdown")
report: Report = read_report("run.json")
