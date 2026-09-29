# Copyright 2025 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Domain parsing, execution, and canonical identities."""
from __future__ import annotations
import ast
import re
from collections import Counter
from fractions import Fraction
from typing import Any

def _parse_graph_coloring_answer(candidate: str, n: int) -> list[int] | None:
    text = str(candidate).strip().lower()
    text = re.sub(r"^(?:coloring|answer)\s*(?:is|=|:)\s*", "", text).strip()
    compact = re.sub(r"[\s,;|\[\]\(\)\{\}:.-]+", "", text)
    if len(compact) == n and all(ch in "123" for ch in compact):
        return [int(ch) for ch in compact]
    tokens = re.findall(r"[123]", text)
    if len(tokens) == n:
        return [int(token) for token in tokens]
    return None


def _parse_graph_digit_sequence(candidate: str, expected_len: int) -> list[int] | None:
    text = str(candidate).strip().lower()
    text = re.sub(
        r"^(?:missing\s+)?(?:colors?|digits?|answer)\s*(?:are|is|=|:)\s*",
        "",
        text,
    ).strip()
    compact = re.sub(r"[\s,;|\[\]\(\)\{\}:.-]+", "", text)
    if len(compact) == expected_len and all(ch in "123" for ch in compact):
        return [int(ch) for ch in compact]
    tokens = re.findall(r"[123]", text)
    if len(tokens) == expected_len:
        return [int(token) for token in tokens]
    return None


def _verify_graph_coloring_colors(
    colors: list[int] | None,
    spec: dict[str, Any],
) -> bool:
    """Validate one already-parsed coloring object against its problem."""

    try:
        n = int(spec["n"])
        edges = spec["edges"]
    except Exception:
        return False
    if colors is None or len(colors) != n:
        return False
    partial_colors = spec.get("partial_colors")
    if partial_colors is not None:
        try:
            if len(partial_colors) != n:
                return False
            for index, color in enumerate(partial_colors):
                if color is None:
                    continue
                if colors[index] != int(color):
                    return False
        except Exception:
            return False
    for edge in edges:
        try:
            u, v = int(edge[0]), int(edge[1])
        except Exception:
            return False
        if u < 1 or v < 1 or u > n or v > n:
            return False
        if colors[u - 1] == colors[v - 1]:
            return False
    return True


def _verify_graph_coloring_answer(candidate: str, spec: dict[str, Any]) -> bool:
    colors = _graph_coloring_from_candidate(candidate, spec)
    return _verify_graph_coloring_colors(colors, spec)


def _graph_coloring_from_candidate(
    candidate: str,
    spec: dict[str, Any],
) -> list[int] | None:
    try:
        n = int(spec["n"])
    except Exception:
        return None
    colors = _parse_graph_coloring_answer(candidate, n)
    partial_colors = spec.get("partial_colors")
    if colors is None and partial_colors is not None:
        try:
            hidden_positions = [
                index for index, color in enumerate(partial_colors) if color is None
            ]
            fill = _parse_graph_digit_sequence(candidate, len(hidden_positions))
            if fill is not None:
                colors = [
                    int(color) if color is not None else 0 for color in partial_colors
                ]
                for index, color in zip(hidden_positions, fill):
                    colors[index] = color
        except Exception:
            return None
    if colors is None or len(colors) != n:
        return None
    return colors



def validate_reference(spec, level):
    """Validate this domain reference before any response is scored."""
    from modebench.validation import InputError, positive_integer
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
