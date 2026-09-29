"""Typed graph-coloring grading."""
from modebench.diagnostics import result


def grade(candidate, spec, text):
    from .verifier import _graph_coloring_from_candidate, _verify_graph_coloring_colors
    colors = _graph_coloring_from_candidate(candidate, spec)
    if colors is None:
        return result('malformed', text=text, detail='invalid coloring surface')
    if not _verify_graph_coloring_colors(colors, spec):
        return result('incorrect', text=text, detail='edge or fixed-color violation')
    return result('correct', key='graph_coloring:' + ''.join(map(str, colors)), text=text)
