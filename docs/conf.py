"""Build public guides against the released package, without source-path overrides."""

from importlib import metadata
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

from docutils import nodes
from sphinx.util.docutils import SphinxDirective

ROOT = Path(__file__).resolve().parents[1]
project = "ModeBench"
release = "0.4.0"
assert metadata.version("modebench") == release, "Install the documented package version"
version = release
copyright = "2026, Liv G. d'Aliberti and contributors"
extensions = ["sphinx.ext.autodoc", "sphinx.ext.doctest", "sphinx_copybutton"]
html_theme = "furo"
html_title = f"{project} {release}"
html_baseurl = "https://liv-daliberti.github.io/modeBench/"
html_static_path = ["_static"]
html_css_files = ["style.css"]
html_show_sourcelink = False
html_show_sphinx = False
autodoc_typehints = "none"
autodoc_docstring_signature = False
autodoc_member_order = "bysource"
exclude_patterns = ["requirements.txt"]
html_theme_options = {
    "light_css_variables": {"color-brand-primary": "#14635c", "color-brand-content": "#14635c"},
    "dark_css_variables": {"color-brand-primary": "#91b8ff", "color-brand-content": "#91b8ff"},
    "announcement": 'Guides for ModeBench 0.4.0 · <a href="https://pypi.org/project/modebench/0.4.0/">PyPI</a> · <a href="https://github.com/liv-daliberti/modeBench">GitHub</a> · <a href="https://liv-daliberti.github.io/remax/">Companion project</a>',
    "source_repository": "https://github.com/liv-daliberti/modeBench/",
    "source_branch": "main",
    "source_directory": "docs/",
}


class ReadmeTable(SphinxDirective):
    """Render an existing evidence-checked table without a second copy of its scores."""

    required_arguments = 1

    def run(self):
        path = ROOT / "README.md"
        self.env.note_dependency(str(path))
        marker = self.arguments[0]
        content = (
            path.read_text()
            .split(f"<!-- {marker}:start -->", 1)[1]
            .split(f"<!-- {marker}:end -->", 1)[0]
        )
        lines = [
            line.strip().strip("|").split("|")
            for line in content.splitlines()
            if line.startswith("|")
        ]
        rows = [[cell.strip() for cell in row] for row in lines]
        assert len(rows) >= 3 and all(len(row) == len(rows[0]) for row in rows)
        table = nodes.table(classes=["performance-table"])
        group = nodes.tgroup(cols=len(rows[0]))
        for _ in rows[0]:
            group += nodes.colspec(colwidth=1)
        for container, values in [(nodes.thead(), rows[:1]), (nodes.tbody(), rows[2:])]:
            for row in values:
                tr = nodes.row()
                for cell in row:
                    tr += nodes.entry("", nodes.paragraph(text=cell))
                container += tr
            group += container
        table += group
        return [table]


class CommandHelp(SphinxDirective):
    """Capture the published CLI's help in an isolated process outside the checkout."""

    required_arguments = 1
    final_argument_whitespace = True

    def run(self):
        args = shlex.split(self.arguments[0])
        modules = {"modebench": "modebench.cli", "remax": "remax", "remax-run": "remax.launcher"}
        module = modules[args.pop(0)]
        with tempfile.TemporaryDirectory(prefix="docs-cli-") as directory:
            output = subprocess.check_output(
                [sys.executable, "-I", "-m", module, *args, "--help"],
                cwd=directory,
                text=True,
                timeout=30,
            )
        return [nodes.literal_block(output, output, language="text")]


def clean_generated_docstrings(app, what, name, obj, options, lines):
    # Dataclass-generated docstrings repeat the annotated constructor and make
    # Sphinx treat type names as prose cross-references. The signature is retained.
    from dataclasses import is_dataclass

    if (
        isinstance(obj, type)
        and is_dataclass(obj)
        and lines
        and lines[0].startswith(obj.__name__ + "(")
    ):
        lines.clear()
    # This legacy objective is named for comparison, not part of the public API.
    for index, line in enumerate(lines):
        lines[index] = line.replace(
            ":func:`canonical_replay_uniform_loss`", "``canonical_replay_uniform_loss``"
        )


def setup(app):
    app.connect("autodoc-process-docstring", clean_generated_docstrings)
    app.add_directive("readme-table", ReadmeTable)
    app.add_directive("command-help", CommandHelp)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
