Citation and maintenance
========================

Citation
--------

The paper is accepted at `MATH-AI 2026 <https://mathai-2026.github.io/>`_ and under
review at ICLR 2027. Cite the paper and record software and dataset identities:

.. literalinclude:: ../README.md
   :language: bibtex
   :start-after: ```bibtex
   :end-before: ```

`CITATION.cff <https://github.com/liv-daliberti/modeBench/blob/main/CITATION.cff>`_
contains machine-readable metadata. Liv G. d'Aliberti is the maintenance and
scientific-review contact.

Licenses
--------

Code is Apache-2.0. The authors' copyrightable dataset contributions are CC BY 4.0;
upstream USDA FoodData Central material remains CC0/public domain. See
`DATA_LICENSE <https://github.com/liv-daliberti/modeBench/blob/main/DATA_LICENSE>`_
and the `source attribution <https://github.com/liv-daliberti/modeBench/blob/main/provenance/licenses/data-sources.json>`_.
Model weights and unrelated third-party software have their own terms.

Contribute a change
-------------------

From a checkout, install ``.[dev]`` and run ``make check``. Explain any change to
benchmark meaning and preserve the frozen conformance corpus and dataset identities.
The conformance corpus has automated independent mathematical checks; external
human review is still outstanding.

Build these guides
------------------

The documentation builds against the published package, not an editable import.
In a Linux Python 3.10–3.12 environment, from the repository root:

.. code-block:: console

   python -m pip install 'modebench[data]==0.4.0' -r docs/requirements.txt
   python -m sphinx -W --keep-going -b doctest docs outputs/docs-doctest
   python -m sphinx -n -W --keep-going -b html docs outputs/docs
   python ops/check_docs.py outputs/docs
   python -m http.server --directory outputs/docs 8000

Open ``http://localhost:8000``. Pull requests build and check the site; main-branch
updates deploy it through GitHub Pages. All guide sources live in ``docs/`` as
reStructuredText, leaving README as the repository's only Markdown file.
