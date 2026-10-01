ModeBench
=========

**Measure whether a model finds different correct solutions to the same task.**
ModeBench checks correctness with executable verifiers and counts prompt-local
solution modes. It runs on CPU and evaluates saved responses from any model.

.. image:: ../assets/modebench-domains.png
   :alt: The five ModeBench domains, each illustrated with two verified solution modes.

Start with :doc:`quickstart` for an offline example covering all five domains.
Use :doc:`datasets` when you are ready to collect responses on frozen tasks,
then :doc:`evaluation` to score them and :doc:`metrics` to interpret the report.

.. tip::
   This guide targets the published **modebench 0.4.0** package on Linux,
   CPython 3.10–3.12. No PyTorch, CUDA, or model server is needed to evaluate responses.
   For learning from verified modes, see the
   `Re:Max / Re:Dr guide <https://liv-daliberti.github.io/remax/>`_.

.. toctree::
   :maxdepth: 1
   :caption: User guide

   quickstart
   datasets
   evaluation
   metrics
   results
   api
   reference
   contributing

`PyPI <https://pypi.org/project/modebench/0.4.0/>`_ ·
`GitHub <https://github.com/liv-daliberti/modeBench>`_ ·
`Report a problem <https://github.com/liv-daliberti/modeBench/issues>`_
