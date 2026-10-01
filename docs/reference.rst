CLI reference and troubleshooting
=================================

The help below is captured from the installed package during the site build.

Top-level commands
------------------

.. command-help:: modebench

Dataset discovery
-----------------

.. command-help:: modebench datasets

Prepare model requests
----------------------

.. command-help:: modebench prepare

Evaluate saved responses
------------------------

.. command-help:: modebench evaluate

Read a report
-------------

.. command-help:: modebench report

Common problems
---------------

``modebench`` is not found
   Activate the virtual environment, or run ``python -m modebench.cli``.

Parquet loading needs ``datasets``
   Install ``python -m pip install 'modebench[data]==0.4.0'`` in the active environment.

Unknown split
   Use ``eval``, not ``test``, and inspect available splits with ``datasets list``.

Cache identity mismatch or offline miss
   Restore the named corrupt file from the pinned source, or fetch the same split
   once without ``--offline``. Do not change expected hashes to accept different data.

Output already exists
   Read the completed report or choose a fresh output filename. Resume applies to
   interrupted evaluations without a final receipt.

Resume identity mismatch
   Restore the original input, configuration, software and dataset, or begin a
   separately identified evaluation with a new output path.

PCMD is missing or unreportable
   Check eligible verified-response counts and the support threshold; see :doc:`metrics`.

Exit code 3, timeout, or worker failure
   Inspect status/detail diagnostics. Fix the environment or reference problem and
   repeat under an explicit retry policy; do not turn the failure into a model score.

For a bug report, include the package version, command, platform, dataset identity,
and a minimal response example in
`GitHub Issues <https://github.com/liv-daliberti/modeBench/issues>`_.
