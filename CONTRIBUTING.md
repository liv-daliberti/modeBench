# Contributing

Install the development dependencies and run the maintained checks:

```sh
python -m pip install -e '.[dev]'
make check
```

Use [GitHub issues](https://github.com/liv-daliberti/modeBench/issues) for bugs or protocol questions. Include the repository commit, Python version, domain/level, a minimal response and reference specification, and expected versus observed verification behavior. Avoid including credentials or private model data in examples.

## Scientific compatibility

A validator, prompt, or canonicalization change can alter the benchmark. Explain whether a change affects accepted responses, canonical mode identity, prompt wording, split membership, or metric aggregation. Add focused behavioral regression coverage for those changes.

Keep frozen Parquet splits and retained evidence immutable. Give newly generated datasets or changed prompt conditions separate version identities. Never update expected numerical evidence merely to make a failing check pass.

The core benchmark should remain usable without PyTorch or a training framework. Keep optional dataset tools behind the `data` extra and optimization code in the separate methods project. Preserve the external Python execution boundary.

For a pull request, describe the concrete behavior change, scientific implications, and checks performed. Preserve source copyright notices and dataset attribution.

## Release metadata

`RELEASE_MANIFEST.json` records the reviewed release files, not transient caches. After deliberate source or documentation edits, a maintainer can refresh its entries and the final hashes in `PROVENANCE.json`:

```sh
python ops/verify_release.py --refresh
python ops/verify_release.py
```

Review the manifest diff before committing. Refreshing hashes records changed bytes; it does not establish that the changes preserve benchmark behavior. Tests and frozen evidence checks remain separate requirements.
