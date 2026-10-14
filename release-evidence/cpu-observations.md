# CPU candidate validation — 2026-10-14

The complete suite passed: **187 passed, 14 skipped, 1 warning**, in 102.20 seconds. All skipped tests are opt-in network source reads. This includes 85 adversarial artifact-gate cases. No weights, GPU, raw evaluation partitions or external source downloads were used.

Command from the source root: `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q -p no:cacheprovider`.

Tests imported this public source copy using Python 3.13 and the existing reference dependencies documented in `requirements-tested.txt`. This validates CPU behavior, not complete installation into a fresh environment. The latter remains a separate blocked condition. The warning is an existing conversion of a gradient-bearing gate tensor to a scalar for logging in `kev/loop.py`.

Seven figure redraws also passed using an external output directory; original table-value assertions passed. They used only the packaged aggregate results and do not establish full row-based figure reproduction.

## Resource-link documentation update

The project website, paper section, organization repository and private model collection links were added after the CPU run. All 119 existing non-documentation files (excluding the integrity inventory and readiness control files) match the previously tested candidate byte for byte, as recorded in `cpu-tested-files.json`. Research code, tests, numerical inputs, dependency declarations and gate implementation are unchanged. The prior CPU evidence is retained on this basis and rebound to the updated artifact; the 187-test suite was not rerun for these documentation changes. Artifact hygiene, resource-link consistency, archive contents and the default publication block were rechecked.
