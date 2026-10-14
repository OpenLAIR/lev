# Candidate validation

On 2026-10-14, a complete dependency installation into a fresh Python 3.13.14 virtual environment succeeded using the official `torch==2.8.0+cpu` wheel and the source package's `dev` extra. `pip check` passed, PyTorch had no CUDA runtime, and user/system site packages were disabled. See `release-evidence/clean-install.json` and the exact installed version inventory in `release-evidence/clean-install-packages.json`.

The full CPU suite then passed **187 tests**, with 14 opt-in network tests skipped and one existing tensor-logging warning (21.28 seconds). All 85 adversarial release-gate cases passed. The tests cover tiny random models, converter/generator invariants, analysis identities, result export, workload stoppers and artifact validation. `release-evidence/cpu-tests.json`, `cpu-observations.md` and `cpu-tested-files.json` record the result and tested source hashes.

Seven figure redraws succeeded in the same fresh environment using existing aggregate data, preserving table-value assertions. They do not recompute raw-row confidence intervals or rerun locked evaluation. Outputs were generated outside the source artifact.

The initial hosted CPU CI installation failed before tests because the default CUDA-enabled PyTorch dependencies exhausted runner disk space. The workflow now explicitly selects the official CPU-only wheel, disables pip download caching and checks that no CUDA wheel was substituted. The repaired installation and validation commands passed locally in a new environment; hosted CI is rerun after the candidate is pushed.

The exact-inventory checker validates every file and directory, rejects unexpected/generated content, checks public metadata schemas and requires the reviewed credential scanner. The only credential exception is one anchored API-Bank checksum line in `data/sources.json`; real credentials elsewhere or extra configuration exemptions fail. Scientific floating-point values and hashes remain unchanged.

Fresh CPU installation and tests do not establish full-size GPU parity, CUDA kernel compatibility, data redistribution rights or complete row-based figure reproduction. Those conditions remain separately gated by `RELEASE_READINESS.json`. No GPU experiment, dataset evaluation, billed API call or public release is part of these checks.
