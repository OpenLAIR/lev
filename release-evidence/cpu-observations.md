# Fresh CPU installation and validation — 2026-10-14

A new Python 3.13.14 virtual environment with user and system site packages disabled installed the official `torch==2.8.0+cpu` wheel, followed by the complete source package with its `dev` dependencies. Both installs used `--no-cache-dir`; `pip check` reported no broken requirements. The installed PyTorch reported `torch.version.cuda is None`, and the package inventory contains no NVIDIA runtime packages. Exact installed versions are recorded in `clean-install-packages.json`.

The complete suite then passed: **187 passed, 14 skipped, 1 warning**, in 21.28 seconds. All skipped tests are opt-in network source reads. This includes 85 adversarial artifact-gate cases. No pretrained weights, GPU, raw evaluation partitions or external source datasets were used.

Command from the isolated working-copy root: `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q -p no:cacheprovider`. The working copy contained the same research code, tests, metadata inputs and dependency declarations as this candidate; `cpu-tested-files.json` records their hashes. The warning is an existing conversion of a gradient-bearing gate tensor to a scalar for logging in `kev/loop.py`.

Seven figure redraws also passed in the new environment with outputs outside the source snapshot. Original table-value assertions passed. They used packaged aggregate results and do not establish full row-based figure reproduction.

The earlier hosted CPU installation exhausted disk space while installing the default CUDA-enabled PyTorch dependencies, before tests began. The CPU workflow now selects the official CPU-only wheel explicitly, checks dependencies and verifies that a CUDA wheel was not substituted. Research code, tests and release-gate requirements were unchanged. This fresh installation and test run supersedes the earlier CPU result obtained with existing dependencies. Hosted CI will rerun after the candidate is pushed; no hosted success is implied by this local validation. GPU parity and optional CUDA kernels remain separately unvalidated.
