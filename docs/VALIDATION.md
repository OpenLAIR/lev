# Candidate validation

Validation is scoped to a source candidate. The 2026-10-14 complete CPU run passed **187 tests**, with 14 opt-in network tests skipped and one existing tensor-logging warning (102.20 seconds). All 85 adversarial gate tests passed. See `release-evidence/cpu-tests.json` and `release-evidence/cpu-observations.md`. The CPU suite uses tiny random models, converter/generator invariants, analysis identities, result-export and workload-stopper checks. Release-gate adversarial tests exercise explicit inventory, directories, manifests, metadata, content, secrets, malformed files and evidence binding. Network source reads are opt-in and skipped in the normal suite.

Seven figure redraws use existing aggregate data and preserve table-value assertions. They do not recompute raw-row confidence intervals or rerun any locked evaluation. The runner writes into an external output directory through a temporary working copy.

Earlier packaging checks installed a wheel without dependencies in an isolated interpreter and resolved declared dependencies. This does not establish a successful complete dependency installation, CPU test suite in that environment, optional CUDA kernel compatibility, full-size GPU parity or complete data reproduction. These remain separately tracked by `RELEASE_READINESS.json`.

The exact-inventory checker validates every file and directory, rejects unexpected/generated content, checks public metadata schemas and requires the reviewed credential scanner. The only credential exception is one anchored API-Bank checksum line in `data/sources.json`; real credentials elsewhere or extra configuration exemptions must fail. Scientific floating-point values and hashes are preserved even when digit sequences resemble execution identifiers.

Successful candidate checks do not clear publication conditions. Run the commands in RELEASE_STATUS.md against the exact artifact to inspect its current status. No GPU experiment, dataset evaluation, billed API call or publication is part of source assembly.
