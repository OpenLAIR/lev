# Candidate hygiene and release readiness

This is the research source release for [OpenLAIR/lev](https://github.com/OpenLAIR/lev). Companion seed-0 model repositories, three Dataset repositories and their collection are public on Hugging Face; [project resources](RESOURCES.md) link the website, paper, source, models and datasets. The Dataset repositories host rights-filtered subsets with explicit source coverage, licenses and checksums; full-corpus rights and complete reconstruction remain open, and full-size GPU parity has not been re-run on the released packages. The conditions below are the checks `scripts/verify_release.py` enforces for a formal release tag.

## Artifact hygiene

`python scripts/verify_release.py --mode hygiene` verifies the exact `RELEASE_FILES.json` inventory and `MANIFEST.sha256`, paths, file types, UTF-8 content, public metadata and a mandatory Gitleaks 8.30.1 scan. Install PyYAML for workflow validation. No file or directory is ignored: generated caches, build products, environments, weights, downloads, symlinks, special files and unlisted empty directories cause failure. Inline scanner-ignore comments are disabled. The narrow checksum exception in `.gitleaks.toml` cannot be broadened without rejection.

Check a pristine source snapshot. For a Git checkout, create a snapshot outside the checkout using `git archive HEAD | tar -x -C <empty-directory>` and pass `--root <empty-directory>`. Run with `PYTHONDONTWRITEBYTECODE=1`. Install/build in a different working copy, not the snapshot. Validation never silently cleans or changes the input.

## Publication readiness

The default `python scripts/verify_release.py` additionally enforces `RELEASE_READINESS.json`. Every required condition must be passed with evidence bound to the current artifact: fresh dependency installation, CPU tests, full-size GPU parity, weight identities, data rights, data reconstruction, all-figure reproduction, public endpoints and author review. Current results and unresolved conditions are recorded in RELEASE_READINESS.json. Tiny model tests and aggregate figure redraws do not satisfy missing GPU or full reconstruction checks.

Evidence belongs in an inventoried `release-evidence/` file. Evidence must use the strict JSON schema in `scripts/verify_release.py`: `schema_version`, matching `check_id`, `status=passed`, and all check-specific `results` fields. Plain text, skipped/unknown results, incomplete metrics and cross-check evidence reuse fail. Each passed condition supplies its file hash and the artifact's `subject_sha256`. The subject covers all payload files including evidence, excluding only the inventory, integrity manifest and readiness JSON. Evidence records must not embed their own subject digest: first finalize the evidence files, then calculate the subject and write it into readiness. Changing any payload invalidates previous readiness bindings. A blocked condition has a nonempty reason and null evidence/digest. The distribution must also explicitly be `release`.

`python scripts/package_release.py --out ../lev.zip` refuses to package while any condition is blocked. Explicit `--candidate` permits an unpublished archive after hygiene passes; its root is `lev-candidate/` and its archive comment identifies the candidate. Candidate packaging is not release approval. CI runs hygiene, fresh installation and CPU checks on every change, and enforces readiness for release tags or an explicit release-check dispatch. CI does not publish.

Seven of thirteen study figures can be redrawn from aggregate inputs. Full row-based reproduction, final weight validation, per-source permissions and complete data acquisition remain pending. Online adaptive hidden-state continuation is outside the implemented serving interfaces.
