---
pretty_name: LevBench
language:
- en
task_categories:
- text-classification
---
# LevBench

[Project website](https://openlair.github.io/lev/) · [Paper (arXiv)](https://arxiv.org/abs/2610.20951) · [PDF](https://arxiv.org/pdf/2610.20951) · [GitHub](https://github.com/OpenLAIR/lev) · [Hugging Face collection](https://huggingface.co/collections/songdj/lev)

[Resource availability](../docs/RESOURCES.md)

Hub repository: [songdj/LevBench](https://huggingface.co/datasets/songdj/LevBench). No question text is packaged or uploaded by this release candidate. LevBench corresponds to frozen eval-v1: 4,750 development records (4,870 questions) and 9,500 locked-test records (9,740 questions), organized into id, near, deep, far and unknowable strata. A record may contain multiple questions; split and bootstrap units must retain group identity.

Sources, revisions and historical partition hashes are in `evals/levbench-v1/manifest.json` and `data/sources.json`. Generated tasks vary rule depth, dependency-chain length, ledger steps and date-chain length. Source/template/difficulty relations to training are explicit. StrategyQA is report-only; missing-information items have no determinate answer and must not be mixed into ordinary answerable accuracy.

The staged subset hosts 3,570 development and 7,140 test records under reviewed source-specific conditions; withheld sources remain explicit in the full ordering recipe. There is no blanket Apache license for the combined dataset. Preserve source attribution, share-alike and noncommercial conditions where they apply, and do not redistribute unclear/rightsholder-restricted sources based only on a mirror's metadata. A permissive, restricted-research, or build-only subset must have its own complete source manifest, notices and new file hashes; subsets are not the original frozen corpus.

The locked test was evaluated once for the study. This preparation does not reopen it. Releasing its metadata or reporting its fixed aggregate results does not create a fresh holdout for tuning.


The Hugging Face Dataset card is the current source of package coverage, missing materials, source licenses and immutable file identities. The code candidate contains metadata only; source-specific hosted subsets have new hashes and do not replace the full original frozen inputs. See [data coverage](../docs/DATA.md#hosted-coverage).
