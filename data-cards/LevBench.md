---
pretty_name: LevBench
language:
- en
task_categories:
- text-classification
---
# LevBench — release draft

[Project website](https://openlair.github.io/lev/) · [Paper](https://openlair.github.io/lev/#paper) · [GitHub](https://github.com/OpenLAIR/lev) · [Hugging Face collection — private](https://huggingface.co/collections/songdj/lev-thinking-fast-and-deep-with-looped-decision-models-6acefb135880c9d373cf42ba)

[Resource availability](../docs/RESOURCES.md)

Planned repository: `songdj/LevBench`. No question text is packaged or uploaded by this release candidate. LevBench corresponds to frozen eval-v1: 4,750 development records (4,870 questions) and 9,500 locked-test records (9,740 questions), organized into id, near, deep, far and unknowable strata. A record may contain multiple questions; split and bootstrap units must retain group identity.

Sources, revisions and historical partition hashes are in `evals/levbench-v1/manifest.json` and `data/sources.json`. Generated tasks vary rule depth, dependency-chain length, ledger steps and date-chain length. Source/template/difficulty relations to training are explicit. StrategyQA is report-only; missing-information items have no determinate answer and must not be mixed into ordinary answerable accuracy.

License configuration and distribution remain pending a per-source check. There is no blanket Apache license for the combined dataset. Preserve source attribution, share-alike and noncommercial conditions where they apply, and do not redistribute unclear/rightsholder-restricted sources based only on a mirror's metadata. A permissive, restricted-research, or build-only subset must have its own complete source manifest, notices and new file hashes; subsets are not the original frozen corpus.

The locked test was evaluated once for the study. This preparation does not reopen it. Releasing its metadata or reporting its fixed aggregate results does not create a fresh holdout for tuning.
