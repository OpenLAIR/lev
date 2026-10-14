---
pretty_name: Lev-train-v2
language:
- en
task_categories:
- text-classification
---
# Lev-train-v2 — private staging

[Project website](https://openlair.github.io/lev/) · [Paper](https://openlair.github.io/lev/#paper) · [GitHub](https://github.com/OpenLAIR/lev) · [Hugging Face collection — private](https://huggingface.co/collections/songdj/lev)

[Resource availability](../docs/RESOURCES.md)

Private repository: [songdj/Lev-train-v2](https://huggingface.co/datasets/songdj/Lev-train-v2). The source candidate includes metadata and builders, with no training text.

S31 is the headline training recipe: 18,764 original mixture records plus 12,576 decision-v7 replay records = 31,340 before tokenizer/context filtering (31,329 retained for headline Ouro-2.6B). S60 and S120 use 47,424 and 107,424 records in the respective JSONL plus the same replay, totaling 60,000 and 120,000 before filtering. The larger mixtures extend the same S31 corpus; there is no substitute S31 version hidden behind the train-v2 name.

The incremental pool has 89,723 examples and uses stratified deterministic prefixes. See `data/train-v2/manifest.json` for subset composition, source revisions and partition hashes. Do not train on eval-only sources or held-out generated templates. Builders require the original screening references, including the evaluation splits; see docs/DATA.md before attempting reconstruction.

No combined data license is asserted. Review every original source and mirror revision, preserve attribution, and create explicitly versioned redistribution subsets when rights differ. Source-code licensing does not cover dataset text.


The private Dataset card is the current source of package coverage, missing materials, source licenses and immutable file identities. The code candidate contains metadata only; source-specific hosted subsets have new hashes and do not replace the full original frozen inputs. See [data coverage](../docs/DATA.md#private-staging-coverage).
