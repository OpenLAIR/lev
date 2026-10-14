---
pretty_name: Lev-results
---
# Lev-results

[Project website](https://openlair.github.io/lev/) · [Paper (alphaXiv)](https://www.alphaxiv.org/pdf/2610.lev-looped-decision-models) · [PDF](https://openlair.github.io/lev/assets/lev-paper.pdf) · [GitHub](https://github.com/OpenLAIR/lev) · [Hugging Face collection](https://huggingface.co/collections/songdj/lev)

[Resource availability](../docs/RESOURCES.md)

Hub repository: [songdj/Lev-results](https://huggingface.co/datasets/songdj/Lev-results). This candidate packages aggregate JSON inputs under `results/` and `figures/`, not the full per-question collection. Aggregate figure redraws and re-estimation from row-level observations are different reproducibility levels.

Depth-row schema (version 1): `model`, `panel`, `split`, `depths`, `source_sha256`, and `rows`. Each row retains `id`, `group`, question identifier `question`, `source`, `task`, `type`, `variant`, numeric `label` and/or per-depth `correct`, `state_tokens`, `tokens`, logits as `[depth][option]`, and any `tier`, `difficulty` (`kind`, `k`), `unanswerable`, `report_only`. Question identity is `(id, question)`; bootstrap identity is `group`. No state, question instructions, criteria or option description is included. Model identity, depth/option ordering, inference dtype, temperature status and calibration provenance must accompany the final collection.

`scripts/export_results.py` projects the standard depth-row schema and refuses locked test export. It intentionally does not export JevBench's different per-depth probability files. The Dataset uses a separate hash-allowlisted publication adapter for these files, preserves stored values and option order, and replaces textual identities with stable hashes. Its own manifest and audit cover exported native and JevBench files; the normal code exporter retains its existing restrictions.

The original cost analyses need token counts, group membership, per-depth latency fits and system/model mapping, not just ids and logits. Fitted latency tables are supplied in `results/latency/latency-fits.json`. Every row export needs an input hash, output hash and permissions review; a text-free projection does not by itself settle source terms.


The Hugging Face Dataset card is the current source of package coverage, missing materials, source licenses and immutable file identities. The code candidate contains metadata only; source-specific hosted subsets have new hashes and do not replace the full original frozen inputs. See [data coverage](../docs/DATA.md#hosted-coverage).
