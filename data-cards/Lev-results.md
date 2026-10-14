---
pretty_name: Lev-results
---
# Lev-results — release draft

[Project website](https://openlair.github.io/lev/) · [Paper](https://openlair.github.io/lev/#paper) · [GitHub](https://github.com/OpenLAIR/lev) · [Hugging Face collection — private](https://huggingface.co/collections/songdj/lev)

[Resource availability](../docs/RESOURCES.md)

Planned repository: `songdj/Lev-results`. This candidate packages aggregate JSON inputs under `results/` and `figures/`, not the full per-question collection. Aggregate figure redraws and re-estimation from row-level observations are different reproducibility levels.

Depth-row schema (version 1): `model`, `panel`, `split`, `depths`, `source_sha256`, and `rows`. Each row retains `id`, `group`, question identifier `question`, `source`, `task`, `type`, `variant`, numeric `label` and/or per-depth `correct`, `state_tokens`, `tokens`, logits as `[depth][option]`, and any `tier`, `difficulty` (`kind`, `k`), `unanswerable`, `report_only`. Question identity is `(id, question)`; bootstrap identity is `group`. No state, question instructions, criteria or option description is included. Model identity, depth/option ordering, inference dtype, temperature status and calibration provenance must accompany the final collection.

`scripts/export_results.py` projects the standard depth-row schema and refuses locked test export. It intentionally does not export JevBench's different per-depth probability files. The latter need `id`, `tier`, `valid`, per-depth correctness and an aligned numeric probability-vector convention; retain a stable option index and remove textual option keys before release. Add a dedicated adapter and parity validation before claiming JevBench rows are supported by this exporter.

The original cost analyses need token counts, group membership, per-depth latency fits and system/model mapping, not just ids and logits. Fitted latency tables are supplied in `results/latency/latency-fits.json`. Every future row export needs an input hash, output hash and permissions review; a text-free projection does not by itself settle source terms.
