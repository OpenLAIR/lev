# Figure reproduction

Run `PYTHONDONTWRITEBYTECODE=1 python figures/reproduce.py --output-dir ../lev-figures` from the checkout after installing the `dev` extra. This redraws seven of the 13 figures used in the paper from packaged aggregate inputs, without loading a model or reading raw evaluation partitions. The original table-value assertions remain in the plotting scripts. Generated PDFs are written outside the artifact; source verification ignores no generated files.

| Paper figure | Entry point | Packaged input | Status |
|---|---|---|---|
| `fig_ablation.pdf` | `make_ablation.py` | `extra_numbers.json` → `rules` | Aggregate redraw |
| `fig_training_col_v2.pdf` | `make_training_v2.py` | `adaptive_numbers.json`, `results/training/*.json` | Aggregate redraw |
| `fig_paired.pdf` | `make_appendix.py` | Unchanged source table transcription `TABLE_PAIRED` | Redraw of study table values, no new CI computation |
| `fig_unlabelled.pdf` | `make_appendix.py` | `extra_numbers.json` → `unlabelled`, `transfer` | Aggregate redraw |
| `fig_difficulty.pdf` | `make_appendix.py` | Original table values `TABLE_DEPTH`, `TABLE_UNKNOWABLE` | Table-value redraw, no raw-row re-estimation |
| `fig_pareto_v2.pdf` | `make_appendix.py` | `results/jevbench-*/pareto/points.json`, dense-control `report.json` summaries | Aggregate redraw |
| `fig_cascade_col.pdf` | `make_cascade_v2.py --column` | `results/locked/cascade-latency.json` | Existing aggregate test result; no test rows opened |
| `fig_teaser_tiers.pdf` | Original `make_teaser_tiers.py` + helpers | Missing per-question JevBench depth outputs and calibration rows for cross-fitted tier stops, plus packaged cascade summary | Row-dependent reproduction pending |
| `fig_motivation_v2.pdf` | Original `make_motivation.py` | Missing pinned JevBench v1.2 results and per-item system outputs | External-input reproduction pending |
| `fig_method_focused.pdf` | Manual method diagram | No statistical input; authored diagram not included as editable source here | Diagram-source export pending |
| `fig_where.pdf` | Original `make_where.py`, `make_flips.py` | Packaged point/tier summaries plus missing per-question correctness transitions by breadth area | Row-dependent reproduction pending |
| `fig_adaptive_v2.pdf` | Original `make_adaptive_v2.py`, `make_adaptive.py` | Packaged headline aggregates, missing development/calibration depth rows for full routing curves | Row-dependent reproduction pending |
| `fig_routing_curve.pdf` | Original `make_adaptive_v2.py` routing-curve helper | Missing development/calibration depth rows and the fitted cross-fold predictions | Row-dependent reproduction pending |

The row-dependent scripts are not advertised as runnable in this candidate. They will be packaged with their complete sanitized input mapping when the approved `Lev-results` collection exists. Ids/logits alone would be insufficient: label or correctness, groups, tokens, tier/difficulty and flags are needed, and JevBench probability files need a distinct numeric-option schema.

`figure-1-preview.svg` is a display copy converted from the existing paper Figure 1, not a re-estimation. It illustrates the README and is not counted among the seven successfully reproducible figures.

The latencies plotted in the cost figure are based on previously measured H200 bf16 per-depth fits. The original accuracy and cost implementations differ in precision and fusion paths; these are implementation-specific comparisons, not universal deployment speedups. A redraw verifies consistency with stored summaries; it does not independently verify the underlying data, bootstrap intervals or benchmark measurements.
