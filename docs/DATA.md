# Data inputs and provenance

No training or evaluation question text is distributed in this source candidate. `evals/**/manifest.json`, `evals/levbench-v1/frozen.json`, and `data/train-v2/manifest.json` are metadata only. Companion Dataset repositories [LevBench](https://huggingface.co/datasets/songdj/LevBench), [Lev-train-v2](https://huggingface.co/datasets/songdj/Lev-train-v2), and [Lev-results](https://huggingface.co/datasets/songdj/Lev-results) are published on Hugging Face. The first two contain separately versioned, rights-filtered source subsets and full positional recipes; results contain existing numerical outputs with question text removed. Authorized access is required.

## Inventory

| Paper name | Code name | Role |
|---|---|---|
| LevBench | eval-v1 | 4,750 development records / 4,870 questions; 9,500 locked-test records / 9,740 questions |
| Kev-Seen | decision-v7 | Train replay, temperature calibration, design evaluation |
| Kev-Transfer | transfer-v9 | Design evaluation on held-out sources |
| Kev-Skills | hard-v1 | Generated training and held-out-template evaluation |
| Kev-Breadth | breadth-v1 | Independent evaluation-only sources; not redistributed here |
| JevBench public | pinned upstream JevBench | 231 public benchmark tasks; external acquisition only |

LevBench's id, near, deep, far and unknowable strata distinguish source/template familiarity, extrapolated generated difficulty, unseen sources and missing decisive information. StrategyQA is report-only; unknowable examples are reported separately from answerable accuracy/gates.

## Training inputs

S31 contains night2 (2,225), hard-v1 (6,000), devtools-v1 (5,320), documents-v1 (5,219), plus decision-v7 replay (12,576). `scripts/prepare_training_mix.py` concatenates the first four in that order. It requires `evals/night2/all.jsonl` already present and uses `kev.suite.load_split` for the other training/calibration partitions. That loader verifies partition hashes and can fetch existing upstream `jaredpalmer/kev-suites` partitions at its pinned revision where permitted. It cannot invent missing or access-restricted data.

Train-60k/120k add graded generators, new hard-skill items, additional examples of existing public sources, ARC/OpenBookQA/CommonsenseQA and the audited CondaQA, Social IQa, CosmosQA, MathQA, QASC and PIQA sources. The incremental pool contains 89,723 records. Train-60k and Train-120k use deterministic stratified prefixes; see the exact counts and hashes in `data/train-v2/manifest.json`.

## Builders and full reconstruction dependencies

Adapted builders are under `scripts/`; public names and metadata schemas differ from the research artifacts. Run `bash scripts/run_data.sh` for usage. The training and evaluation builders can download source datasets and screen candidate records against existing collections. They are not self-contained downloads of LevBench.

Exact reconstruction additionally requires the original S31 partitions, the complete reference collections used for overlap screening, the pinned JevBench public set and the external MMLU-Pro reference subset. The screener enumerates metadata under `evals/` and may read calibration/development/test for deduplication; the training builder explicitly screens against both eval-v1 splits. Do not run these builders against the existing locked study. Only metadata relevant to the public candidate is retained; it is insufficient to reconstruct the complete reference inventory. Screening against this reduced checkout does not establish equivalence to the study corpus.

Some original prerequisite builders (night2, devtools, documents, breadth and the external reference subset) are not bundled in the retained source snapshot. Use the pinned upstream source and source terms to recover them, or provide the original hash-matched partitions. Thus this candidate **does not yet claim byte-for-byte end-to-end data reconstruction from a clean clone**. Neither skipping unavailable references nor excluding restricted sources reconstructs the frozen corpus: such output needs a new version and fresh hashes.

The source revisions and recorded license statements are in `data/sources.json`. The Hugging Face Dataset packages include per-source decisions, evidence references, full license copies, attribution, coverage and checksums. In the hosted packages, 31 of 49 source families meet the packaging conditions; 18 remain reconstruction-only. Each hosted source retains its own terms, including noncommercial/share-alike conditions where applicable. The historical source inventory in this code snapshot is not itself hosting clearance. A Hugging Face config name alone is not a license notice. A noncommercial/share-alike license is not itself a categorical prohibition on redistribution; its conditions must be satisfied. Unclear mirrors and independently copyrighted question banks require separate review.

## Public artifact identities

`PROVENANCE.json` records only public project, upstream and author identities. Dataset partition hashes remain scientific reference identities. Metadata fields, public paths and model names are projected into a public schema; these metadata files and adapted builders are not byte-identical copies of the original artifacts. `MANIFEST.sha256` records the actual public bytes. No question partition was opened for this projection. Detailed development origins are outside the public artifact.


## Hosted coverage

| Input | Original records | Hosted records |
|---|---:|---:|
| LevBench development | 4,750 | 3,570 |
| LevBench test | 9,500 | 7,140 |
| S31 mixture | 18,764 | 8,925 |
| Replay | 12,576 | 5,576 |
| Increment pool | 89,723 | 71,522 |
| S60 mixture (before replay) | 47,424 | 31,798 |
| S120 mixture (before replay) | 107,424 | 79,596 |

Scales overlap and should not be added. See each HF package's `metadata/coverage.json`, `metadata/source-policy.json`, `recipes/` and `manifest.json`. Source files retain native requests and answers; sanitized metadata and source filtering give them new hashes. Ordering recipes retain every original position and record hash, including withheld positions without their text. Training inputs have 57 repeated record IDs in a withheld code-review source; positional identities are retained instead of silently deduplicating the corpus.

Existing seed-0 training metrics distinguish admission from constructed size: Lev-1.4B and Lev-2.6B accepted 31,329 / 59,987 / 119,985 records at S31 / S60 / S120; Lev-3B accepted 31,334 / 59,994 / 119,994. These counts describe the full original inputs, not the smaller hosted subsets. No training or tokenization was rerun to derive them.

A dedicated, hash-allowlisted publication route performed mechanical frozen-test packaging under explicit author authorization. Normal evaluation and `scripts/export_results.py` test restrictions remain unchanged. Dataset packaging did not run inference, scoring, training or tuning. GPU parity is independent of the data packages and has not been re-run on the released model packages.
