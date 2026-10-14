# Reproducing Lev

## Three supported workflows

1. **Fixed-depth inference:** `model.set_depth(k)` followed by normal inference computes a selected depth. See `examples/fixed_depth.py`.
2. **All-depth readouts and offline routing:** `forward_steps_batch` computes all depths once; the analysis chooses a stopping depth using those saved logits. Reported routing costs are estimates from depth/token costs or the measured latency fits. This does not implement early stopping during that forward pass.
3. **Workload-calibrated stopper configuration:** the JSON utility below stores the full classifiers, ordered features, temperatures and threshold. It is a new portable calibration entry point, not an export of the paper's two-fold fit and not an online continuation engine.

## Train the S31 recipe

Acquire licensed inputs as described in DATA.md. S31 is the original 18,764-record mixture plus 12,576 replay records; it is not a different training corpus from the headline runs. The Ouro-2.6B headline tokenizer/context filter retains 31,329 of those 31,340 records. Counts after filtering can depend on the tokenizer.

```bash
NPROC_PER_NODE=4 bash scripts/run_train.sh \
  ByteDance/Ouro-2.6B 1ed04250da1a9936042725d302e81c8fa2ab5abd \
  data/local/kev08b-release-mix.jsonl outputs/Lev-2.6B --seed 0
```

The wrapper uses LoRA rank 16, learning rate 1e-4, one epoch, effective batch 64, length-sorted microbatches, bf16 autocast, a 5,120-token state admission limit and `p_none_pair=0.25`. Default step losses are uniform. Other backbone revisions are in `configs/backbone-revisions.json`.

Train-60k and Train-120k append 28,660 and 88,660 incremental records to the original mixture. Their JSONL files contain 47,424 and 107,424 records, respectively; the 12,576 replay records are supplied separately. Neither command is a smoke test, and no training is executed during release assembly.

## Read development logits and evaluate stopping

```bash
bash scripts/run_read.sh outputs/Lev-2.6B outputs/rows
bash scripts/run_stopper.sh outputs/rows outputs/analysis
```

Depth rows use fp32, TF32 disabled and batch one. Per-depth temperatures are fitted on decision-v7 calibration labels. The agreement target and stopping threshold use agreement with the deepest prediction and need no target-workload labels. Scoring accuracy and fitting temperature still require their respective labeled splits.

`--fit xfit` fits a different classifier/threshold pair for each of two record-group folds in every target panel. Fold identity is CRC32(group) modulo two; fit on calibration plus the other panel fold and choose the threshold on that other fold. `--fit calibration` fits only the calibration rows. A single threshold cannot reproduce the xfit method across panels. In this code, routing is at question level and costs separately aggregate by question, deepest question in a record, and token-weighted record depth.

G1 combines accuracy non-inferiority and question cost <=0.70; G2 requires an advantage over random escalation; G3 requires a positive lower confidence bound against the fixed-depth frontier. The paper's figure summaries also compare at token-weighted record cost; those CIs are not interchangeable with the question-cost G3. Bootstrap resampling is over record groups, 2,000 replicates, seed zero.

Locked test results are already fixed. This staging work only packages existing aggregate summaries and never reevaluates test. The public loader retains its explicit test guard for independent users; no command in this guide invokes it.

## Export reusable workload configurations

```bash
python scripts/stopper_config.py fit \
  --model songdj/Lev-2.6B@YOUR_CHECKPOINT_REVISION \
  --calibration outputs/rows/decision-v7-calibration.json \
  --adaptation outputs/rows/eval-v1-development.json --out outputs/stopper.json
python scripts/stopper_config.py apply \
  --model songdj/Lev-2.6B@YOUR_CHECKPOINT_REVISION \
  --config outputs/stopper.json --rows outputs/rows/eval-v1-development.json \
  --out outputs/stops.json
```

Use a pinned public checkpoint identity, and the same feature implementation, temperatures and option ordering. Adaptation ignores any labels in the workload rows. For an unbiased evaluation, calibrate on separate workload requests from those used to report results; applying on the calibration workload above only demonstrates file compatibility. JSON stores numeric logistic coefficients or a constant classifier for a degenerate agreement target. No pickle is used. The utility refuses test-labeled inputs. It returns zero-based stop depths after reading all depth logits; no runtime speedup is implied.

## Export text-free results

```bash
python scripts/export_results.py --input outputs/rows/eval-v1-development.json \
  --output outputs/public-rows.json --panel eval-v1-development --model Lev-2.6B-s0
```

The exporter keeps record/question identity, group, source/task, labels or per-depth correctness, token counts, difficulty, tier, flags and logits. It drops state, instructions, criteria, option descriptions, targets and original paths. It checks scalar types and dimensions, rejects duplicate question identities, refuses overwrite, and does not support locked test export. Review source permissions before distributing any row collection. See data-cards/Lev-results.md for the distinction between depth-row and JevBench probability schemas.

`PROVENANCE.json` records public identities. `MANIFEST.sha256` verifies the public bytes. Preserved dataset partition hashes identify scientific inputs; public metadata and builder projections are not original byte-identical artifacts. Work in a separate working copy when generating training or analysis outputs; the pristine source artifact gate permits no generated files.
