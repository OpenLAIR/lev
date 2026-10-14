---
library_name: peft
base_model: Nanbeige/Nanbeige4.2-3B-Base
tags:
- decision-model
- recurrent-model
- lora
---
# Lev-3B — release draft

[Project website](https://openlair.github.io/lev/) · [Paper](https://openlair.github.io/lev/#paper) · [GitHub](https://github.com/OpenLAIR/lev) · [Hugging Face collection — private](https://huggingface.co/collections/songdj/lev)

[Resource availability](../docs/RESOURCES.md)

Hub repository: [songdj/Lev-3B](https://huggingface.co/songdj/Lev-3B) **(private)**. Seed-0 artifacts are privately staged; public availability and GPU parity remain pending. Lev is a typed decision model with one checkpoint supporting 2 native recurrent depths; it returns probabilities over caller-provided options and does not generate a chain of thought.

## Provenance

- Base: `Nanbeige/Nanbeige4.2-3B-Base` at `4a38e817c14b9f2c4b69b1e05cbda90395872f6b`.
- Public checkpoint identity: `Lev-3B-s0`, seed 0.
- S31: original mixture 18,764 + decision-v7 replay 12,576 records before context admission. LoRA rank 16, learning rate 1e-4, one epoch, effective batch 64, uniform per-depth loss, shared pointer head, length sorting, bf16 training autocast, max state 5,120, none-pair augmentation 0.25.
- Required weight layout: LoRA adapter weights/config, `head.pt`, tokenizer assets, and a separate `temperatures.json` for per-depth calibration. Existing calibration values are packaged at `calibration/Lev-3B-s0.json`; verify them against the eventual public checkpoint. The source `head.pt` has scalar `temperature=1.0`, not a per-depth temperature array. Its `base_revision` pins the base even if `adapter_config.json` has no revision.
- Public artifact hashes and checkpoint revision: **pending public release verification**. Additional seed revisions must be described separately.

## Existing results

| Depth | JevBench public accuracy | JevBench hard accuracy | H200 bf16 p50 (ms) |
|---|---|---|---|
| 1 | 69.3% | 46.8% | 23.0 |
| 2 | 72.7% | 49.5% | 44.7 |


JevBench scores use the 231 public tasks, fp32 accuracy readouts and a separate bf16 batch-one timing pass. Times are specific to an H200 and this implementation; they are not a portable speed guarantee.

Locked LevBench answerable/non-report-only accuracy at depths 1–2: 50.81%, 59.31%. Offline adaptive accuracy: 58.22%, token-weighted record cost 0.884 relative to the deepest readout. The headline seed passed G3 on locked LevBench; this is a workload-specific result. G3 and the paper's token-cost CI should not be conflated; see docs/REPRODUCING.md.

These values are copied programmatically from packaged aggregate inputs; no test examples or weights were evaluated during release preparation.

## Usage and limitations

Use `examples/fixed_depth.py` with a local checkpoint. For all-depth scores, every recurrent step is computed. The paper's stopper is cross-fitted separately on each workload, including two classifiers/thresholds per panel. It is not a universal threshold shipped with this checkpoint. `scripts/stopper_config.py` is an offline calibration utility; online adaptive continuation is not implemented in this release.

Depth helps harder items more than easier ones, but can also overturn a correct earlier decision. Calibration depends on domain and workload. Most controls and the 1.4B/3B headline models have one evaluated seed in the locked study. Use independent development requests for new calibration and never tune on the fixed locked test.

## License and intended use

The fixed-revision upstream card declares Apache-2.0. See docs/LICENSING.md before preparing weight notices. This draft deliberately leaves the HF weight-license metadata unset until the exact released artifacts and notices are reviewed. Source code remains Apache-2.0, separately from dataset terms.

Paper citation and arXiv identifier: pending the final public record; see the [paper section](https://openlair.github.io/lev/#paper).


Companion datasets are [privately staged](../docs/RESOURCES.md): [LevBench](https://huggingface.co/datasets/songdj/LevBench), [Lev-train-v2](https://huggingface.co/datasets/songdj/Lev-train-v2), and [Lev-results](https://huggingface.co/datasets/songdj/Lev-results). Benchmark/training downloads are source-specific rights-filtered subsets; the complete frozen corpus still requires lawful acquisition of withheld sources.
