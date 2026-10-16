---
library_name: peft
base_model: ByteDance/Ouro-1.4B
tags:
- decision-model
- recurrent-model
- lora
---
# Lev-1.4B

[Project website](https://openlair.github.io/lev/) · [Paper (arXiv)](https://arxiv.org/abs/2610.20951) · [PDF](https://arxiv.org/pdf/2610.20951) · [GitHub](https://github.com/OpenLAIR/lev) · [Hugging Face collection](https://huggingface.co/collections/songdj/lev)

[Resource availability](../docs/RESOURCES.md)

Hub repository: [songdj/Lev-1.4B](https://huggingface.co/songdj/Lev-1.4B). Seed-0 artifacts; full-size GPU parity has not been re-run on the released package. Lev is a typed decision model with one checkpoint supporting 4 native recurrent depths; it returns probabilities over caller-provided options and does not generate a chain of thought.

## Provenance

- Base: `ByteDance/Ouro-1.4B` at `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`.
- Public checkpoint identity: `Lev-1.4B-s0`, seed 0.
- S31: original mixture 18,764 + decision-v7 replay 12,576 records before context admission. LoRA rank 16, learning rate 1e-4, one epoch, effective batch 64, uniform per-depth loss, shared pointer head, length sorting, bf16 training autocast, max state 5,120, none-pair augmentation 0.25.
- Required weight layout: LoRA adapter weights/config, `head.pt`, tokenizer assets, and a separate `temperatures.json` for per-depth calibration. Existing calibration values are packaged at `calibration/Lev-1.4B-s0.json`; verify them against the eventual public checkpoint. The source `head.pt` has scalar `temperature=1.0`, not a per-depth temperature array. Its `base_revision` pins the base even if `adapter_config.json` has no revision.
- Public artifact hashes: `manifest.json` in [songdj/Lev-1.4B](https://huggingface.co/songdj/Lev-1.4B); pin a Hub commit for reproducible downloads. Additional seed revisions must be described separately.

## Existing results

| Depth | JevBench public accuracy | JevBench hard accuracy | H200 bf16 p50 (ms) |
|---|---|---|---|
| 1 | 58.0% | 31.5% | 15.4 |
| 2 | 65.4% | 40.5% | 29.6 |
| 3 | 68.0% | 45.0% | 43.7 |
| 4 | 68.0% | 44.1% | 57.8 |


JevBench scores use the 231 public tasks, fp32 accuracy readouts and a separate bf16 batch-one timing pass. Times are specific to an H200 and this implementation; they are not a portable speed guarantee.

Locked LevBench answerable/non-report-only accuracy at depths 1–4: 43.07%, 51.34%, 56.48%, 58.42%. Offline adaptive accuracy: 57.51%, token-weighted record cost 0.812 relative to the deepest readout. The headline seed passed G3 on locked LevBench; this is a workload-specific result. G3 and the paper's token-cost CI should not be conflated; see docs/REPRODUCING.md.

These values are copied programmatically from packaged aggregate inputs; no test examples or weights were evaluated during release preparation.

## Usage and limitations

Use `examples/fixed_depth.py` with a local checkpoint. For all-depth scores, every recurrent step is computed. The paper's stopper is cross-fitted separately on each workload, including two classifiers/thresholds per panel. It is not a universal threshold shipped with this checkpoint. `scripts/stopper_config.py` is an offline calibration utility; online adaptive continuation is not implemented in this release.

Depth helps harder items more than easier ones, but can also overturn a correct earlier decision. Calibration depends on domain and workload. Most controls and the 1.4B/3B headline models have one evaluated seed in the locked study. Use independent development requests for new calibration and never tune on the fixed locked test.

## License and intended use

The upstream card states Apache-2.0 and also says “This model is intended for research purposes only. It is provided as-is without warranties for production use.” See docs/LICENSING.md for the distinction between that statement and the declared license. This draft deliberately leaves the HF weight-license metadata unset until the exact released artifacts and notices are reviewed. Source code remains Apache-2.0, separately from dataset terms.

Paper: [arXiv:2610.20951](https://arxiv.org/abs/2610.20951) · [PDF](https://arxiv.org/pdf/2610.20951) · [alphaXiv](https://www.alphaxiv.org/abs/2610.20951).


Companion datasets are [published on Hugging Face](../docs/RESOURCES.md): [LevBench](https://huggingface.co/datasets/songdj/LevBench), [Lev-train-v2](https://huggingface.co/datasets/songdj/Lev-train-v2), and [Lev-results](https://huggingface.co/datasets/songdj/Lev-results). Benchmark/training downloads are source-specific rights-filtered subsets; the complete frozen corpus still requires lawful acquisition of withheld sources.
