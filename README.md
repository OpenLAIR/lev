<p align="center">
  <a href="https://openlair.github.io/lev/">
    <img src="https://openlair.github.io/lev/assets/social.png" alt="Lev: Thinking fast and deep. One checkpoint. Multiple depths." width="100%">
  </a>
</p>

<h1 align="center">Lev · System 1.5 Decision Models</h1>

<p align="center">
  <strong>Thinking Fast and Deep with Looped Decision Models</strong><br>
  One checkpoint. Multiple depth operating points. Probabilities over your options.
</p>

<p align="center">
  <a href="https://www.alphaxiv.org/pdf/2610.lev-looped-decision-models"><img src="https://img.shields.io/badge/Paper-alphaXiv-b31b1b?style=flat-square" alt="Paper on alphaXiv"></a>
  <a href="https://openlair.github.io/lev/"><img src="https://img.shields.io/badge/Website-Lev-2448ff?style=flat-square" alt="Project website"></a>
  <a href="https://huggingface.co/collections/songdj/lev"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models%20%26%20Data-ffcc4d?style=flat-square" alt="Models and data on Hugging Face"></a>
  <a href="https://github.com/OpenLAIR/lev/actions/workflows/verify.yml"><img src="https://img.shields.io/github/actions/workflow/status/OpenLAIR/lev/verify.yml?branch=main&style=flat-square&label=CPU%20tests" alt="CPU tests"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/code-Apache%202.0-5b6467?style=flat-square" alt="Code license: Apache 2.0"></a>
</p>

<p align="center">
  <a href="https://openlair.github.io/lev/"><strong>Website</strong></a> ·
  <a href="https://www.alphaxiv.org/pdf/2610.lev-looped-decision-models"><strong>Paper</strong></a> ·
  <a href="https://openlair.github.io/lev/assets/lev-paper.pdf"><strong>PDF</strong></a> ·
  <a href="https://huggingface.co/collections/songdj/lev"><strong>Models &amp; data</strong></a> ·
  <a href="#quick-start"><strong>Quick start</strong></a> ·
  <a href="#reproduce-the-study"><strong>Reproduce</strong></a> ·
  <a href="#citation"><strong>Citation</strong></a>
</p>

> **Release:** the code, three model checkpoints and three datasets are public. Read the paper on [alphaXiv](https://www.alphaxiv.org/pdf/2610.lev-looped-decision-models) or download the [PDF](https://openlair.github.io/lev/assets/lev-paper.pdf); the arXiv listing will be linked once it is announced.

## One checkpoint, more room to think

Lev adapts pretrained recurrent language models into typed decision models. A shared pointer head returns a probability for each caller-provided option after each recurrent step. You choose the depth—and therefore the amount of computation—without switching checkpoints or generating an answer string.

| Flexible depth | Measured accuracy | Structured decisions |
|---|---|---|
| **Up to 4 native depths** from the same checkpoint | **63.2% → 76.2%** for Lev-2.6B, depth 1 → 4 | **One probability per option** with a shared pointer head |

The accuracy result uses JevBench's **231 public tasks** and fp32 readouts. Fixed-depth scaling is the primary result. The agreement stopper provides an additional **offline** routing analysis; its gains depend on the workload. **Lev-2.6B did not pass G3 on locked LevBench in any of its three study seeds.** See [scope and limitations](#scope-and-limitations).

## Model family

Each Hugging Face repository holds the seed-0 LoRA adapter, decision head, tokenizer assets and per-depth temperatures. This source repository holds the model cards and calibration metadata.

| Model | Backbone | Native depths | JevBench accuracy, first → deepest | Artifacts |
|---|---|---|---|---|
| **Lev-1.4B** | Ouro-1.4B | 1–4 | 58.0% → 68.0% | [🤗 Lev-1.4B](https://huggingface.co/songdj/Lev-1.4B) · [Model card](model-cards/Lev-1.4B.md) |
| **Lev-2.6B** | Ouro-2.6B | 1–4 | 63.2% → 76.2% | [🤗 Lev-2.6B](https://huggingface.co/songdj/Lev-2.6B) · [Model card](model-cards/Lev-2.6B.md) |
| **Lev-3B** | Nanbeige4.2-3B-Base | 1–2 | 69.3% → 72.7% | [🤗 Lev-3B](https://huggingface.co/songdj/Lev-3B) · [Model card](model-cards/Lev-3B.md) |

These are existing study results, not new evaluations of the released packages. The cards document backbone revisions, seed coverage, calibration and limitations.

| Dataset | What it holds |
|---|---|
| [🤗 LevBench](https://huggingface.co/datasets/songdj/LevBench) | Development and locked-test decisions across five tiers (rights-filtered subset) |
| [🤗 Lev-train-v2](https://huggingface.co/datasets/songdj/Lev-train-v2) | S31 / S60 / S120 training mixtures, replay and order recipes (rights-filtered subset) |
| [🤗 Lev-results](https://huggingface.co/datasets/songdj/Lev-results) | Per-depth outputs, calibration and latency evidence, without question text |

## Quick start

### 1. Install and run CPU tests

Use Linux and Python 3.12 or 3.13. From the source checkout, create a separate environment and working copy. The distribution is named `lev`; Python imports remain `kev`.

```bash
python3.13 -m venv ../lev-env
source ../lev-env/bin/activate

# Use the CPU wheel for validation; no CUDA dependency download is needed.
python -m pip install --no-cache-dir 'torch==2.8.0+cpu' \
  --index-url https://download.pytorch.org/whl/cpu
cp -R . ../lev-work
python -m pip install --no-cache-dir '../lev-work[dev]'
python -m pip check

(cd ../lev-work && PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 python -m pytest -q -p no:cacheprovider)
```

The validated fresh CPU installation passed **187 tests**, with **14 opt-in network tests skipped**. Tests use tiny randomly initialized models and require no pretrained weights or dataset downloads. Exact dependency versions and evidence are in [validation](docs/VALIDATION.md).

### 2. Score a request at a chosen depth

For model inference, download a checkpoint (`hf download songdj/Lev-2.6B --local-dir ../lev-weights/Lev-2.6B`) and prepare a **separate CUDA-capable PyTorch environment** using the [installation notes](docs/INSTALL.md). The CPU-only test environment above is for validation. A checkpoint needs its adapter configuration, adapter weights and `head.pt`; the loader fetches the pinned backbone, so cache it first for offline use.

```bash
python examples/fixed_depth.py \
  --checkpoint ../lev-weights/Lev-2.6B \
  --depth 2 --device cuda
```

The example scores an original two-option toy request. A chosen fixed depth executes only that many recurrent steps. `--all-depths` returns every depth's output from a full-depth pass; **it does not save computation**. Full-size GPU parity has not been re-run on the released packages.

For calibrated probabilities, verify checkpoint identity and add `--temperatures calibration/Lev-2.6B-s0.json`. Per-depth temperatures are separate from the scalar temperature in `head.pt`. Without this option, the example explicitly labels its scores as uncalibrated.

## Reproduce the study

| What you want to do | Start here |
|---|---|
| Understand the training recipe, readouts and offline stopper | [Reproduction guide](docs/REPRODUCING.md) |
| Redraw available paper figures from aggregate results | [Figure inventory and commands](figures/README.md) |
| Inspect data sources, hashes and reconstruction prerequisites | [Data guide](docs/DATA.md) · [Dataset cards](data-cards/) |
| Review checkpoint details and per-depth calibration | [Model cards](model-cards/) · [Calibration files](calibration/) |
| Inspect installation checks and release conditions | [Validation](docs/VALIDATION.md) · [Release gates](docs/RELEASE_STATUS.md) |

**Seven of thirteen paper figures** can currently be redrawn from the packaged aggregates:

```bash
PYTHONDONTWRITEBYTECODE=1 python figures/reproduce.py \
  --output-dir ../lev-figures
```

The redraws preserve the study's table-value assertions. They do not recompute confidence intervals from the complete per-question collection. The companion [datasets](docs/RESOURCES.md) publish source-specific licensed subsets and existing text-free results. Complete source rights and end-to-end reconstruction remain open; these subsets are not the full frozen corpus.

## Scope and limitations

- **Adaptive routing is offline.** The saved all-depth logits support analysis of early exits. Online hidden-state continuation and adaptive serving are not implemented. The paper's stopper is cross-fitted per workload, rather than supplied as a universal checkpoint threshold.
- **Gains are workload-dependent.** Lev-2.6B failed the locked LevBench G3 criterion in all three study seeds. The Lev-1.4B and Lev-3B headline seeds passed there; that does not establish a general advantage over fixed-depth frontiers. Greater depth can also overturn an earlier correct decision.
- **Accuracy and timing have different measurement paths.** Accuracy uses fp32 readouts; reported H200 timing uses bf16, batch one and implementation-specific kernels. Latency comparisons are not portable speed guarantees. G3 and the paper's token-weighted cost intervals are different analyses.
- **What has and has not been re-run.** CPU installation and tests pass on every push. Full-size GPU parity on the released packages, complete data reconstruction and all-figure reproduction have not been re-run; see [release status](docs/RELEASE_STATUS.md). Never use the fixed locked test to choose a new rule or checkpoint.

[Full reproduction details](docs/REPRODUCING.md) · [Current release conditions](docs/RELEASE_STATUS.md)

### Verify a source artifact

The checker requires **Gitleaks 8.30.1** and PyYAML. Run it against a pristine source snapshot; see the [snapshot instructions](docs/RELEASE_STATUS.md) for Git checkouts. Environments, caches, downloads and generated outputs must remain outside that snapshot—nothing is silently ignored.

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/verify_release.py --mode hygiene

# Formal release verification remains blocked while required conditions are open.
PYTHONDONTWRITEBYTECODE=1 python scripts/verify_release.py
```

## Citation

**Lev: Thinking Fast and Deep with Looped Decision Models**  
Dingjie Song, Zhiling Yan, Yuxuan Zhang, and Lichao Sun.

Read the paper on [alphaXiv](https://www.alphaxiv.org/pdf/2610.lev-looped-decision-models) or download the [PDF](https://openlair.github.io/lev/assets/lev-paper.pdf). The arXiv identifier will be added when it is announced.

```bibtex
@misc{song2026lev,
  title  = {Lev: Thinking Fast and Deep with Looped Decision Models},
  author = {Song, Dingjie and Yan, Zhiling and Zhang, Yuxuan and Sun, Lichao},
  year   = {2026},
  note   = {Preprint},
  url    = {https://openlair.github.io/lev/}
}
```

## License and acknowledgments

The source code is licensed under [Apache-2.0](LICENSE) and derived from [Jared Palmer's Kev](https://github.com/jaredpalmer/kev). See [NOTICE](NOTICE) for attribution.

Backbone weights and dataset text have separate terms. Review [licensing](docs/LICENSING.md) and the relevant model or dataset card before redistributing artifacts.
