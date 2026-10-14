# Installation and validation boundaries

A source checkout is needed for `scripts/`, `analysis/`, `evals/` metadata, and `figures/`; the wheel exposes the `kev` Python package only. Run shell examples from the checkout root. Keep a pristine source snapshot for artifact verification; perform builds and experiments in a separate working copy.

The CPU reference dependency set uses Python 3.13 with torch 2.8.0, transformers 5.17.0, peft 0.21.0, accelerate 1.15.0, datasets 5.0.1, numpy 2.5.3, scikit-learn 1.9.1, pydantic 2.13.5, matplotlib 3.11.2 and pytest 9.1.1. These exact versions are listed in `requirements-tested.txt`; compatible ranges are in `pyproject.toml`. CPU tests cover native loop behavior and tiny reference models. Full-size backbone loading and numerical parity require a GPU and released checkpoints and remain a release gate.

For core native Ouro/Nanbeige inference, PyTorch's attention implementation suffices. The paper's hybrid Qwen3.5 dense-control kernel path additionally used flash-linear-attention 0.5.2, Triton 3.7.1 and causal-conv1d 1.7.0. PyTorch 2.8 declares a different Triton dependency: this was an explicit environment override, not a generally resolved dependency set. Install those kernels in a separate CUDA environment only when reproducing that implementation, using wheels matching Python, CUDA and the PyTorch ABI. This candidate does not claim a clean resolver has validated that optional override.

JevBench is an optional external benchmark with its own code and data terms:

```bash
git clone https://github.com/fstandhartinger/jevbench.git ../lev-jevbench
git -C ../lev-jevbench checkout --detach bb05a335bc809e61b20c0f745d25499a82b326fc
python -m pip install -e ../lev-jevbench
python scripts/jevbench_eval.py --help
```

Use only its public suite. A Jev/OpenRouter comparison also requires an explicitly supplied `OPENROUTER_API_KEY` and incurs API charges; this package reads that key from the environment only. It never searches neighboring directories for credentials.
