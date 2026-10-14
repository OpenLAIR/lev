"""Looped decoder layers: a contiguous block of the backbone's layers runs `repeats` times with shared weights.

The backbone keeps its own weights and parameter names; only the order in which its layers run changes. With the block
[start, end) and repeats R, the schedule is

    layers[:start], (layers[start:end]) x R, layers[end:]

so a 24-layer backbone with 8:16 x 2 runs 32 layer passes over 24 layers' weights. Passes after the first are gated
(ReZero style): pass p of layer i returns h + g[p, i] * (layer(h) - h), with g zero-initialised, so a freshly looped
model computes exactly what the unlooped one did (tests/test_loop.py) and training decides how much of the extra depth
to use. `gate="none"` re-applies the layers ungated (the naive retrofit), `"scalar"` learns one gate per repeated
layer, `"channel"` (default) one per hidden channel. `share=False` is the unshared control (depth up-scaling): passes
after the first run deep copies of the block (registered as `loop_copies`, so peft gives them their own LoRA), same
schedule and compute, more parameters.

Input injection (`inject=w0`, v2, after McLeish et al. 2025 and Looped Qwen 2026): every pass after the first starts its
block from z = w * e + (1 - w) * h instead of h, where e is the block's input on the first pass (the prelude output, the
same token's state) and w = sigmoid(a_p) is learned per pass, initialised at w0. Re-injecting e keeps later passes on the
pretrained manifold; with w near 1 a later pass recomputes the first, so the loop starts close to the base without the
zero-initialised gate that never opened in v1. The block entry is recorded per forward stream (keyed by the rotary
embedding tensor every layer of a forward receives), so transformers' forward, kev.shared_prefix (state and branch
streams) and cached serving passes each mix with their own e.

Depth at runtime: `set_repeats(lm, r)` runs the block r times (r may differ from the trained `repeats`; passes past the
trained ones reuse the last pass's gate and injection weight). kev.train samples r per micro-batch (--loop_depths) and
KEV_LOOP_REPEATS sets it for evaluation, for depth curves.

How it is wired, so every existing path keeps working:
- `LoopedLayers` replaces the backbone's `layers` ModuleList. Its registered children are the original layers (state_dict
  names, peft targets, `.to()`, gradient checkpointing flags are unchanged), while iterating or slicing it yields the
  schedule, which is what transformers' forward (`self.layers[: num_hidden_layers]`) and kev.shared_prefix
  (`for layer in base.layers`) read.
- Each schedule entry is a `LoopedLayer` that sets its mixer's `layer_idx` to its schedule position for the call, so a
  DynamicCache built from the expanded config (`num_hidden_layers`, `layer_types` rewritten to the schedule) keeps one
  slot per pass: attention keys/values and Gated DeltaNet conv/recurrent states never collide across passes.
- The gates live in `LoopGates`, a module of DecisionModel (trained with the head, saved in head.pt under `loop`).

Not supported with a loop (refused where they are reached): kev.fused_qwen35, kev.cuda_graphs, the MLX backend and
full-weight training/saving (save_pretrained would write the expanded config).
"""
import copy
import math
import re
from collections import OrderedDict
from dataclasses import asdict, dataclass, replace

import torch
import torch.nn as nn

GATES = ("channel", "scalar", "none")
SPEC = re.compile(r"(\d+):(\d+)x(\d+)")


@dataclass(frozen=True)
class LoopSpec:
    start: int
    end: int
    repeats: int
    gate: str = "channel"
    share: bool = True
    inject: float | None = None   # initial weight of the re-injected block input on passes >= 1; None = no injection

    @classmethod
    def parse(cls, text, gate="channel", share=True, inject=None):
        """'8:16x2' -> layers 8..15 run twice."""
        m = SPEC.fullmatch(text.strip())
        if not m: raise ValueError(f"--loop must look like START:ENDxREPEATS (e.g. 8:16x2), got {text!r}")
        return cls(int(m[1]), int(m[2]), int(m[3]), gate, share, inject)

    @classmethod
    def from_dict(cls, d):
        return None if d is None else cls(**d)

    def to_dict(self):
        return asdict(self)

    @property
    def width(self):
        return self.end - self.start

    def validate(self, n_layers, layer_types=None, interval=None):
        if not 0 <= self.start < self.end <= n_layers: raise ValueError(f"loop block {self.start}:{self.end} is not inside the backbone's {n_layers} layers")
        if self.repeats < 1: raise ValueError("loop repeats must be >= 1")
        if self.gate not in GATES: raise ValueError(f"loop gate must be one of {GATES}")
        if self.inject is not None and not 0 < self.inject < 1: raise ValueError("loop inject (initial weight) must be in (0, 1)")
        if interval and (self.start % interval or self.end % interval):
            # hybrid bases interleave Gated DeltaNet and attention layers with this period; an aligned block keeps the
            # expanded layer_types periodic (the MLX loader and the fused kernels assume it)
            raise ValueError(f"loop block {self.start}:{self.end} must be aligned to full_attention_interval={interval}")

    def schedule(self, n_layers):
        """[(layer index, pass)] in execution order; pass 0 is a layer's first (ungated) run."""
        return ([(i, 0) for i in range(self.start)] + [(i, p) for p in range(self.repeats) for i in range(self.start, self.end)]
                + [(i, 0) for i in range(self.end, n_layers)])

    def label(self):
        return (f"{self.start}:{self.end}x{self.repeats}/{self.gate}" + ("" if self.share else "/unshared")
                + ("" if self.inject is None else f"/inject{self.inject:g}"))


class LoopGates(nn.Module):
    """The loop's own parameters. g[p - 1, i - start]: the gate of pass p >= 1 of block layer i, zero-initialised (the
    looped model starts as the base). inject_logit[p - 1]: the injection weight of pass p >= 1, sigmoid(.) = w0 at init.
    A pass past the trained ones (set_repeats beyond `repeats`) reuses the last pass's values."""

    def __init__(self, spec, hidden_size):
        super().__init__()
        self.spec = spec
        n = max(spec.repeats - 1, 0)
        shape = {"channel": (n, spec.width, hidden_size), "scalar": (n, spec.width, 1), "none": (0,)}[spec.gate]
        self.g = nn.Parameter(torch.zeros(shape))
        if spec.inject is not None:
            self.inject_logit = nn.Parameter(torch.full((n,), math.log(spec.inject / (1 - spec.inject))))

    def _pass(self, p):
        n = max(self.spec.repeats - 1, 0)
        if n == 0: raise ValueError("a loop trained with repeats=1 has no parameters for later passes")
        return min(p, n) - 1

    def get(self, layer, p):
        return None if self.spec.gate == "none" or p == 0 else self.g[self._pass(p), layer - self.spec.start]

    def inject_weight(self, p):
        return None if self.spec.inject is None or p == 0 else torch.sigmoid(self.inject_logit[self._pass(p)])

    def summary(self):
        """Mean |g| per pass (>= 1): how much of each extra pass the model uses; with injection also w per pass."""
        out = [] if self.spec.gate == "none" or self.g.numel() == 0 else [round(float(self.g[p].abs().mean()), 5) for p in range(self.g.shape[0])]
        if self.spec.inject is not None: out += [f"w={float(torch.sigmoid(a)):.3f}" for a in self.inject_logit]
        return out


def _mixer(layer):
    return getattr(layer, "linear_attn", None) or getattr(layer, "self_attn")


class LoopedLayer(nn.Module):
    """One schedule entry: the shared layer run at schedule position `idx` (its cache slot), gated on repeat passes.
    The layer and the gates are held as plain attributes, not submodules, so they are registered once (in LoopedLayers
    and DecisionModel) and never twice in a state_dict or a peft scan."""

    def __init__(self, layer, layer_index, p, idx, gates, entries):
        super().__init__()
        object.__setattr__(self, "inner", layer)
        object.__setattr__(self, "gates", gates)
        object.__setattr__(self, "entries", entries)   # LoopedLayers.entries: block input of the first pass, per stream
        self.layer_index, self.p, self.idx = layer_index, p, idx
        self.block_entry = layer_index == gates.spec.start

    @property
    def block_type(self):   # kev.shared_prefix picks the mask by it
        return self.inner.block_type

    def _inject(self, hidden_states, kwargs):
        """Record the first pass's block input, or mix it into a later pass's (see the module docstring)."""
        if not self.block_entry or self.gates.spec.inject is None: return hidden_states
        rope = kwargs.get("position_embeddings")
        key = id(rope[0]) if rope is not None else None
        if self.p == 0:
            self.entries[key] = hidden_states; self.entries.move_to_end(key)
            while len(self.entries) > 4: self.entries.popitem(last=False)   # a forward has at most two streams (state, branch)
            return hidden_states
        e = self.entries.get(key)
        if e is None: raise RuntimeError("loop injection: no recorded block input for this forward stream")
        w = self.gates.inject_weight(self.p).to(hidden_states.dtype)
        return w * e + (1 - w) * hidden_states

    def _run(self, call, hidden_states, *args, **kwargs):
        hidden_states = self._inject(hidden_states, kwargs)
        mixer = _mixer(self.inner)
        own = mixer.layer_idx
        mixer.layer_idx = self.idx
        try:
            out = call(hidden_states, *args, **kwargs)
        finally:
            mixer.layer_idx = own
        g = self.gates.get(self.layer_index, self.p)
        return out if g is None else hidden_states + g.to(out.dtype) * (out - hidden_states)

    def __call__(self, hidden_states, *args, **kwargs):
        # transformers' forward: through the layer's own __call__ (its gradient checkpointing applies)
        return self._run(self.inner, hidden_states, *args, **kwargs)

    def forward(self, hidden_states, *args, **kwargs):
        # kev.shared_prefix calls layer.forward directly (it checkpoints the whole step itself)
        return self._run(self.inner.forward, hidden_states, *args, **kwargs)


class LoopedLayers(nn.ModuleList):
    """The backbone's layers (registered as before) whose iteration, length and indexing are the loop schedule."""

    def __init__(self, layers, spec, gates, copies=None, config=None):
        base = list(layers)
        object.__setattr__(self, "schedule", [])   # ModuleList.__init__ registers through len(self): the originals go in at 0..n-1
        super().__init__(base)
        for k, v in dict(spec=spec, base=base, gates=gates, copies=copies, config=config, entries=OrderedDict(),
                         base_types=list(getattr(config, "layer_types", None) or [])).items():
            object.__setattr__(self, k, v)
        self.set_repeats(spec.repeats)

    def set_repeats(self, r):
        """Run the block r times from now on: rebuild the schedule and keep the config's depth and layer types (what
        caches and per-layer masks follow) in step with it."""
        if r < 1: raise ValueError("loop repeats must be >= 1")
        if self.copies is not None and r > self.spec.repeats: raise ValueError("the unshared control has copies for its trained depth only")
        spec, base, copies = replace(self.spec, repeats=r), self.base, self.copies
        pick = lambda i, p: base[i] if p == 0 or copies is None else copies[(p - 1) * spec.width + i - spec.start]
        order = spec.schedule(len(base))
        object.__setattr__(self, "schedule", [LoopedLayer(pick(i, p), i, p, j, self.gates, self.entries) for j, (i, p) in enumerate(order)])
        object.__setattr__(self, "repeats", r)
        if self.config is not None:
            if self.base_types: self.config.layer_types = [self.base_types[i] for i, _ in order]
            self.config.num_hidden_layers = len(self.schedule)

    def __iter__(self):
        return iter(self.schedule)

    def __len__(self):
        return len(self.schedule)

    def __getitem__(self, idx):
        return self.schedule[idx]


def text_model(lm):
    """The decoder stack (`layers`, `config`) under a possible peft wrapper."""
    base = lm.get_base_model() if hasattr(lm, "get_base_model") else lm
    return base


def install(lm, spec):
    """Loop `lm`'s layers by `spec` in place; -> LoopGates (to be registered by the caller). Rewrites the config's depth
    and layer types to the schedule, so caches and per-layer masks follow it."""
    base = text_model(lm)
    if isinstance(base.layers, LoopedLayers): raise ValueError("backbone is already looped")
    cfg = base.config
    n = len(base.layers)
    spec.validate(n, getattr(cfg, "layer_types", None), getattr(cfg, "full_attention_interval", None))
    gates = LoopGates(spec, cfg.hidden_size)
    copies = None
    if not spec.share:   # registered on the decoder stack: saved, moved and LoRA-adapted like any layer
        copies = nn.ModuleList([copy.deepcopy(base.layers[i]) for _ in range(1, spec.repeats) for i in range(spec.start, spec.end)])
        base.loop_copies = copies
    cfg.loop_base_layers, cfg.loop_spec = n, spec.to_dict()
    base.layers = LoopedLayers(base.layers, spec, gates, copies, cfg)   # rewrites cfg.layer_types / num_hidden_layers
    return gates


def set_repeats(lm, r):
    """Run a looped backbone's block r times (training samples r per micro-batch; KEV_LOOP_REPEATS for evaluation)."""
    layers = text_model(lm).layers
    if not isinstance(layers, LoopedLayers): raise ValueError("backbone is not looped")
    layers.set_repeats(r)


def is_looped(lm):
    return isinstance(getattr(text_model(lm), "layers", None), LoopedLayers)
