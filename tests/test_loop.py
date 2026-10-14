"""kev.loop: looped decoder layers on a tiny random hybrid Qwen3.5 (no downloads, CPU, fp32)."""
import copy
import json
import sys

import pytest
import torch

# transformers binds causal-conv1d / fla kernels at import when installed, and they are CUDA-only: these CPU tests need the
# Qwen3.5 layers on the torch reference path (also the exact fp32 one the parity checks want). Run this file on its own.
for name in ("causal_conv1d", "fla"): sys.modules.setdefault(name, None)

from kev.loop import LoopSpec, LoopedLayers, install
from kev.model import SPECIAL, DecisionModel, encode, rows_of

N_LAYERS, INTERVAL = 4, 2   # linear, full, linear, full: loop blocks align to 2


@pytest.fixture(scope="module")
def tiny_base(tmp_path_factory):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen3_5ForCausalLM, Qwen3_5TextConfig
    root = tmp_path_factory.mktemp("tiny-loop")
    words = "it is charged twice which team billing shipping refund angry the customer".split()
    vocab = {t: i for i, t in enumerate(["<unk>", "<pad>", *SPECIAL, *words])}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>")); tk.pre_tokenizer = pre_tokenizers.Whitespace()
    config = Qwen3_5TextConfig(vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=N_LAYERS, num_attention_heads=2, num_key_value_heads=1,
                               head_dim=16, linear_num_value_heads=2, linear_num_key_heads=1, linear_key_head_dim=8, linear_value_head_dim=8,
                               layer_types=["linear_attention", "full_attention"] * 2, full_attention_interval=INTERVAL, pad_token_id=1)
    torch.manual_seed(0)
    Qwen3_5ForCausalLM(config).save_pretrained(root / "base")
    PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>", pad_token="<pad>", additional_special_tokens=SPECIAL).save_pretrained(root / "base")
    rows = [{"state": "the customer is charged twice" + " it" * i, "questions": {
        "team": {"type": "choice", "instructions": "which team", "criteria": {"billing": None, "shipping": None, "refund": None}, "label": ["billing", "shipping", "refund"][i % 3]},
        "angry": {"type": "noul", "instructions": "is the customer angry", "label": i % 2 == 0}}} for i in range(16)]
    (root / "data.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return root


def make(tiny_base, loop=None, lora=None, seed=1):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tiny_base / "base")
    torch.manual_seed(seed)   # same pointer head (and LoRA init) across the models a test compares
    return tok, DecisionModel(str(tiny_base / "base"), tok, "cpu", loop=loop, lora=lora).eval()


RECS = [{"state": "the customer is charged twice it it", "questions": [
            {"instr": "which team", "options": ["billing", "shipping", "refund"], "label": 0},
            {"instr": "is the customer angry", "options": ["no", "yes"], "label": 1}]},
        {"state": "the customer is angry", "questions": [{"instr": "which team", "options": ["refund", "billing"], "label": 0}]}]


def logits(model, tok, path):
    encs = [encode(tok, r) for r in RECS]
    with torch.no_grad():
        if path == "rows": return [z for e in encs for z in model.forward(e)]
        if path == "shared": return [z for zs in model.forward_rows_batch(encs, shared_prefix=True) for z in zs]
        if path == "prefix": return [torch.log(p) for e in encs for p in model.probs(e)]   # serving: state once into a DynamicCache, rows from it
    raise ValueError(path)


def close(a, b, tol=1e-5):
    return all(torch.allclose(x.float(), y.float(), atol=tol) for x, y in zip(a, b))


def softmax_close(a, b, tol=1e-5):
    return close([torch.log_softmax(x, -1) for x in a], [torch.log_softmax(y, -1) for y in b], tol)


def randomize(gates, scale=0.5):
    with torch.no_grad(): gates.g.normal_(0, scale)


def test_schedule_and_config(tiny_base):
    tok, m = make(tiny_base, LoopSpec(2, 4, 3))
    layers = m.lm.layers
    assert isinstance(layers, LoopedLayers) and len(layers) == 8
    assert [(l.layer_index, l.p) for l in layers] == [(0, 0), (1, 0), (2, 0), (3, 0), (2, 1), (3, 1), (2, 2), (3, 2)]
    assert m.lm.config.num_hidden_layers == 8 and m.lm.config.layer_types == ["linear_attention", "full_attention"] * 4
    assert len(list(layers.children())) == N_LAYERS   # registered once
    _, plain = make(tiny_base)
    assert set(m.lm.state_dict()) == set(plain.lm.state_dict())   # parameter names unchanged
    assert m.loop.g.shape == (2, 2, 32)
    with pytest.raises(ValueError): make(tiny_base, LoopSpec(1, 3, 2))   # not aligned to full_attention_interval


@pytest.mark.parametrize("path", ["rows", "shared", "prefix"])
def test_zero_gates_are_the_base(tiny_base, path):
    """Freshly looped (gates 0) = the unlooped model, on every forward path."""
    tok, plain = make(tiny_base)
    _, looped = make(tiny_base, LoopSpec(2, 4, 3))
    assert close(logits(plain, tok, path), logits(looped, tok, path))


def unrolled(tiny_base, spec):
    """Reference for an ungated loop: a plain (unshared) model whose layer list is the schedule, every entry a deep copy
    with its own layer_idx: no LoopedLayer, no shared modules."""
    tok, m = make(tiny_base)
    base = m.lm
    layers = list(base.layers)
    seq = []
    for j, (i, _) in enumerate(spec.schedule(len(layers))):
        layer = copy.deepcopy(layers[i])
        (getattr(layer, "linear_attn", None) or layer.self_attn).layer_idx = j
        seq.append(layer)
    base.layers = torch.nn.ModuleList(seq)
    base.config.layer_types = [base.config.layer_types[i] for i, _ in spec.schedule(len(layers))]
    base.config.num_hidden_layers = len(seq)
    return tok, m


@pytest.mark.parametrize("path", ["rows", "shared", "prefix"])
def test_ungated_loop_equals_unrolled_copies(tiny_base, path):
    spec = LoopSpec(2, 4, 3, gate="none")
    tok, looped = make(tiny_base, spec)
    _, ref = unrolled(tiny_base, spec)
    assert close(logits(looped, tok, path), logits(ref, tok, path))
    _, plain = make(tiny_base)
    assert not close(logits(looped, tok, path), logits(plain, tok, path))   # the loop does change the function


@pytest.mark.parametrize("gate", ["channel", "scalar"])
def test_paths_agree_with_trained_gates(tiny_base, gate):
    """Non-zero gates: the row form, the shared prefix (training) and the DynamicCache prefix (serving) agree, i.e. every
    pass of a shared layer has its own cache slot (attention keys/values and DeltaNet conv/recurrent states)."""
    tok, m = make(tiny_base, LoopSpec(0, 4, 2, gate=gate))
    randomize(m.loop)
    rows = logits(m, tok, "rows")
    assert softmax_close(rows, logits(m, tok, "shared"))
    assert softmax_close(rows, logits(m, tok, "prefix"))
    _, plain = make(tiny_base)
    assert not close(rows, logits(plain, tok, "rows"))


def test_lora_applied_once_and_shared(tiny_base):
    """peft finds each shared layer once: the adapter has the unlooped model's tensors, and gradients reach it from every pass."""
    from peft import get_peft_model_state_dict
    _, plain = make(tiny_base, lora=4)
    tok, m = make(tiny_base, LoopSpec(2, 4, 2, gate="none"), lora=4)
    assert set(get_peft_model_state_dict(m.lm)) == set(get_peft_model_state_dict(plain.lm))
    m.train()
    z = m.forward(encode(tok, RECS[0]))[0]
    z.logsumexp(-1).backward()
    grads = {n: p.grad for n, p in m.lm.named_parameters() if "lora_B" in n and "layers.2." in n}
    assert grads and all(g is not None for g in grads.values())


def test_gate_gradient_flows_from_zero(tiny_base):
    tok, m = make(tiny_base, LoopSpec(2, 4, 2), lora=4)
    m.train()
    torch.nn.functional.cross_entropy(m.forward(encode(tok, RECS[0]))[0][None], torch.tensor([0])).backward()
    assert m.loop.g.grad is not None and m.loop.g.grad.abs().sum() > 0


def test_train_and_reload(tiny_base, tmp_path, monkeypatch):
    """kev.train --loop saves the spec and gates in head.pt; kev.checkpoint rebuilds the same looped model."""
    from kev import train
    from kev.checkpoint import Checkpoint, LoadOptions, read_meta
    out = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", ["kev.train", "--base", str(tiny_base / "base"), "--data", str(tiny_base / "data.jsonl"), "--device", "cpu",
                                      "--batch", "2", "--lr", "1e-3", "--lora", "4", "--epochs", "1", "--loop", "2:4x2", "--loop_lr", "5e-2", "--out", str(out)])
    train.main()
    meta = read_meta(out)
    assert meta.loop["spec"] == LoopSpec(2, 4, 2).to_dict() and meta.loop["gates"]["g"].abs().sum() > 0
    tok, m = Checkpoint(str(out)).load("cpu", LoadOptions(merge=False))
    assert torch.equal(m.loop.g.detach(), meta.loop["gates"]["g"])
    _, merged = Checkpoint(str(out)).load("cpu", LoadOptions())
    assert softmax_close(logits(m, tok, "rows"), logits(merged, tok, "prefix"), 1e-4)
    # warm start: an unlooped run can seed a looped one (fresh gates); a looped one only the same loop
    plain = tmp_path / "plain"
    monkeypatch.setattr(sys, "argv", ["kev.train", "--base", str(tiny_base / "base"), "--data", str(tiny_base / "data.jsonl"), "--device", "cpu",
                                      "--batch", "2", "--lora", "4", "--epochs", "1", "--out", str(plain)])
    train.main()
    _, fresh = make(tiny_base, LoopSpec(2, 4, 2), lora=4)
    Checkpoint(str(plain)).warm_start(fresh, read_meta(out))
    assert fresh.loop.g.abs().sum() == 0
    _, other = make(tiny_base, LoopSpec(0, 2, 2), lora=4)
    ours = read_meta(out); ours.loop = {"spec": LoopSpec(0, 2, 2).to_dict(), "gates": None}
    with pytest.raises(ValueError): Checkpoint(str(out)).warm_start(other, ours)


def test_unshared_control(tiny_base):
    """share=False: repeat passes run registered deep copies with their own adapter; zero gates are still the base, and
    the copies are separate parameters (the schedule's compute, more weights)."""
    from peft import get_peft_model_state_dict
    tok, plain = make(tiny_base, lora=4)
    _, m = make(tiny_base, LoopSpec(2, 4, 3, share=False), lora=4)
    m.head.load_state_dict(plain.head.state_dict())   # the copies' adapters draw from the RNG before the head is built
    assert close(logits(plain, tok, "rows"), logits(m, tok, "rows"))
    extra = set(get_peft_model_state_dict(m.lm)) - set(get_peft_model_state_dict(plain.lm))
    assert extra and all("loop_copies" in k for k in extra) and len({k.split("loop_copies.")[1].split(".")[0] for k in extra}) == 4
    randomize(m.loop)
    rows = logits(m, tok, "rows")
    assert softmax_close(rows, logits(m, tok, "shared")) and softmax_close(rows, logits(m, tok, "prefix"))


def test_lora_resume_is_identical(tiny_base, tmp_path, monkeypatch):
    """A looped LoRA run stopped after step 3 of 16 (resume point: optimizer, scheduler, RNG, data position and the trainable
    tensors) and continued with --resume 1 ends with the same adapter, head and gates as one uninterrupted run."""
    from peft import load_peft_weights
    from kev import train
    from kev.checkpoint import read_meta
    common = ["kev.train", "--base", str(tiny_base / "base"), "--data", str(tiny_base / "data.jsonl"), "--device", "cpu", "--batch", "2", "--lr", "1e-3",
              "--accum", "1", "--lora", "4", "--epochs", "2", "--loop", "2:4x2", "--loop_lr", "5e-2"]
    whole, split = tmp_path / "whole", tmp_path / "split"
    monkeypatch.setattr(sys, "argv", common + ["--out", str(whole)]); train.main()
    monkeypatch.setattr(sys, "argv", common + ["--out", str(split), "--stop_after", "3"]); train.main()
    assert not (split / "head.pt").exists() and (split / "resume").exists()
    monkeypatch.setattr(sys, "argv", common + ["--out", str(split), "--resume", "1"]); train.main()
    a, b = load_peft_weights(str(whole), device="cpu"), load_peft_weights(str(split), device="cpu")
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    ma, mb = read_meta(whole), read_meta(split)
    assert torch.equal(ma.loop["gates"]["g"], mb.loop["gates"]["g"]) and all(torch.equal(ma.head[k], mb.head[k]) for k in ma.head)


# --- v2: input injection, runtime depth, teacher KL ------------------------------------------------------------------

def test_set_repeats_rebuilds_schedule_and_config(tiny_base):
    """Depth is a runtime knob: r=1 is the plain model whatever the trained gates; the config follows the schedule."""
    from kev.loop import set_repeats
    tok, plain = make(tiny_base)
    _, m = make(tiny_base, LoopSpec(2, 4, 3))
    randomize(m.loop)
    set_repeats(m.lm, 1)
    assert len(m.lm.layers) == N_LAYERS and m.lm.config.num_hidden_layers == N_LAYERS
    assert close(logits(plain, tok, "rows"), logits(m, tok, "rows")) and close(logits(plain, tok, "prefix"), logits(m, tok, "prefix"))
    set_repeats(m.lm, 5)   # past the trained 3 passes: the last pass's gates are reused
    assert len(m.lm.layers) == 2 + 2 * 5 + 0 and m.lm.config.layer_types == ["linear_attention", "full_attention"] * 6
    assert softmax_close(logits(m, tok, "rows"), logits(m, tok, "prefix"))


@pytest.mark.parametrize("r", [2, 3])
def test_injection_paths_agree(tiny_base, r):
    """With the block input re-injected, the row form, the shared prefix (state and branch streams) and the cached
    serving pass still compute the same function: each stream mixes with its own recorded block input."""
    from kev.loop import set_repeats
    tok, m = make(tiny_base, LoopSpec(2, 4, 3, gate="none", inject=0.5))
    with torch.no_grad(): m.loop.inject_logit.normal_(0, 1)
    set_repeats(m.lm, r)
    rows = logits(m, tok, "rows")
    assert softmax_close(rows, logits(m, tok, "shared")) and softmax_close(rows, logits(m, tok, "prefix"))
    _, plain = make(tiny_base)
    assert not close(rows, logits(plain, tok, "rows"))


def test_injection_weight_one_recomputes_the_base(tiny_base):
    """w -> 1: every later pass starts from the first pass's block input, so it recomputes the first pass and the loop
    is the base function (the identity-preserving end of the injection weight)."""
    tok, plain = make(tiny_base)
    _, m = make(tiny_base, LoopSpec(2, 4, 3, gate="none", inject=0.5))
    with torch.no_grad(): m.loop.inject_logit.fill_(30.0)
    assert close(logits(plain, tok, "rows"), logits(m, tok, "rows"), 1e-4)


def test_injection_gradient_reaches_prelude(tiny_base):
    tok, m = make(tiny_base, LoopSpec(2, 4, 2, gate="none", inject=0.5), lora=4)
    m.train()
    torch.nn.functional.cross_entropy(m.forward(encode(tok, RECS[0]))[0][None], torch.tensor([0])).backward()
    assert m.loop.inject_logit.grad is not None and m.loop.inject_logit.grad.abs().sum() > 0


def test_train_depth_sampling_with_teacher(tiny_base, tmp_path, monkeypatch):
    """kev.train samples the depth per micro-batch and adds the teacher KL; the checkpoint keeps the trained depth and
    the injection weights, and KEV_LOOP_REPEATS (LoadOptions.loop_repeats) changes the depth at load."""
    from kev import train
    from kev.checkpoint import Checkpoint, LoadOptions, read_meta
    common = ["kev.train", "--base", str(tiny_base / "base"), "--data", str(tiny_base / "data.jsonl"), "--device", "cpu", "--batch", "2",
              "--accum", "1", "--lora", "4", "--epochs", "1"]
    plain, looped = tmp_path / "plain", tmp_path / "looped"
    monkeypatch.setattr(sys, "argv", common + ["--out", str(plain)]); train.main()
    monkeypatch.setattr(sys, "argv", common + ["--out", str(looped), "--init_from", str(plain), "--teacher", str(plain), "--teacher_w", "1",
                                               "--loop", "2:4x3", "--loop_gate", "none", "--loop_inject", "0.5", "--loop_depths", "1,2,3", "--loop_lr", "5e-2"])
    train.main()
    meta = read_meta(looped)
    assert meta.loop["spec"]["inject"] == 0.5 and not torch.allclose(meta.loop["gates"]["inject_logit"], torch.zeros(2))
    _, m = Checkpoint(str(looped)).load("cpu", LoadOptions(merge=False))
    assert len(m.lm.layers) == 2 + 2 * 3
    _, m1 = Checkpoint(str(looped)).load("cpu", LoadOptions(merge=False, loop_repeats=1))
    assert len(m1.lm.layers) == N_LAYERS
    with pytest.raises(ValueError): Checkpoint(str(plain)).load("cpu", LoadOptions(loop_repeats=2))
