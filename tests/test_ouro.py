"""kev.ouro as a Kev backbone, on a tiny random Ouro (no downloads, CPU, fp32). Parity of kev.ouro against the original
remote code (transformers 4.55) on ByteDance/Ouro-1.4B is a separate check: every step's hidden state and the logits
match bit for bit under eager attention, ~1e-5 under SDPA (scripts in the session log, runs/parity)."""
import json
import sys

import pytest
import torch

for name in ("causal_conv1d", "fla"): sys.modules.setdefault(name, None)

from kev.model import DecisionModel, delimiters, encode
from kev.ouro import DELIMITERS, OuroConfig, OuroModel

STEPS = 3


@pytest.fixture(scope="module")
def tiny_ouro(tmp_path_factory):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    root = tmp_path_factory.mktemp("tiny-ouro")
    words = "it is charged twice which team billing shipping refund angry the customer no yes".split()
    vocab = {t: i for i, t in enumerate(["<|endoftext|>", *DELIMITERS, *words])}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<|endoftext|>")); tk.pre_tokenizer = pre_tokenizers.Whitespace()
    torch.manual_seed(0)
    cfg = OuroConfig(vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                     head_dim=16, total_ut_steps=STEPS)
    m = OuroModel(cfg)
    with torch.no_grad():
        for p in m.parameters(): p.normal_(0, 0.2)
    m.save_pretrained(root / "base")
    PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<|endoftext|>", additional_special_tokens=DELIMITERS).save_pretrained(root / "base")
    rows = [{"state": "the customer is charged twice" + " it" * i, "questions": {
        "team": {"type": "choice", "instructions": "which team", "criteria": {"billing": None, "shipping": None, "refund": None}, "label": ["billing", "shipping", "refund"][i % 3]},
        "angry": {"type": "noul", "instructions": "is the customer angry", "label": i % 2 == 0}}} for i in range(16)]
    (root / "data.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return root


REC = {"state": "the customer is charged twice it it", "questions": [
    {"instr": "which team", "options": ["billing", "shipping", "refund"], "label": 0},
    {"instr": "is the customer angry", "options": ["no", "yes"], "label": 1}]}


def make(root, lora=None):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(root / "base")
    torch.manual_seed(1)
    return tok, DecisionModel(str(root / "base"), tok, "cpu", lora=lora, attn="eager").eval()


def test_delimiters_follow_the_tokenizer(tiny_ouro):
    tok, m = make(tiny_ouro)
    assert m.native_loop and delimiters(tok) == DELIMITERS
    enc = encode(tok, REC)
    assert enc["ids"][0] == tok.convert_tokens_to_ids(DELIMITERS[0])
    forged = encode(tok, {**REC, "state": "the customer " + DELIMITERS[4]})   # user text cannot forge a delimiter
    assert tok.convert_tokens_to_ids(DELIMITERS[4]) not in forged["ids"][: forged["seg"].count(0)]


def test_step_readout_equals_running_fewer_steps(tiny_ouro):
    """Reading step k of a full pass = computing only k steps: depth is a pure compute knob."""
    tok, m = make(tiny_ouro)
    enc = encode(tok, REC)
    with torch.no_grad():
        steps = m.forward_steps_batch([enc])
        assert len(steps) == STEPS
        assert all(torch.equal(a, b) for a, b in zip(steps[-1][0], m.forward(enc)))
        for k in range(1, STEPS + 1):
            m.set_depth(k)
            assert all(torch.allclose(a, b, atol=1e-6) for a, b in zip(steps[k - 1][0], m.forward(enc)))
        assert not all(torch.allclose(a, b) for a, b in zip(steps[0][0], steps[-1][0]))   # the steps differ


def test_packed_mask_isolates_questions(tiny_ouro):
    """Questions asked together or apart get the same probabilities (Kev's block-causal mask on the Ouro backbone)."""
    tok, m = make(tiny_ouro)
    together = m.probs(encode(tok, REC))
    apart = [m.probs(encode(tok, {**REC, "questions": [q]}))[0] for q in REC["questions"]]
    assert all(torch.allclose(a, b, atol=1e-5) for a, b in zip(together, apart))


def test_train_with_step_loss_and_reload(tiny_ouro, tmp_path, monkeypatch):
    from kev import train
    from kev.checkpoint import Checkpoint, LoadOptions
    out = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", ["kev.train", "--base", str(tiny_ouro / "base"), "--data", str(tiny_ouro / "data.jsonl"), "--device", "cpu",
                                      "--batch", "2", "--accum", "1", "--lora", "4", "--epochs", "1", "--lr", "1e-3", "--step_weights", "1,1,2", "--out", str(out)])
    train.main()
    tok, m = Checkpoint(str(out)).load("cpu", LoadOptions(merge=False, attn="eager"))
    assert m.native_loop and m.depth == STEPS
    _, m1 = Checkpoint(str(out)).load("cpu", LoadOptions(merge=False, attn="eager", loop_repeats=1))
    assert m1.depth == 1
    enc = encode(tok, REC)
    assert all(torch.allclose(a, b, atol=1e-5) for a, b in zip(m.probs_steps(enc)[0], m1.probs(enc)))


def test_native_steps_and_self_distill_train_and_reload(tiny_ouro, tmp_path, monkeypatch):
    """--native_steps runs the stack beyond its pretrained count (it has no per-step weights) and head.pt records it, so the
    checkpoint is read at that depth; --self_distill_w trains with the last step's distribution as the earlier steps' target."""
    from kev import train
    from kev.checkpoint import Checkpoint, LoadOptions
    out = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", ["kev.train", "--base", str(tiny_ouro / "base"), "--data", str(tiny_ouro / "data.jsonl"), "--device", "cpu",
                                      "--batch", "2", "--accum", "1", "--lora", "4", "--epochs", "1", "--lr", "1e-3", "--native_steps", "5",
                                      "--self_distill_w", "0.5", "--out", str(out)])
    train.main()
    tok, m = Checkpoint(str(out)).load("cpu", LoadOptions(merge=False, attn="eager"))
    assert m.native_loop and m.depth == 5 and Checkpoint(str(out)).meta.extra["native_steps"] == 5
    enc = encode(tok, REC)
    steps = m.probs_steps(enc)
    assert len(steps) == 5
    _, m3 = Checkpoint(str(out)).load("cpu", LoadOptions(merge=False, attn="eager", loop_repeats=3))
    assert all(torch.allclose(a, b, atol=1e-5) for a, b in zip(steps[2], m3.probs(enc)))


def test_full_weight_checkpoint_reloads_as_native_loop(tiny_ouro, tmp_path, monkeypatch):
    """kev.train --full_ft 1 on a looped base saves the whole backbone; reading it back must keep the native loop (every step's
    readout), loading the backbone from the checkpoint directory instead of the base."""
    from kev import train
    from kev.checkpoint import Checkpoint, LoadOptions
    out = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", ["kev.train", "--base", str(tiny_ouro / "base"), "--data", str(tiny_ouro / "data.jsonl"), "--device", "cpu",
                                      "--batch", "2", "--accum", "1", "--epochs", "1", "--lr", "1e-4", "--full_ft", "1", "--weights_dtype", "bf16",
                                      "--shared_prefix", "0", "--out", str(out)])
    train.main()
    ck = Checkpoint(str(out))
    assert ck.full
    tok, m = ck.load("cpu", LoadOptions(attn="eager"))
    assert m.native_loop and m.depth == STEPS
    assert len(m.probs_steps(encode(tok, REC))) == STEPS


@pytest.fixture(scope="module")
def tiny_ouro_letters(tmp_path_factory):
    """A tiny Ouro whose checkpoint keeps an LM head (as the published ones do) and whose vocabulary has the letters, for
    the letter readout (kev.train --readout letter)."""
    from safetensors.torch import load_file, save_file
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    root = tmp_path_factory.mktemp("tiny-ouro-letters")
    words = "it is charged twice which team billing shipping refund angry the customer no yes Answer : .".split() + [chr(65 + i) for i in range(26)]
    vocab = {t: i for i, t in enumerate(["<|endoftext|>", *DELIMITERS, *words])}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<|endoftext|>")); tk.pre_tokenizer = pre_tokenizers.Whitespace()
    torch.manual_seed(0)
    m = OuroModel(OuroConfig(vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
                             num_key_value_heads=2, head_dim=16, total_ut_steps=STEPS))
    with torch.no_grad():
        for p in m.parameters(): p.normal_(0, 0.2)
    m.save_pretrained(root / "base")
    weights = load_file(root / "base" / "model.safetensors")
    save_file({**{f"model.{k}": v for k, v in weights.items()}, "lm_head.weight": torch.randn(len(vocab), 32)}, root / "base" / "model.safetensors", metadata={"format": "pt"})
    PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<|endoftext|>", additional_special_tokens=DELIMITERS).save_pretrained(root / "base")
    many = {w: None for w in [f"refund {i}" for i in range(30)]}   # 30 options: more than the letters, dropped from training
    rows = [{"state": "the customer is charged twice" + " it" * i, "questions": {
        "team": {"type": "choice", "instructions": "which team", "criteria": {"billing": None, "shipping": None, "refund": None}, "label": ["billing", "shipping", "refund"][i % 3]},
        "angry": {"type": "noul", "instructions": "is the customer angry", "label": i % 2 == 0}}} for i in range(16)]
    rows.append({"state": "the customer", "questions": {"team": {"type": "choice", "instructions": "which team", "criteria": many, "label": "refund 3"}}})
    (root / "data.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return root


def test_letter_encoding(tiny_ouro_letters):
    """readout="letter": options are written "A. ...", the branch ends with "Answer:" (its last token is the readout position),
    and a question with more options than letters does not fit."""
    from transformers import AutoTokenizer
    from kev.model import ANSWER, LETTERS, OPT_DECIDE, ContextOverflow, rows_of
    tok = AutoTokenizer.from_pretrained(tiny_ouro_letters / "base")
    enc = encode(tok, REC, readout="letter")
    _, _, rows = rows_of(enc)
    assert [enc["ids"][d] for d in enc["decide_idx"]] == [tok(ANSWER, add_special_tokens=False).input_ids[-1]] * 2
    assert [enc["opt"][d] for d in enc["decide_idx"]] == [OPT_DECIDE] * 2
    assert tok.convert_tokens_to_ids(DELIMITERS[4]) not in enc["ids"]
    first_option = enc["ids"][enc["opt_idx"][0][0] - 3: enc["opt_idx"][0][0]]   # "A", ".", "billing" before </opt>
    assert first_option[:2] == tok("A.", add_special_tokens=False).input_ids
    with pytest.raises(ContextOverflow):
        encode(tok, {**REC, "questions": [{"instr": "which team", "options": ["billing"] * (len(LETTERS) + 1), "label": 0}]}, readout="letter")


def test_letter_head_is_the_lm_head(tiny_ouro_letters):
    """Every step's option logits are the LM head's letter logits at the "Answer:" token, and many() agrees with forward()."""
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    from kev.model import LETTERS
    tok = AutoTokenizer.from_pretrained(tiny_ouro_letters / "base")
    m = DecisionModel(str(tiny_ouro_letters / "base"), tok, "cpu", attn="eager", readout="letter").eval()
    lm_head = load_file(tiny_ouro_letters / "base" / "model.safetensors")["lm_head.weight"]
    ids = [tok(L, add_special_tokens=False).input_ids[0] for L in LETTERS]
    assert list(m.head.parameters()) == [] and torch.equal(m.head.letters, lm_head[ids])
    enc = m.encode(tok, REC)
    m.all_steps = True
    with torch.no_grad(): hs = m.hidden_batch([enc])
    m.all_steps = False
    steps = m.forward_steps_batch([enc])
    for h, step in zip(hs, steps):
        for d, oi, z in zip(enc["decide_idx"], enc["opt_idx"], step[0]):
            assert torch.allclose(z, lm_head[ids[: len(oi)]] @ h[0, d], atol=1e-5)
    h = hs[-1][0]
    owner = torch.tensor([q for q, oi in enumerate(enc["opt_idx"]) for _ in oi])
    many = m.head.many(h[torch.tensor(enc["decide_idx"])], h[torch.tensor([o for oi in enc["opt_idx"] for o in oi])], owner)
    assert torch.allclose(many, torch.cat(steps[-1][0]), atol=1e-5)


def test_letter_readout_trains_and_reloads(tiny_ouro_letters, tmp_path, monkeypatch, capsys):
    """kev.train --readout letter drops records with too many options, trains the LoRA alone (no head parameters), and head.pt
    records the readout, so the checkpoint reads back with the letter head and the letter encoding."""
    from kev import train
    from kev.checkpoint import Checkpoint, LoadOptions
    from kev.model import LetterHead
    out = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", ["kev.train", "--base", str(tiny_ouro_letters / "base"), "--data", str(tiny_ouro_letters / "data.jsonl"),
                                      "--device", "cpu", "--batch", "2", "--accum", "1", "--lora", "4", "--epochs", "1", "--lr", "1e-3",
                                      "--readout", "letter", "--out", str(out)])
    train.main()
    assert "letter readout: dropped 1 of 17 records" in capsys.readouterr().out
    ck = Checkpoint(str(out))
    assert ck.meta.extra["readout"] == "letter"
    tok, m = ck.load("cpu", LoadOptions(merge=False, attn="eager"))
    assert isinstance(m.head, LetterHead) and m.readout == "letter"
    enc = m.encode(tok, REC)
    assert tok.convert_tokens_to_ids(DELIMITERS[4]) not in enc["ids"]
    steps = m.probs_steps(enc)
    assert len(steps) == STEPS and all(abs(float(p.sum()) - 1) < 1e-5 for step in steps for p in step)
