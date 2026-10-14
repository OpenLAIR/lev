"""kev.nanbeige as a Kev backbone, on a tiny random Nanbeige (no downloads, CPU, fp32). Parity against the published remote
code on Nanbeige4.2-3B-Base is scripts/nanbeige_parity.py (two venvs; runs/nanbeige-parity/report.json)."""
import json
import sys

import pytest
import torch

for name in ("causal_conv1d", "fla"): sys.modules.setdefault(name, None)

from kev.model import DecisionModel, delimiters, encode, load_tokenizer
from kev.nanbeige import CHAT_TOKENS, DELIMITERS, NanbeigeConfig, NanbeigeModel

LOOPS = 2


@pytest.fixture(scope="module")
def tiny_nanbeige(tmp_path_factory):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    root = tmp_path_factory.mktemp("tiny-nanbeige")
    words = "it is charged twice which team billing shipping refund angry the customer no yes".split()
    specials = ["<unk>", "<s>", "</s>", *CHAT_TOKENS]
    vocab = {t: i for i, t in enumerate([*specials, *words])}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>")); tk.pre_tokenizer = pre_tokenizers.Whitespace()
    torch.manual_seed(0)
    cfg = NanbeigeConfig(vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         head_dim=16, num_loops=LOOPS)
    m = NanbeigeModel(cfg)
    with torch.no_grad():
        for p in m.parameters(): p.normal_(0, 0.2)
    m.save_pretrained(root / "base")
    PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>", bos_token="<s>", eos_token="</s>", pad_token="<unk>",
                            additional_special_tokens=CHAT_TOKENS).save_pretrained(root / "base")
    rows = [{"state": "the customer is charged twice" + " it" * i, "questions": {
        "team": {"type": "choice", "instructions": "which team", "criteria": {"billing": None, "shipping": None, "refund": None}, "label": ["billing", "shipping", "refund"][i % 3]},
        "angry": {"type": "noul", "instructions": "is the customer angry", "label": i % 2 == 0}}} for i in range(16)]
    (root / "data.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return root


REC = {"state": "the customer is charged twice it it", "questions": [
    {"instr": "which team", "options": ["billing", "shipping", "refund"], "label": 0},
    {"instr": "is the customer angry", "options": ["no", "yes"], "label": 1}]}


def make(root, lora=None):
    tok = load_tokenizer(str(root / "base"))
    torch.manual_seed(1)
    return tok, DecisionModel(str(root / "base"), tok, "cpu", lora=lora, attn="eager").eval()


def test_config_refuses_what_kev_does_not_implement():
    with pytest.raises(ValueError, match="enable_mhc"): NanbeigeConfig(enable_mhc=True)
    with pytest.raises(ValueError, match="emb_split_num"): NanbeigeConfig(emb_split_num=4)
    assert NanbeigeConfig(loop_loss_weights=[0.3, 0.3]).total_ut_steps == 3   # what the remote code runs


def test_delimiters_follow_the_tokenizer(tiny_nanbeige):
    tok, m = make(tiny_nanbeige)
    assert m.native_loop and m.depth == LOOPS and delimiters(tok) == DELIMITERS
    enc = encode(tok, REC)
    assert enc["ids"][0] == tok.convert_tokens_to_ids("<s>")   # the state token is BOS
    for d in DELIMITERS:   # user text cannot forge a delimiter
        forged = encode(tok, {**REC, "state": "the customer " + d})
        assert tok.convert_tokens_to_ids(d) not in forged["ids"][1: forged["seg"].count(0)]


def test_step_readout_equals_running_fewer_loops(tiny_nanbeige):
    tok, m = make(tiny_nanbeige)
    enc = encode(tok, REC)
    with torch.no_grad():
        steps = m.forward_steps_batch([enc])
        assert len(steps) == LOOPS
        m.set_depth(1)
        assert all(torch.allclose(a, b, atol=1e-6) for a, b in zip(steps[0][0], m.forward(enc)))
        assert not all(torch.allclose(a, b) for a, b in zip(steps[0][0], steps[1][0]))


def test_packed_mask_isolates_questions(tiny_nanbeige):
    tok, m = make(tiny_nanbeige)
    together = m.probs(encode(tok, REC))
    apart = [m.probs(encode(tok, {**REC, "questions": [q]}))[0] for q in REC["questions"]]
    assert all(torch.allclose(a, b, atol=1e-5) for a, b in zip(together, apart))


def test_train_and_reload(tiny_nanbeige, tmp_path, monkeypatch):
    from kev import train
    from kev.checkpoint import Checkpoint, LoadOptions
    out = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", ["kev.train", "--base", str(tiny_nanbeige / "base"), "--data", str(tiny_nanbeige / "data.jsonl"), "--device", "cpu",
                                      "--batch", "2", "--accum", "1", "--lora", "4", "--epochs", "1", "--lr", "1e-3", "--step_weights", "1,1", "--out", str(out)])
    train.main()
    tok, m = Checkpoint(str(out)).load("cpu", LoadOptions(merge=False, attn="eager"))
    assert m.native_loop and m.depth == LOOPS
    _, m1 = Checkpoint(str(out)).load("cpu", LoadOptions(merge=False, attn="eager", loop_repeats=1))
    enc = encode(tok, REC)
    assert all(torch.allclose(a, b, atol=1e-5) for a, b in zip(m.probs_steps(enc)[0], m1.probs(enc)))
