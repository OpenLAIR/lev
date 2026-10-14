"""Nanbeige4.2-3B (Nanbeige LLM Lab, 2026) as a Kev backbone: a looped LM pretrained on 28T tokens with its whole
22-layer stack applied `num_loops` times (2), the loss on the last pass only (`loop_loss_weights` empty).

The published remote code is written for transformers 4.42 and carries options the 4.2 checkpoints leave off
(hyper-connections / mHC, n-gram embeddings, depth attention, LoopSplit, shared loop KV); kev.nanbeige refuses a config
that turns any of them on. What remains is a Llama-style pre-norm decoder (GQA, RoPE, SwiGLU) run as a loop:

    h = embed(ids)
    for loop in range(num_loops):        # weight-shared: the same 22 layers every loop
        for layer in layers: h = layer(h)   # x + attn(norm1(x)), then x + mlp(norm2(x))
        h = norm(h)                      # the normed state is both this loop's readout and the next loop's input

which is kev.ouro's loop with a plain pre-norm layer instead of Ouro's sandwich norm, so NanbeigeModel is an OuroModel
(packed pass under Kev's block-causal mask, no KV cache, `step_hidden_states` per loop, depth = loops computed).
Only the last loop was trained as a readout; depth 1 is a readout Kev's head learns (--step_weights decides whether).
scripts/nanbeige_parity.py checks the hidden states against the original implementation.
"""
import torch

from transformers.modeling_layers import GradientCheckpointingLayer

from .ouro import MLP, Attention, OuroConfig, OuroModel, RMSNorm

MODEL_TYPE = "nanbeige"
# Kev's five delimiters (state, question, option open, option close, decide) for Nanbeige's tokenizer. The state token is
# BOS, so every sequence starts as in pretraining; the other four are chat-template tokens of the post-trained model,
# ids 166100-166106. The Base tokenizer (a 166,100-piece sentencepiece model) does not name them, but the Base's embedding
# rows for them are trained (norms 0.16-0.21 where the unused rows from 166107 are zero), so `tokenizer` adds them back at
# those ids. kev.model.user_tokens rewrites them in caller text, so they stay unforgeable.
DELIMITERS = ["<s>", "<|im_start|>", "<tool_call>", "</tool_call>", "<|im_end|>"]
CHAT_TOKENS = ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>", "<tool_call>", "</tool_call>"]   # ids 166100.. in order
# 4.2 leaves every one of these at its default; a config that enables one is a different architecture
UNSUPPORTED = {"enable_hyper_connection": False, "enable_mhc": False, "enable_double_loop_split": False, "loop_share_kv": False,
               "enable_depth_attention": False, "qk_layernorm": False, "attention_bias": False, "mlp_bias": False,
               "emb_split_num": None, "insert_ngram_layer_idx": None, "ngram_insert_all_layers": False, "rope_scaling": None,
               "skip_loop_final_norm": False}


class NanbeigeConfig(OuroConfig):
    model_type = MODEL_TYPE

    def __init__(self, vocab_size=166144, hidden_size=3072, intermediate_size=10752, num_hidden_layers=22, num_attention_heads=48,
                 num_key_value_heads=8, head_dim=128, rms_norm_eps=1e-5, rope_theta=1e7, num_loops=2, loop_loss_weights=None, **kwargs):
        bad = {k: kwargs[k] for k, v in UNSUPPORTED.items() if kwargs.get(k, v) not in (v, [] if v is None else v)}
        if bad: raise ValueError(f"kev.nanbeige implements the plain looped decoder only; this config enables {bad}")
        if loop_loss_weights: num_loops = len(loop_loss_weights) + 1   # what the remote code runs
        for k in UNSUPPORTED: kwargs.pop(k, None)
        kwargs.pop("total_ut_steps", None)
        super().__init__(vocab_size=vocab_size, hidden_size=hidden_size, intermediate_size=intermediate_size, num_hidden_layers=num_hidden_layers,
                         num_attention_heads=num_attention_heads, num_key_value_heads=num_key_value_heads, head_dim=head_dim,
                         rms_norm_eps=rms_norm_eps, rope_theta=rope_theta, total_ut_steps=num_loops, **kwargs)
        self.num_loops = num_loops


class Layer(GradientCheckpointingLayer):
    def __init__(self, c):
        super().__init__()
        self.self_attn, self.mlp = Attention(c), MLP(c)
        self.input_layernorm, self.post_attention_layernorm = RMSNorm(c.hidden_size, c.rms_norm_eps), RMSNorm(c.hidden_size, c.rms_norm_eps)

    def forward(self, x, cos, sin, mask):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, mask)
        return x + self.mlp(self.post_attention_layernorm(x))


class NanbeigeModel(OuroModel):
    """The looped base model. `steps` (default config.num_loops) is how many passes of the stack run."""
    config_class = NanbeigeConfig
    layer_class = Layer


def tokenizer(name, revision=None):
    """The Base's own sentencepiece model (transformers' SentencePieceBackend, the published slow tokenizer's backend) with
    the chat tokens added back at their post-trained ids.

    Not transformers 5's LlamaTokenizer: it rebuilds the BPE model as a `tokenizers` one, which prepends "▁" to the first
    word (this model has add_dummy_prefix False) and, with that fixed, still segments some words differently from
    sentencepiece ('Bracketed' -> 'Brack'+'eted' for 'Bra'+'cket'+'ed'; 22 of 2,061 JevBench texts). This one equals the
    published tokenizer on all 2,061 (scripts/nanbeige_parity.py). AutoTokenizer is not used either: the repo's config.json
    has an auto_map, so it would ask to run the repo's code. A directory without tokenizer.model (the tests' tiny
    tokenizer, which already has the chat tokens) loads as a fast tokenizer."""
    import os
    from huggingface_hub import hf_hub_download
    from transformers import PreTrainedTokenizerFast, SentencePieceBackend
    try:
        model = os.path.join(name, "tokenizer.model") if os.path.isdir(name) else hf_hub_download(name, "tokenizer.model", revision=revision)
    except Exception:
        model = None
    if model is None or not os.path.exists(model): return PreTrainedTokenizerFast.from_pretrained(name, revision=revision)
    tok = SentencePieceBackend(vocab_file=model, bos_token="<s>", eos_token="</s>", unk_token="<unk>", pad_token="<unk>")
    first = len(tok)
    tok.add_special_tokens({"additional_special_tokens": CHAT_TOKENS})
    if first != 166100 or tok.convert_tokens_to_ids(CHAT_TOKENS) != list(range(166100, 166100 + len(CHAT_TOKENS))):
        raise ValueError(f"{name}: not the Nanbeige4.2 tokenizer kev.nanbeige expects (sentencepiece size {first})")
    return tok


def load(name, revision=None, dtype=torch.float32, attn="sdpa"):
    config = NanbeigeConfig.from_pretrained(name, revision=revision)
    config._attn_implementation = attn
    return NanbeigeModel.from_pretrained(name, revision=revision, config=config, dtype=dtype)
