"""Ouro (ByteDance, Zhu et al. 2025) as a Kev backbone: a natively looped LM, pretrained on 7.7T tokens with its whole
decoder stack applied `total_ut_steps` times (4) and a loss at every step.

The published checkpoints ship remote code written for transformers 4.55 that no longer loads under transformers 5
(rope init, config and weight-tying APIs moved), so this is a small reimplementation of the base model only, loading the
same weights (`model.*` keys of the OuroForCausalLM checkpoint; the LM head and the exit gate are not needed by Kev):

    h = embed(ids)
    for step in range(steps):            # weight-shared: the same 24 layers every step
        for layer in layers: h = layer(h)   # sandwich norm: x + norm2(attn(norm1(x))), then the same around the MLP
        h = norm(h)                      # the normed state is both this step's readout and the next step's input
        step_states.append(h)

Kev reads its pointer head on every step's state (`step_hidden_states`), so one forward gives a decision per depth and a
depth is chosen at inference by computing only that many steps (`steps`). There is no KV cache: Kev scores an
attention-only backbone in one packed pass under its block-causal mask (DecisionModel.prefix_min_tokens keeps serving
off the prefix cache for this backbone). tests/test_ouro.py checks the logits against the original implementation.
"""
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.utils import ModelOutput

MODEL_TYPE = "ouro"
# Kev's five delimiters (state, question, option open, option close, decide) for Ouro's SmolLM-style tokenizer: added
# special tokens that ordinary decision text never contains (kev.model.delimiters picks them by tokenizer)
DELIMITERS = ["<filename>", "<issue_start>", "<jupyter_code>", "<jupyter_output>", "<empty_output>"]


class OuroConfig(PretrainedConfig):
    model_type = MODEL_TYPE

    def __init__(self, vocab_size=49152, hidden_size=2048, intermediate_size=5632, num_hidden_layers=24, num_attention_heads=16,
                 num_key_value_heads=16, head_dim=128, rms_norm_eps=1e-6, rope_theta=1e6, total_ut_steps=4, hidden_act="silu",
                 pad_token_id=None, **kwargs):
        self.vocab_size, self.hidden_size, self.intermediate_size = vocab_size, hidden_size, intermediate_size
        self.num_hidden_layers, self.num_attention_heads, self.num_key_value_heads = num_hidden_layers, num_attention_heads, num_key_value_heads
        self.head_dim, self.rms_norm_eps, self.rope_theta = head_dim, rms_norm_eps, rope_theta
        self.total_ut_steps, self.hidden_act = total_ut_steps, hidden_act
        kwargs.pop("layer_types", None)
        super().__init__(pad_token_id=pad_token_id, **kwargs)


class RMSNorm(nn.Module):
    def __init__(self, d, eps):
        super().__init__()
        self.weight, self.eps = nn.Parameter(torch.ones(d)), eps

    def forward(self, x):
        dt = x.dtype
        x = x.float()
        return self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)).to(dt)


def rotate_half(x):
    a, b = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-b, a), -1)


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.h, self.kv, self.d = c.num_attention_heads, c.num_key_value_heads, c.head_dim
        self.q_proj = nn.Linear(c.hidden_size, self.h * self.d, bias=False)
        self.k_proj = nn.Linear(c.hidden_size, self.kv * self.d, bias=False)
        self.v_proj = nn.Linear(c.hidden_size, self.kv * self.d, bias=False)
        self.o_proj = nn.Linear(self.h * self.d, c.hidden_size, bias=False)
        self.eager = c._attn_implementation == "eager"

    def forward(self, x, cos, sin, mask):
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.h, self.d).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.kv, self.d).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.kv, self.d).transpose(1, 2)
        q, k = q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
        if self.kv != self.h:
            k, v = k.repeat_interleave(self.h // self.kv, 1), v.repeat_interleave(self.h // self.kv, 1)
        m = mask.to(q.dtype)
        if self.eager:
            a = torch.softmax((q @ k.transpose(-1, -2)) * self.d ** -0.5 + m, -1, dtype=torch.float32).to(q.dtype)
            o = a @ v
        else:
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
        return self.o_proj(o.transpose(1, 2).reshape(B, L, -1))


class MLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.gate_proj = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.up_proj = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.down_proj = nn.Linear(c.intermediate_size, c.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Layer(GradientCheckpointingLayer):
    def __init__(self, c):
        super().__init__()
        self.self_attn, self.mlp = Attention(c), MLP(c)
        n = lambda: RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.input_layernorm, self.input_layernorm_2, self.post_attention_layernorm, self.post_attention_layernorm_2 = n(), n(), n(), n()

    def forward(self, x, cos, sin, mask):
        x = x + self.input_layernorm_2(self.self_attn(self.input_layernorm(x), cos, sin, mask))
        return x + self.post_attention_layernorm_2(self.mlp(self.post_attention_layernorm(x)))


@dataclass
class OuroOutput(ModelOutput):
    last_hidden_state: torch.Tensor = None
    step_hidden_states: tuple = None


class OuroModel(PreTrainedModel):
    """The looped base model. `steps` (default config.total_ut_steps) is how many passes of the stack run."""
    config_class = OuroConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _supports_sdpa = True
    _no_split_modules = ["Layer"]
    layer_class = None   # Layer (set below); kev.nanbeige.NanbeigeModel runs its pre-norm layer in the same loop

    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([self.layer_class(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.steps = config.total_ut_steps
        self.post_init()

    def _init_weights(self, module):   # every tensor comes from the checkpoint; nothing to initialise
        pass

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None, use_cache=None, **kwargs):
        if past_key_values is not None or use_cache:
            raise ValueError("the Ouro backbone has no KV cache (Kev scores it in one packed pass)")
        x = self.embed_tokens(input_ids)
        B, L = input_ids.shape
        if position_ids is None: position_ids = torch.arange(L, device=x.device)[None].expand(B, -1)
        # computed here, not kept as a buffer: transformers 5 builds the model on the meta device, so a non-persistent
        # buffer made in __init__ would never get real values
        c = self.config
        inv_freq = 1.0 / (c.rope_theta ** (torch.arange(0, c.head_dim, 2, dtype=torch.float32, device=x.device) / c.head_dim))
        freqs = position_ids[:, :, None].float() * inv_freq[None, None, :]
        emb = torch.cat((freqs, freqs), -1)
        cos, sin = emb.cos()[:, None].to(x.dtype), emb.sin()[:, None].to(x.dtype)
        mask = self._mask(attention_mask, B, L, x.device)
        states = []
        for _ in range(self.steps):
            for layer in self.layers:
                x = layer(x, cos, sin, mask)
            x = self.norm(x)
            states.append(x)
        return OuroOutput(last_hidden_state=x, step_hidden_states=tuple(states))

    @staticmethod
    def _mask(attention_mask, B, L, device):
        """Additive [B,1,L,L]: a 4D mask (Kev's block-causal one) as given; a 2D padding mask or None -> causal."""
        if attention_mask is not None and attention_mask.dim() == 4: return attention_mask.float()
        allow = torch.ones(L, L, dtype=torch.bool, device=device).tril()[None].expand(B, -1, -1)
        if attention_mask is not None: allow = allow & attention_mask[:, None, :].bool()
        allow = allow | torch.eye(L, dtype=torch.bool, device=device)[None]
        return torch.zeros(B, 1, L, L, device=device).masked_fill(~allow[:, None], torch.finfo(torch.float32).min)


OuroModel.layer_class = Layer
NATIVE_LOOPS = (MODEL_TYPE, "nanbeige")   # natively looped backbones: Ouro here, Nanbeige4.2 in kev.nanbeige


def model_type(name, revision=None):
    """A base's model_type from its config.json alone (no remote code, no prompt); None when it has none."""
    import json, os
    from huggingface_hub import hf_hub_download
    try:
        path = os.path.join(name, "config.json") if os.path.isdir(name) else hf_hub_download(name, "config.json", revision=revision)
        return json.loads(Path(path).read_text(encoding="utf-8")).get("model_type")
    except Exception:
        return None


def is_ouro(name, revision=None):
    """Whether a base is a natively looped checkpoint kev implements (Ouro or Nanbeige4.2)."""
    return model_type(name, revision) in NATIVE_LOOPS


def load(name, revision=None, dtype=torch.float32, attn="sdpa"):
    if model_type(name, revision) == "nanbeige":
        from .nanbeige import load as load_nanbeige
        return load_nanbeige(name, revision, dtype=dtype, attn=attn)
    config = OuroConfig.from_pretrained(name, revision=revision)
    config._attn_implementation = attn
    return OuroModel.from_pretrained(name, revision=revision, config=config, dtype=dtype)
