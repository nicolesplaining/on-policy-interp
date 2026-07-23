"""Model loading and an architecture adapter shared by all subsystems.

The adapter abstracts the one architectural wrinkle that matters for hooks and
activation extraction: Gemma-3 4B/12B/27B load as
``Gemma3ForConditionalGeneration`` with layers under
``model.model.language_model.layers``, whereas Gemma-3-1B / Qwen / Llama use the
standard ``model.model.layers``. Copied in spirit from
look-ahead ``patch_all_layers_unified.py``.
"""
from __future__ import annotations

import os
from typing import Optional

import torch

# Default project models (outline Section 6).
STUDENT_MODEL = "google/gemma-3-4b-it"
TEACHER_MODEL = "google/gemma-3-27b-it"


def hf_token() -> Optional[str]:
    """Resolve an HF token from env or the on-cluster ``~/.hf_token`` file."""
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        if os.environ.get(key):
            return os.environ[key]
    path = os.path.expanduser("~/.hf_token")
    if os.path.exists(path):
        with open(path) as f:
            tok = f.read().strip()
            return tok or None
    return None


class ModelAdapter:
    """Abstracts over Gemma multimodal vs standard decoder-only layer layout."""

    def __init__(self, model):
        self.model = model
        if hasattr(model, "model") and hasattr(model.model, "language_model"):
            # Gemma3ForConditionalGeneration (4B, 12B, 27B)
            self._layers_fn = lambda m: m.model.language_model.layers
            self._device_fn = lambda m: m.model.language_model.embed_tokens.weight.device
            self._n_layers_fn = lambda m: m.config.text_config.num_hidden_layers
            self._lm = lambda m: m.model.language_model
        elif hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
            # Standard decoder-only: Qwen3, Llama, Gemma-3-1B
            self._layers_fn = lambda m: m.model.layers
            self._device_fn = lambda m: m.model.embed_tokens.weight.device
            self._n_layers_fn = lambda m: m.config.num_hidden_layers
            self._lm = lambda m: m.model
        else:
            raise RuntimeError(
                f"Unrecognized architecture for {type(model).__name__}. Expected "
                "model.model.language_model (Gemma multimodal) or "
                "model.model.embed_tokens (standard decoder-only)."
            )

    def get_layers(self):
        return self._layers_fn(self.model)

    def get_input_device(self):
        return self._device_fn(self.model)

    def get_n_layers(self) -> int:
        return self._n_layers_fn(self.model)

    def get_language_model(self):
        return self._lm(self.model)


def _is_gemma3(model_name: str) -> bool:
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_name, token=hf_token(), trust_remote_code=True)
        return getattr(cfg, "model_type", "") == "gemma3"
    except Exception:
        return "gemma-3" in model_name.lower() and "-1b" not in model_name.lower()


def load_model(
    model_name: str,
    dtype: torch.dtype = torch.bfloat16,
    device_map: str = "auto",
    quantization: Optional[str] = None,
    for_training: bool = False,
    shard: bool = False,
    max_memory: Optional[dict] = None,
):
    """Load ``model_name`` and return ``(model, tokenizer, adapter)``.

    - Gemma-3 4B+ load via ``Gemma3ForConditionalGeneration``.
    - ``quantization`` in {"8bit","4bit"} enables bitsandbytes (teacher inference).
    - ``for_training`` disables ``device_map`` sharding (let the trainer place the
      model), *unless* ``shard=True`` — then the model is spread across GPUs with
      naive model parallelism (``device_map="auto"``, optional ``max_memory``),
      which is how the 12B+ student is full-fine-tuned across H100s.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    token = hf_token()
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    quant_kwargs = {}
    if quantization in ("8bit", "4bit"):
        from transformers import BitsAndBytesConfig
        if quantization == "8bit":
            quant_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=dtype
            )

    load_kwargs = dict(token=token, trust_remote_code=True, torch_dtype=dtype, **quant_kwargs)
    if shard:
        # Naive model parallelism across GPUs (works for full-FT training too).
        load_kwargs["device_map"] = "auto"
        if max_memory:
            load_kwargs["max_memory"] = max_memory
    elif for_training:
        # single-device training; trainer calls .to(device) afterwards.
        load_kwargs.pop("device_map", None)
    else:
        load_kwargs["device_map"] = device_map

    if _is_gemma3(model_name):
        from transformers import Gemma3ForConditionalGeneration
        model = Gemma3ForConditionalGeneration.from_pretrained(model_name, **load_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    if not for_training:
        model.eval()
    adapter = ModelAdapter(model)
    return model, tokenizer, adapter
