"""
Representation extractor: loads a decoder-only LM and, for each stimulus,
runs a forward pass with PyTorch hooks to capture the hidden state of the
**last token** at every transformer layer.

Output shape per stimulus: (n_layers, hidden_size)

The last-token vector is the standard extraction point for decoder-only models
(GPT-2, LLaMA, etc.) because it has attended over the full context and is the
position the model uses to make its next-step prediction.
"""

from __future__ import annotations

import gc
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

from .stimuli import Stimulus


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class RepresentationExtractor:
    """
    Wraps a HuggingFace causal-LM and extracts per-layer hidden states for a
    list of text stimuli using forward hooks.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier (e.g. "gpt2-xl" or "gpt2").
    device : str
        "cuda" or "cpu".
    dtype : torch.dtype
        Model precision; float16 saves VRAM on large models.
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self._model: Optional[AutoModelForCausalLM] = None
        self._tokenizer: Optional[AutoTokenizer] = None

    # ------------------------------------------------------------------ #
    # Lazy loading                                                         #
    # ------------------------------------------------------------------ #

    def load(self) -> None:
        if self._model is not None:
            return
        print(f"Loading model: {self.model_name}  (dtype={self.dtype})")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Detect encoder-only models (BERT/RoBERTa/DeBERTa) via architectures
        # or model_type, and use AutoModel instead of AutoModelForCausalLM.
        cfg = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
        arch_names = " ".join((getattr(cfg, "architectures", None) or [""])).lower()
        mtype = str(getattr(cfg, "model_type", "")).lower()
        signal = f"{arch_names} {mtype}"
        encoder_only = any(k in signal for k in [
            "bert", "roberta", "deberta", "electra", "modernbert", "xlm-roberta",
        ]) and "causallm" not in signal
        model_cls = AutoModel if encoder_only else AutoModelForCausalLM
        print(f"  arch_signal='{signal}', encoder_only={encoder_only}, using {model_cls.__name__}")
        # device="auto" shards the model across every visible GPU. Required for
        # 32B+ checkpoints, which do not fit on a single 40GB A100 in bf16.
        kwargs = dict(torch_dtype=self.dtype, low_cpu_mem_usage=True,
                      trust_remote_code=True)
        if self.device == "auto":
            kwargs["device_map"] = "auto"
            self._model = model_cls.from_pretrained(self.model_name, **kwargs)
        elif self.device == "mps":
            # Apple silicon: device_map="auto" mis-shards here, so place explicitly.
            self._model = model_cls.from_pretrained(self.model_name, **kwargs).to("mps")
        else:
            self._model = model_cls.from_pretrained(
                self.model_name, **kwargs).to(self.device)
        self._model.eval()
        # Remote-code models (DeepSeek-V2, Phi-3.5) break on the new DynamicCache
        # API; we never need the KV cache for single forward passes.
        self._model.config.use_cache = False
        print(
            f"  Layers: {self._model.config.num_hidden_layers}  |  "
            f"Hidden size: {self._model.config.hidden_size}  |  "
            f"Params: {sum(p.numel() for p in self._model.parameters()) / 1e6:.0f}M"
        )

    def unload(self) -> None:
        del self._model, self._tokenizer
        self._model = None
        self._tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @property
    def n_layers(self) -> int:
        self.load()
        return self._model.config.num_hidden_layers  # type: ignore[union-attr]

    @property
    def hidden_size(self) -> int:
        self.load()
        return self._model.config.hidden_size  # type: ignore[union-attr]

    # ------------------------------------------------------------------ #
    # Core extraction                                                      #
    # ------------------------------------------------------------------ #

    def _get_layer_modules(self) -> List[torch.nn.Module]:
        """Return the list of transformer-block modules in layer order."""
        cfg = self._model.config  # type: ignore[union-attr]
        model_type = getattr(cfg, "model_type", "gpt2")

        # GPT-2 family
        if hasattr(self._model, "transformer") and hasattr(
            self._model.transformer, "h"  # type: ignore[union-attr]
        ):
            return list(self._model.transformer.h)  # type: ignore[union-attr]

        # LLaMA / Mistral / Gemma family
        if hasattr(self._model, "model") and hasattr(
            self._model.model, "layers"  # type: ignore[union-attr]
        ):
            return list(self._model.model.layers)  # type: ignore[union-attr]

        # Mamba (state-space): model.model.layers works; also try backbone.layers
        if hasattr(self._model, "backbone") and hasattr(
            self._model.backbone, "layers"
        ):
            return list(self._model.backbone.layers)

        # Encoder-only (BERT/RoBERTa/DeBERTa) as AutoModel: encoder.layer
        if hasattr(self._model, "encoder") and hasattr(
            self._model.encoder, "layer"
        ):
            return list(self._model.encoder.layer)
        if hasattr(self._model, "roberta") and hasattr(
            self._model.roberta.encoder, "layer"
        ):
            return list(self._model.roberta.encoder.layer)
        if hasattr(self._model, "deberta") and hasattr(
            self._model.deberta.encoder, "layer"
        ):
            return list(self._model.deberta.encoder.layer)

        raise NotImplementedError(
            f"Cannot auto-detect layer list for model_type='{model_type}'. "
            "Please add a branch in _get_layer_modules()."
        )

    @torch.no_grad()
    def extract_one(self, text: str) -> np.ndarray:
        """
        Extract the last-token hidden state from every layer for a single text.

        Returns
        -------
        np.ndarray of shape (n_layers, hidden_size)
        """
        self.load()
        tokenizer = self._tokenizer  # type: ignore[union-attr]
        model = self._model  # type: ignore[union-attr]

        # With device_map="auto" the model is sharded, so inputs go to whichever
        # device holds the embedding layer rather than to a literal "auto".
        input_device = model.device if self.device == "auto" else self.device
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(input_device)

        layer_modules = self._get_layer_modules()
        captured: Dict[int, torch.Tensor] = {}

        def make_hook(layer_idx: int):
            def hook(module, input, output):
                # output is a tuple; first element is the hidden-state tensor
                # shape: (batch=1, seq_len, hidden_size)
                hidden = output[0] if isinstance(output, tuple) else output
                # grab last token, detach, move to CPU
                captured[layer_idx] = hidden[0, -1, :].detach().cpu().float()
            return hook

        handles = [
            mod.register_forward_hook(make_hook(i))
            for i, mod in enumerate(layer_modules)
        ]

        try:
            model(**inputs, use_cache=False)
        finally:
            for h in handles:
                h.remove()

        # Stack into (n_layers, hidden_size)
        reps = np.stack(
            [captured[i].numpy() for i in range(len(layer_modules))], axis=0
        )
        return reps  # shape: (n_layers, hidden_size)

    def extract_all(
        self, stimuli: List[Stimulus], verbose: bool = True
    ) -> np.ndarray:
        """
        Extract representations for all stimuli.

        Returns
        -------
        np.ndarray of shape (n_stimuli, n_layers, hidden_size)
        """
        self.load()
        all_reps = []
        for i, stim in enumerate(stimuli):
            if verbose:
                print(
                    f"  [{i+1:02d}/{len(stimuli)}] "
                    f"{stim.concept_name:28s} | {stim.form_name:10s}"
                )
            reps = self.extract_one(stim.text)
            all_reps.append(reps)

        return np.stack(all_reps, axis=0)  # (n_stimuli, n_layers, hidden_size)


# ---------------------------------------------------------------------------
# Convenience: extract and save
# ---------------------------------------------------------------------------

def extract_and_save(
    stimuli: List[Stimulus],
    model_name: str,
    save_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> Tuple[np.ndarray, int, int]:
    """
    Run extraction, save to .npz, return (activations, n_layers, hidden_size).
    """
    extractor = RepresentationExtractor(model_name, device=device, dtype=dtype)
    extractor.load()
    print(f"\nExtracting representations for {len(stimuli)} stimuli...")
    activations = extractor.extract_all(stimuli)
    np.savez_compressed(
        save_path,
        activations=activations,
        model_name=np.array(model_name),
        n_stimuli=np.array(len(stimuli)),
        n_layers=np.array(extractor.n_layers),
        hidden_size=np.array(extractor.hidden_size),
    )
    print(f"Saved to {save_path}.npz  shape={activations.shape}")
    n_layers, hidden_size = extractor.n_layers, extractor.hidden_size
    extractor.unload()
    return activations, n_layers, hidden_size


if __name__ == "__main__":
    from .stimuli import build_stimuli

    stimuli = build_stimuli()
    extractor = RepresentationExtractor("gpt2", device="cuda")
    extractor.load()
    reps = extractor.extract_one(stimuli[0].text)
    print(f"Single stimulus reps shape: {reps.shape}")
    all_reps = extractor.extract_all(stimuli[:4])
    print(f"Batch reps shape: {all_reps.shape}")
