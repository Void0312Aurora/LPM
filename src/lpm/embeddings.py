from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


TOKEN_RE = re.compile(r"\w+|[^\s\w]", re.UNICODE)


def l2_normalize(array: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(array, axis=-1, keepdims=True)
    return array / np.maximum(norm, eps)


def hash_embed(text: str, dim: int = 384) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    tokens = TOKEN_RE.findall(text)
    if not tokens:
        tokens = ["<empty>"]
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "little", signed=False)
        index = value % dim
        sign = 1.0 if (value >> 63) == 0 else -1.0
        vec[index] += sign
    return l2_normalize(vec)


def hash_embed_many(texts: Iterable[str], dim: int = 384) -> torch.Tensor:
    arrays = [hash_embed(text, dim=dim) for text in texts]
    return torch.tensor(np.stack(arrays), dtype=torch.float32)


class QwenEmbedder:
    def __init__(
        self,
        model_path: Path,
        device: str = "auto",
        dtype: str = "bfloat16",
        trust_remote_code: bool = True,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[dtype]
        kwargs = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": trust_remote_code,
        }
        if device == "auto":
            kwargs["device_map"] = "auto"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        self.model.eval()
        if device == "cuda":
            self.model.to("cuda")
        elif device != "auto":
            self.model.to(device)
        self.device = next(self.model.parameters()).device

    @torch.no_grad()
    def encode(self, texts: list[str], batch_size: int = 1, max_length: int = 1024) -> torch.Tensor:
        batches: list[torch.Tensor] = []
        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            encoded = self.tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            outputs = self.model(**encoded, output_hidden_states=True, use_cache=False)
            hidden = outputs.hidden_states[-1].float()
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            pooled = torch.nn.functional.normalize(pooled, dim=-1)
            batches.append(pooled.cpu())
        return torch.cat(batches, dim=0)


def save_embedding_bundle(path: Path, bundle: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, path)


def load_embedding_bundle(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)
