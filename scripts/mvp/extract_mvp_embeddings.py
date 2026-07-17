#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from lpm.data import read_jsonl
from lpm.embeddings import QwenEmbedder, hash_embed_many, save_embedding_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract MVP action and context embeddings.")
    parser.add_argument("--units", type=Path, default=ROOT / "data/mvp/units.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "data/mvp/embeddings.pt")
    parser.add_argument("--backend", choices=["hash", "qwen"], default="hash")
    parser.add_argument("--model-path", type=Path, default=ROOT / "Models/Qwen3.5-9B")
    parser.add_argument("--hash-dim", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    units = read_jsonl(args.units)
    if not units:
        raise SystemExit(f"no units found at {args.units}")

    action_texts = [unit["response_text"] for unit in units]
    global_texts = [unit.get("global_context_text") or "" for unit in units]
    char_texts = [
        (unit.get("character_context_text") or f"[character|{unit['target_character']}]")
        for unit in units
    ]

    if args.backend == "hash":
        action = hash_embed_many(action_texts, dim=args.hash_dim)
        global_context = hash_embed_many(global_texts, dim=args.hash_dim)
        character_context = hash_embed_many(char_texts, dim=args.hash_dim)
    else:
        embedder = QwenEmbedder(args.model_path, device=args.device, dtype=args.dtype)
        action = embedder.encode(action_texts, batch_size=args.batch_size, max_length=args.max_length)
        global_context = embedder.encode(global_texts, batch_size=args.batch_size, max_length=args.max_length)
        character_context = embedder.encode(char_texts, batch_size=args.batch_size, max_length=args.max_length)
    characters = sorted({unit["target_character"] for unit in units})
    character_to_index = {character: index for index, character in enumerate(characters)}
    character_index = torch.tensor([character_to_index[unit["target_character"]] for unit in units], dtype=torch.long)

    bundle = {
        "unit_ids": [unit["unit_id"] for unit in units],
        "characters": characters,
        "character_index": character_index,
        "action": action,
        "global_context": global_context,
        "character_context": character_context,
        "meta": {
            "backend": args.backend,
            "model_path": str(args.model_path),
            "unit_count": len(units),
            "embedding_dim": int(action.shape[-1]),
        },
    }
    save_embedding_bundle(args.output, bundle)
    print(f"saved {len(units)} units with dim={action.shape[-1]} to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
