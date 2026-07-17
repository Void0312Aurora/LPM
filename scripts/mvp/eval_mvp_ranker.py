#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from lpm.data import read_jsonl
from lpm.embeddings import load_embedding_bundle
from lpm.eval import as_serializable_metrics, grouped_pairwise_metrics, pairwise_metrics
from lpm.model import TwoLevelRewardRanker


def make_pair_tensors(pairs: list[dict], unit_to_index: dict[str, int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    state_indices: list[int] = []
    pos_indices: list[int] = []
    neg_indices: list[int] = []
    groups: list[str] = []
    for pair in pairs:
        if (
            pair["unit_id"] not in unit_to_index
            or pair["positive_unit_id"] not in unit_to_index
            or pair["negative_unit_id"] not in unit_to_index
        ):
            continue
        state_indices.append(unit_to_index[pair["unit_id"]])
        pos_indices.append(unit_to_index[pair["positive_unit_id"]])
        neg_indices.append(unit_to_index[pair["negative_unit_id"]])
        groups.append(pair["negative_type"])
    return (
        torch.tensor(state_indices, dtype=torch.long),
        torch.tensor(pos_indices, dtype=torch.long),
        torch.tensor(neg_indices, dtype=torch.long),
        groups,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an MVP reward ranker.")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "data/mvp/ranker.pt")
    parser.add_argument("--embeddings", type=Path, default=ROOT / "data/mvp/embeddings.pt")
    parser.add_argument("--pairs", type=Path, default=ROOT / "data/mvp/pairs.jsonl")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


@torch.no_grad()
def main() -> int:
    args = parse_args()
    bundle = load_embedding_bundle(args.embeddings)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = TwoLevelRewardRanker(**checkpoint["config"]).to(args.device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    pairs = read_jsonl(args.pairs)
    unit_to_index = {unit_id: index for index, unit_id in enumerate(bundle["unit_ids"])}
    state_idx, pos_idx, neg_idx, groups = make_pair_tensors(pairs, unit_to_index)
    if state_idx.numel() == 0:
        raise SystemExit("no pairs reference units present in the embedding bundle")
    state_idx = state_idx.to(args.device)
    pos_idx = pos_idx.to(args.device)
    neg_idx = neg_idx.to(args.device)
    global_context = bundle["global_context"].to(args.device)
    character_context = bundle["character_context"].to(args.device)
    character_index = bundle["character_index"].to(args.device)
    action = bundle["action"].to(args.device)

    pos = model(global_context[state_idx], character_context[state_idx], character_index[state_idx], action[pos_idx])
    neg = model(global_context[state_idx], character_context[state_idx], character_index[state_idx], action[neg_idx])
    wrong_state = model(global_context[neg_idx], character_context[neg_idx], character_index[neg_idx], action[pos_idx])
    metrics = {
        "overall": pairwise_metrics(pos.cpu(), neg.cpu()),
        "by_negative_type": grouped_pairwise_metrics(pos.cpu(), neg.cpu(), groups),
        "correct_state_vs_wrong_state": pairwise_metrics(pos.cpu(), wrong_state.cpu()),
    }
    serializable = as_serializable_metrics(metrics)
    text = json.dumps(serializable, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
