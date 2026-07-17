#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from lpm.data import read_jsonl
from lpm.embeddings import load_embedding_bundle
from lpm.eval import as_serializable_metrics, grouped_pairwise_metrics, pairwise_metrics
from lpm.model import TwoLevelRewardRanker, softplus_ranking_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a two-level MVP reward ranker.")
    parser.add_argument("--embeddings", type=Path, default=ROOT / "data/mvp/embeddings.pt")
    parser.add_argument("--pairs", type=Path, default=ROOT / "data/mvp/pairs.jsonl")
    parser.add_argument("--train-pairs", type=Path, default=None)
    parser.add_argument("--val-pairs", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "data/mvp/ranker.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--state-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--character-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--state-noise-std", type=float, default=0.02)
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--state-mode", choices=["full", "global", "character", "action_only"], default="full")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def make_pair_tensors(pairs: list[dict], unit_to_index: dict[str, int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    state_indices: list[int] = []
    neg_indices: list[int] = []
    pos_indices: list[int] = []
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


@torch.no_grad()
def evaluate(model: TwoLevelRewardRanker, bundle: dict, pair_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]], device: str) -> dict:
    state_idx, pos_idx, neg_idx, groups = pair_tensors
    state_idx = state_idx.to(device)
    pos_idx = pos_idx.to(device)
    neg_idx = neg_idx.to(device)
    global_context = bundle["global_context"].to(device)
    character_context = bundle["character_context"].to(device)
    character_index = bundle["character_index"].to(device)
    action = bundle["action"].to(device)
    model.eval()
    pos = model(global_context[state_idx], character_context[state_idx], character_index[state_idx], action[pos_idx])
    neg = model(global_context[state_idx], character_context[state_idx], character_index[state_idx], action[neg_idx])
    metrics = {
        "overall": pairwise_metrics(pos.cpu(), neg.cpu()),
        "by_negative_type": grouped_pairwise_metrics(pos.cpu(), neg.cpu(), groups),
    }
    wrong_state_reward = model(global_context[neg_idx], character_context[neg_idx], character_index[neg_idx], action[pos_idx])
    metrics["correct_state_vs_wrong_state"] = pairwise_metrics(pos.cpu(), wrong_state_reward.cpu())
    return as_serializable_metrics(metrics)


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    bundle = load_embedding_bundle(args.embeddings)
    pairs = read_jsonl(args.pairs)
    unit_to_index = {unit_id: index for index, unit_id in enumerate(bundle["unit_ids"])}
    if args.train_pairs is not None or args.val_pairs is not None:
        if args.train_pairs is None or args.val_pairs is None:
            raise SystemExit("--train-pairs and --val-pairs must be provided together")
        train_pairs = read_jsonl(args.train_pairs)
        val_pairs = read_jsonl(args.val_pairs)
        if not train_pairs:
            raise SystemExit(f"no train pairs found at {args.train_pairs}")
        if not val_pairs:
            raise SystemExit(f"no validation pairs found at {args.val_pairs}")
    else:
        random.shuffle(pairs)
        split_at = max(1, int(len(pairs) * (1.0 - args.val_fraction)))
        train_pairs = pairs[:split_at]
        val_pairs = pairs[split_at:] or pairs[:]
    train_tensors = make_pair_tensors(train_pairs, unit_to_index)
    val_tensors = make_pair_tensors(val_pairs, unit_to_index)
    if train_tensors[0].numel() == 0:
        raise SystemExit("no train pairs reference units present in the embedding bundle")
    if val_tensors[0].numel() == 0:
        raise SystemExit("no validation pairs reference units present in the embedding bundle")

    input_dim = int(bundle["global_context"].shape[-1])
    model = TwoLevelRewardRanker(
        input_dim=input_dim,
        action_dim=int(bundle["action"].shape[-1]),
        state_dim=args.state_dim,
        num_characters=len(bundle["characters"]),
        character_dim=args.character_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        state_noise_std=args.state_noise_std,
        state_mode=args.state_mode,
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    global_context = bundle["global_context"].to(args.device)
    character_context = bundle["character_context"].to(args.device)
    character_index = bundle["character_index"].to(args.device)
    action = bundle["action"].to(args.device)
    state_idx, pos_idx, neg_idx, _ = train_tensors
    state_idx = state_idx.to(args.device)
    pos_idx = pos_idx.to(args.device)
    neg_idx = neg_idx.to(args.device)

    order = torch.arange(state_idx.numel())
    for epoch in range(1, args.epochs + 1):
        model.train()
        permutation = order[torch.randperm(order.numel())]
        losses: list[float] = []
        for start in range(0, permutation.numel(), args.batch_size):
            batch = permutation[start : start + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            pos = model(global_context[state_idx[batch]], character_context[state_idx[batch]], character_index[state_idx[batch]], action[pos_idx[batch]])
            neg = model(global_context[state_idx[batch]], character_context[state_idx[batch]], character_index[state_idx[batch]], action[neg_idx[batch]])
            loss = softplus_ranking_loss(pos, neg, margin=args.margin)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        if epoch == 1 or epoch == args.epochs or epoch % max(1, args.epochs // 5) == 0:
            metrics = evaluate(model, bundle, val_tensors, args.device)
            print(json.dumps({"epoch": epoch, "loss": round(sum(losses) / len(losses), 6), "val": metrics["overall"]}, ensure_ascii=False))

    final_metrics = evaluate(model, bundle, val_tensors, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": {
                "input_dim": input_dim,
                "action_dim": int(bundle["action"].shape[-1]),
                "state_dim": args.state_dim,
                "num_characters": len(bundle["characters"]),
                "character_dim": args.character_dim,
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
                "state_noise_std": args.state_noise_std,
                "state_mode": args.state_mode,
            },
            "characters": bundle["characters"],
            "metrics": final_metrics,
            "data": {
                "embeddings": str(args.embeddings),
                "pairs": str(args.pairs),
                "train_pairs": str(args.train_pairs) if args.train_pairs is not None else None,
                "val_pairs": str(args.val_pairs) if args.val_pairs is not None else None,
                "train_pair_count": int(train_tensors[0].numel()),
                "val_pair_count": int(val_tensors[0].numel()),
            },
        },
        args.output,
    )
    print(json.dumps({"saved": str(args.output), "metrics": final_metrics}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
