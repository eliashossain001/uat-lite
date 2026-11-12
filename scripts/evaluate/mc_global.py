"""Global MC-dropout baseline on the in-distribution test half: accuracy, ECE, Brier,
NLL, AURC (200-point grid) and fixed-coverage accuracy, with max-prob and
negative-entropy confidence; optional latency overhead.

    python scripts/evaluate/mc_global.py --task sst2 --output_json results/sst2_mc_global_M10.json
"""
import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer

from uat_lite import data, metrics
from uat_lite.inference import enable_dropout, load_checkpoint
from uat_lite.utils import add_common_args, ckpt, save_json


@torch.no_grad()
def collect_mc(model, loader, M, device):
    """Mean probs, labels, predictive entropy and mutual information over M passes."""
    model.eval()
    enable_dropout(model)
    probs_all, member_ent_all, labels_all = [], [], []
    for batch in tqdm(loader, desc=f"MC Global (M={M})", leave=False):
        ids, mask = batch["input_ids"].to(device), batch["attention_mask"].to(device)
        prob_acc = torch.zeros(ids.size(0), model.num_classes, device=device)
        ent_acc = torch.zeros(ids.size(0), device=device)
        for _ in range(M):
            out = model.transformer(input_ids=ids, attention_mask=mask)
            p = F.softmax(model.classifier(model.dropout(out.last_hidden_state[:, 0])), dim=-1)
            prob_acc += p
            ent_acc += -(p * torch.log(p + 1e-12)).sum(dim=-1)
        probs_all.append((prob_acc / M).cpu())
        member_ent_all.append((ent_acc / M).cpu())
        labels_all.append(batch["labels"])
    probs = torch.cat(probs_all)
    pred_ent = -(probs * torch.log(probs + 1e-12)).sum(dim=-1)
    return probs.numpy(), torch.cat(labels_all).numpy(), pred_ent.numpy(), (pred_ent - torch.cat(member_ent_all)).numpy()


def evaluate(probs, labels, confidences, bins, tag):
    correct = (probs.argmax(1) == labels).astype(float)
    m = {"accuracy": float(correct.mean()), "ece": metrics.ece(probs, labels, bins),
         "brier": metrics.brier(probs, labels), "nll": metrics.nll(probs, labels),
         "aurc": metrics.aurc_grid(confidences, correct, 200)[0], "selective": {}}
    N = len(labels)
    for cov in (0.7, 0.8, 0.9):
        m["selective"][f"cov_{cov * 100:.0f}"] = {
            "n_admitted": max(1, int(N * cov)),
            "accuracy": metrics.acc_at_coverage(confidences, correct, cov, rounding="floor")}
    print(f"-- {tag} --  " + "  ".join(f"{k}={m[k]:.4f}" for k in ["accuracy", "ece", "brier", "nll", "aurc"]))
    return m


@torch.no_grad()
def measure_overhead(model, loader, M_values, device, num_warmup=3, num_measure=10):
    batch = next(iter(loader))
    ids, mask = batch["input_ids"].to(device), batch["attention_mask"].to(device)
    results = {}
    for M in [1] + list(M_values):
        model.eval()
        if M > 1:
            enable_dropout(model)

        def step():
            for _ in range(M):
                pooled = model.transformer(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]
                model.classifier(model.dropout(pooled) if M > 1 else pooled)

        for _ in range(num_warmup):
            step()
        times = []
        for _ in range(num_measure):
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            step()
            if device == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        results[M] = float(np.mean(times))
    for M in sorted(results):
        print(f"  M={M:>3d}  {results[M]:.4f} s/batch  {results[M] / results[1]:.2f}x")
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True, choices=["sst2", "mnli", "squad"])
    p.add_argument("--bins", type=int, default=15)
    p.add_argument("--measure_overhead", action="store_true")
    args = add_common_args(p, None, None).parse_args()
    args.ckpt_path = args.ckpt_path or ckpt(args.task)

    model = load_checkpoint(args.ckpt_path, args.device)
    tok = AutoTokenizer.from_pretrained(model.base_model_name)
    _, test = data.dev_test_loaders(args.task, tok, args.max_length, args.batch_size)
    probs, labels, pred_ent, mi = collect_mc(model, test, args.M, args.device)

    out = {"task": args.task, "M": args.M, "bins": args.bins,
           "mc_global_maxprob": evaluate(probs, labels, probs.max(1), args.bins, "conf=max_prob"),
           "mc_global_negentropy": evaluate(probs, labels, -pred_ent, args.bins, "conf=neg_entropy"),
           "uncertainty_stats": {"pred_entropy_mean": float(pred_ent.mean()), "pred_entropy_std": float(pred_ent.std()),
                                 "mutual_info_mean": float(mi.mean()), "mutual_info_std": float(mi.std())}}
    if args.measure_overhead:
        out["overhead"] = {str(k): v for k, v in measure_overhead(model, test, [args.M], args.device).items()}
    if args.output_json:
        save_json(out, args.output_json)


if __name__ == "__main__":
    main()
