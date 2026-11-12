"""Inference latency per batch for each method under identical batch size / sequence length.

Methods: deterministic, + temperature scaling, global MC dropout (M passes),
UAT-LITE attention (``forward_uat``, M passes) and embedding-only MC
(``forward_with_mc_sampling``, M passes). Reports mean ± std ms and the slowdown
relative to the deterministic model, and records the GPU name.

    python scripts/efficiency/gpu_latency.py --batch_size 32 --seq_length 128 --M 10
"""
import argparse
import time
from contextlib import nullcontext

import numpy as np
import torch

from uat_lite.inference import enable_dropout, load_checkpoint
from uat_lite.utils import ckpt, save_json


def timeit(fn, warmup, repeats):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return np.array(times) * 1000.0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt_path", default=ckpt("mnli"))
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seq_length", type=int, default=128)
    p.add_argument("--M", type=int, default=10)
    p.add_argument("--T", type=float, default=1.3, help="temperature used for the +TS rows")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeats", type=int, default=50)
    p.add_argument("--output_json", default="./results/gpu_latency.json")
    args = p.parse_args()
    assert torch.cuda.is_available(), "a CUDA GPU is required"

    model = load_checkpoint(args.ckpt_path, "cuda")
    ids = torch.randint(1000, model.config.vocab_size, (args.batch_size, args.seq_length), device="cuda")
    mask = torch.ones_like(ids)
    amp = torch.autocast("cuda", dtype=torch.float16) if args.fp16 else nullcontext()
    M = args.M

    def deterministic():
        model.eval()
        return model.classifier(model.transformer(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0])

    def mc_dropout():
        model.eval()
        enable_dropout(model)
        return sum(torch.softmax(model.classifier(model.dropout(
            model.transformer(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0])), -1) for _ in range(M)) / M

    methods = {
        "base": deterministic,
        "base_ts": lambda: deterministic() / args.T,
        "mc_dropout": mc_dropout,
        "uat_lite_attention": lambda: model.forward_uat(ids, mask, num_samples=M)["logits_samples"].mean(0),
        "uat_lite_attention_ts": lambda: model.forward_uat(ids, mask, num_samples=M)["logits_samples"].mean(0) / args.T,
        "embedding_mc": lambda: model.forward_with_mc_sampling(ids, mask, num_samples=M)["logits_samples"].mean(0),
    }
    results = {}
    with torch.no_grad(), amp:
        for name, fn in methods.items():
            t = timeit(fn, args.warmup, args.repeats)
            results[name] = {"mean_ms": float(t.mean()), "std_ms": float(t.std())}
    base = results["base"]["mean_ms"]
    gpu = torch.cuda.get_device_name(0)
    print(f"\n  GPU: {gpu} | batch={args.batch_size} seq={args.seq_length} M={M} fp16={args.fp16}")
    for name, r in results.items():
        r["slowdown"] = r["mean_ms"] / base
        print(f"  {name:>22s}  {r['mean_ms']:9.1f} ± {r['std_ms']:6.1f} ms   {r['slowdown']:5.1f}x")
    save_json({"gpu": gpu, **{k: v for k, v in vars(args).items() if k != "output_json"}, "results": results},
              args.output_json)


if __name__ == "__main__":
    main()
