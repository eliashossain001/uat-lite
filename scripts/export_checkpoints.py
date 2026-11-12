"""Export release checkpoints: keep only {config, model_state_dict} (drops optimizer /
scheduler state, ~1.3 GB -> ~0.4 GB) and optionally upload them to the Hugging Face Hub.

    python scripts/export_checkpoints.py --out_dir release/
    python scripts/export_checkpoints.py --out_dir release/ --push_to <hf-user>/uat-lite-checkpoints

Uploaded files can then be used anywhere as
    --ckpt_path hf://<hf-user>/uat-lite-checkpoints/mnli.pt
"""
import argparse
from pathlib import Path

import torch

from uat_lite.models import UATLite

TASKS = ["sst2", "mnli", "squad", "medqa", "pubmedqa"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiments_dir", default="./experiments")
    p.add_argument("--out_dir", default="./release")
    p.add_argument("--tasks", nargs="+", default=TASKS)
    p.add_argument("--push_to", default=None, help="Hugging Face repo id, e.g. <user>/uat-lite-checkpoints")
    p.add_argument("--private", action="store_true")
    args = p.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for task in args.tasks:
        src = Path(args.experiments_dir) / task / "bert-base-uncased" / "checkpoints" / "best_model.pt"
        if not src.exists():
            print(f"[skip] {src} not found")
            continue
        ckpt = torch.load(src, map_location="cpu", weights_only=False)
        model = UATLite(**ckpt["config"])
        model.load_state_dict(ckpt["model_state_dict"])  # validates and drops legacy duplicate keys
        dst = out / f"{task}.pt"
        torch.save({"config": ckpt["config"], "model_state_dict": model.state_dict()}, dst)
        print(f"[OK] {src} -> {dst} ({dst.stat().st_size / 1e6:.0f} MB)")

    if args.push_to:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.push_to, private=args.private, exist_ok=True)
        api.upload_folder(folder_path=str(out), repo_id=args.push_to, allow_patterns=["*.pt"])
        print(f"[OK] uploaded to https://huggingface.co/{args.push_to}")


if __name__ == "__main__":
    main()
