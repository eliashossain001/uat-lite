"""Clinical QA (Panel B): fine-tune Bio_ClinicalBERT on MedQA (4-class) and PubMedQA
(3-class), then report ECE (accuracy) for Baseline, Base+TS, UAT-LITE, UAT-LITE+TS.
Temperatures are fitted on the validation split.

    python scripts/evaluate/clinical.py               # train + evaluate
    python scripts/evaluate/clinical.py --eval_only   # use existing checkpoints
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from uat_lite import data, metrics
from uat_lite.calibration import fit_temperature
from uat_lite.inference import UAT_ESTIMATORS, load_checkpoint, predict_deterministic
from uat_lite.models import UATLite
from uat_lite.train_data import TextClassificationDataset
from uat_lite.utils import add_uat_arg, default_device, save_json, set_seed

METHODS = ["Baseline", "Base+TS", "UAT-LITE", "UAT-LITE+TS"]


def make_loaders(tok, splits, max_length, batch_size):
    """splits: {"train"|"val"|"test": (texts, labels)}. Eval batches are 2x larger."""
    return {name: DataLoader(TextClassificationDataset(t, l, tok, max_length),
                             batch_size=batch_size if name == "train" else 2 * batch_size,
                             shuffle=name == "train", num_workers=0)
            for name, (t, l) in splits.items()}


def medqa_splits():
    texts, labels = data.medqa_examples("train")
    perm = np.random.RandomState(42).permutation(len(texts))
    n_train = int(len(texts) * 0.9)
    pick = lambda idx: ([texts[i] for i in idx], [labels[i] for i in idx])
    return {"train": pick(perm[:n_train]), "val": pick(perm[n_train:]), "test": data.medqa_examples("test")}


def pubmedqa_splits():
    texts, labels = data.pubmedqa_examples()
    pick = lambda idx: ([texts[i] for i in idx], [labels[i] for i in idx])
    return dict(zip(["train", "val", "test"], map(pick, data.pubmedqa_split_indices(len(texts)))))


def train(model, train_loader, val_loader, epochs, lr, weight_decay, device, warmup_ratio=0.1):
    """Cross-entropy on the [CLS] head; keeps the epoch with the best validation accuracy."""
    model.to(device)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total = len(train_loader) * epochs
    sched = get_linear_schedule_with_warmup(opt, int(total * warmup_ratio), total)
    best_acc, best_state = 0.0, None
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        for batch in tqdm(train_loader, desc=f"epoch {epoch + 1}/{epochs}"):
            ids, mask, y = (batch[k].to(device) for k in ("input_ids", "attention_mask", "labels"))
            logits = model.classifier(model.transformer(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0])
            F.cross_entropy(logits, y).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
        logits, y = predict_deterministic(model, val_loader, device)
        acc = float((logits.argmax(-1) == y).float().mean())
        print(f"  val acc {acc:.4f}")
        if acc >= best_acc:
            best_acc, best_state = acc, {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model.to(device).eval()


def evaluate(model, val_loader, test_loader, M, device, uat):
    logits, labels = predict_deterministic(model, test_loader, device)
    labels = labels.numpy()
    T_base = fit_temperature(*predict_deterministic(model, val_loader, device))
    uat_logits, uat_probs, _ = UAT_ESTIMATORS[uat](model, test_loader, M, device)
    dev_logits, _, dev_labels = UAT_ESTIMATORS[uat](model, val_loader, M, device)
    T_uat = fit_temperature(dev_logits, dev_labels)
    probs = {"Baseline": F.softmax(logits, -1), "Base+TS": F.softmax(logits / T_base, -1),
             "UAT-LITE": uat_probs, "UAT-LITE+TS": F.softmax(uat_logits / T_uat, -1)}
    out = {m: {"ece": metrics.ece(probs[m].numpy(), labels), "acc": metrics.accuracy(probs[m].numpy(), labels)}
           for m in METHODS}
    out["Base+TS"]["T"], out["UAT-LITE+TS"]["T"] = T_base, T_uat
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default=default_device())
    p.add_argument("--M", type=int, default=10)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--base_model", default="emilyalsentzer/Bio_ClinicalBERT")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--medqa_ckpt", default="./experiments/medqa/bert-base-uncased/checkpoints/best_model.pt")
    p.add_argument("--pubmedqa_ckpt", default="./experiments/pubmedqa/bert-base-uncased/checkpoints/best_model.pt")
    p.add_argument("--output_json", default="./results/clinical_panel_b.json")
    add_uat_arg(p)
    args = p.parse_args()
    set_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.base_model)

    results = {}
    for name, splits, n_cls, ckpt_path in [("medqa", medqa_splits, 4, args.medqa_ckpt),
                                           ("pubmedqa", pubmedqa_splits, 3, args.pubmedqa_ckpt)]:
        print(f"\n=== {name} ===")
        loaders = make_loaders(tok, splits(), args.max_length, args.batch_size)
        if args.eval_only:
            model = load_checkpoint(ckpt_path, args.device)
        else:
            model = train(UATLite(args.base_model, n_cls), loaders["train"], loaders["val"],
                          args.epochs, args.lr, args.weight_decay, args.device)
            Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"config": {"base_model_name": args.base_model, "num_classes": n_cls},
                        "model_state_dict": model.state_dict()}, ckpt_path)
        results[name] = evaluate(model, loaders["val"], loaders["test"], args.M, args.device, args.uat)

    print(f"\n  {'Method':>12s}  {'MedQA ECE (acc)':>18s}  {'PubMedQA ECE (acc)':>20s}  {'Avg ECE':>8s}")
    for m in METHODS:
        mq, pq = results["medqa"][m], results["pubmedqa"][m]
        print(f"  {m:>12s}  {mq['ece']:>9.3f} ({mq['acc']:.3f})  {pq['ece']:>11.3f} ({pq['acc']:.3f})"
              f"  {0.5 * (mq['ece'] + pq['ece']):>8.4f}")
    save_json({"uat": args.uat, "seed": args.seed, "M": args.M, "base_model": args.base_model, **results}, args.output_json)


if __name__ == "__main__":
    main()
