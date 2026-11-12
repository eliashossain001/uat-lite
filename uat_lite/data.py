"""Evaluation data: in-distribution dev/test splits and OOD test sets.

All loaders return pre-tokenized, unshuffled DataLoaders whose batches hold
``input_ids``, ``attention_mask`` and ``labels``.

In-distribution protocol (used by every evaluation script): take the
validation split, permute it with ``np.random.RandomState(seed)``; the first
``dev_ratio`` fraction is the *dev* half (temperature fitting), the rest is
the *test* half (reported numbers).
"""

import csv
import urllib.request
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

TASKS = {
    "sst2": {"hf": ("glue", "sst2"), "split": "validation", "fields": ["sentence"]},
    "mnli": {"hf": ("glue", "mnli"), "split": "validation_matched", "fields": ["premise", "hypothesis"]},
    "squad": {"hf": ("squad_v2",), "split": "validation", "fields": ["question", "context"]},
}
MNLI_LABELS = {"entailment": 0, "neutral": 1, "contradiction": 2}


class EncodedDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def encode(tokenizer, texts_a, texts_b=None, max_length=128):
    if texts_b is None:
        return tokenizer(texts_a, truncation=True, padding="max_length",
                         max_length=max_length, return_tensors="pt")
    return tokenizer(texts_a, texts_b, truncation=True, padding="max_length",
                     max_length=max_length, return_tensors="pt")


def make_loader(encodings, labels, batch_size=32, idx=None):
    labels = torch.as_tensor(np.asarray(labels), dtype=torch.long)
    if idx is not None:
        encodings = {k: v[idx] for k, v in encodings.items()}
        labels = labels[idx]
    return DataLoader(EncodedDataset(encodings, labels), batch_size=batch_size, shuffle=False)


def dev_test_indices(n, dev_ratio=0.5, seed=42):
    perm = np.random.RandomState(seed).permutation(n)
    n_dev = int(n * dev_ratio)
    return perm[:n_dev], perm[n_dev:]


# ---------------------------------------------------------------------------
# In-distribution tasks
# ---------------------------------------------------------------------------

def load_task_texts(task):
    """Validation split of a training task -> (texts_a, texts_b or None, labels).

    SQuAD 2.0 is the binary answerable (1) / unanswerable (0) task.
    """
    cfg = TASKS[task]
    val = load_dataset(*cfg["hf"])[cfg["split"]]
    a = list(val[cfg["fields"][0]])
    b = list(val[cfg["fields"][1]]) if len(cfg["fields"]) == 2 else None
    if task == "squad":
        labels = [0 if len(x["text"]) == 0 else 1 for x in val["answers"]]
    else:
        labels = list(val["label"])
    return a, b, np.array(labels)


def dev_test_loaders(task, tokenizer, max_length=128, batch_size=32, dev_ratio=0.5, seed=42):
    """(dev_loader, test_loader) for sst2 / mnli / squad."""
    a, b, labels = load_task_texts(task)
    enc = encode(tokenizer, a, b, max_length)
    dev_idx, test_idx = dev_test_indices(len(labels), dev_ratio, seed)
    dev = make_loader(enc, labels, batch_size, dev_idx)
    test = make_loader(enc, labels, batch_size, test_idx)
    print(f"[data] {task}: dev={len(dev.dataset)} test={len(test.dataset)}")
    return dev, test


def mnli_dev_loader(tokenizer, max_length=128, batch_size=32, seed=42):
    """Dev half of MNLI validation_matched (HF glue); temperature fitting for NLI OOD."""
    return dev_test_loaders("mnli", tokenizer, max_length, batch_size, seed=seed)[0]


# ---------------------------------------------------------------------------
# OOD test sets
# ---------------------------------------------------------------------------

def load_anli(tokenizer, max_length=128, batch_size=32):
    """ANLI R1+R2+R3 test (3-class). Returns (loader, round_ids, labels)."""
    prem, hyp, labels, rounds = [], [], [], []
    for rn, split in [("R1", "test_r1"), ("R2", "test_r2"), ("R3", "test_r3")]:
        raw = load_dataset("anli", split=split, trust_remote_code=True)
        prem += list(raw["premise"])
        hyp += list(raw["hypothesis"])
        labels += list(raw["label"])
        rounds += [rn] * len(raw)
    labels = np.array(labels)
    print(f"[data] ANLI: {len(labels)} examples")
    return make_loader(encode(tokenizer, prem, hyp, max_length), labels, batch_size), rounds, labels


def load_hans_hf(tokenizer, max_length=128, batch_size=32):
    """HANS validation via HF datasets (2-class). Returns (loader, heuristics, labels)."""
    raw = load_dataset("hans", split="validation")
    labels = np.array(raw["label"])
    enc = encode(tokenizer, list(raw["premise"]), list(raw["hypothesis"]), max_length)
    print(f"[data] HANS (HF): {len(labels)} examples")
    return make_loader(enc, labels, batch_size), list(raw["heuristic"]), labels


HANS_URL = "https://raw.githubusercontent.com/tommccoy1/hans/master/heuristics_evaluation_set.txt"


def load_hans_tsv(tokenizer, path="./data/hans/heuristics_evaluation_set.txt",
                  max_length=128, batch_size=32):
    """HANS from the official TSV (downloaded if missing), no HF loading script.

    Labels: entailment=0, non-entailment=1. Returns (loader, heuristics, labels).
    """
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[data] downloading HANS -> {path}")
        urllib.request.urlretrieve(HANS_URL, str(path))
    prem, hyp, labels, heur = [], [], [], []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            s1, s2 = row.get("sentence1"), row.get("sentence2")
            lab = row.get("label") or row.get("gold_label")
            h = row.get("heuristic")
            if None in (s1, s2, lab, h):
                continue
            prem.append(s1)
            hyp.append(s2)
            heur.append(h)
            labels.append(0 if lab.strip().lower() == "entailment" else 1)
    labels = np.array(labels, dtype=np.int64)
    print(f"[data] HANS (TSV): {len(labels)} examples")
    return make_loader(encode(tokenizer, prem, hyp, max_length), labels, batch_size), heur, labels


def mnli_dev_loader_tsv(tokenizer, path="./data/mnli/dev_matched.tsv", max_length=128,
                        batch_size=32, seed=42, frac=0.5):
    """Dev half of a local GLUE MNLI dev_matched.tsv (rows with gold label '-' dropped)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"MNLI dev TSV not found: {path} (pass --mnli_dev_tsv)")
    prem, hyp, labels = [], [], []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            s1, s2 = row.get("sentence1"), row.get("sentence2")
            lab = row.get("gold_label") or row.get("label")
            if s1 is None or s2 is None or lab is None:
                continue
            lab = lab.strip().lower()
            if lab not in MNLI_LABELS:
                continue
            prem.append(s1)
            hyp.append(s2)
            labels.append(MNLI_LABELS[lab])
    labels = np.array(labels, dtype=np.int64)
    dev_idx, _ = dev_test_indices(len(labels), frac, seed)
    return make_loader(encode(tokenizer, prem, hyp, max_length), labels, batch_size, dev_idx)


def load_snli_test(tokenizer, max_length=128, batch_size=32):
    """SNLI test (3-class), examples without a gold label (-1) removed."""
    raw = load_dataset("stanfordnlp/snli", split="test")
    keep = [i for i, l in enumerate(raw["label"]) if l != -1]
    prem = [raw["premise"][i] for i in keep]
    hyp = [raw["hypothesis"][i] for i in keep]
    labels = np.array([raw["label"][i] for i in keep])
    print(f"[data] SNLI test: {len(labels)} examples")
    return make_loader(encode(tokenizer, prem, hyp, max_length), labels, batch_size), labels


def load_imdb_test(tokenizer, max_length=128, batch_size=32):
    raw = load_dataset("imdb", split="test")
    labels = np.array(raw["label"])
    print(f"[data] IMDb test: {len(labels)} examples")
    return make_loader(encode(tokenizer, list(raw["text"]), None, max_length), labels, batch_size), labels


def sst2_full_dev_loader(tokenizer, max_length=128, batch_size=32):
    """Full SST-2 validation set (872); temperature fitting for IMDb."""
    raw = load_dataset("glue", "sst2")["validation"]
    return make_loader(encode(tokenizer, list(raw["sentence"]), None, max_length),
                       np.array(raw["label"]), batch_size)


# ---------------------------------------------------------------------------
# Clinical QA (formatted as single-text classification)
# ---------------------------------------------------------------------------

def medqa_examples(split):
    """MedQA-USMLE 4-option -> (texts, labels). ``split`` in {"train", "test"}."""
    raw = load_dataset("GBaker/MedQA-USMLE-4-options")[split]
    letter = {"A": 0, "B": 1, "C": 2, "D": 3}
    texts, labels = [], []
    for q, opts, ans in zip(raw["question"], raw["options"], raw["answer_idx"]):
        opts = [opts.get(chr(65 + i), "") for i in range(4)] if isinstance(opts, dict) else list(opts)
        texts.append(f"Question: {q} Options: " +
                     " ".join(f"({chr(65 + i)}) {o}" for i, o in enumerate(opts)))
        labels.append(letter.get(ans, 0) if isinstance(ans, str) else int(ans))
    return texts, labels


def pubmedqa_examples():
    """PubMedQA pqa_labeled (yes=0, no=1, maybe=2) -> (texts, labels), all 1000 examples."""
    raw = load_dataset("qiaojin/PubMedQA", "pqa_labeled")["train"]
    lab = {"yes": 0, "no": 1, "maybe": 2}
    texts, labels = [], []
    for q, ctx, ans in zip(raw["question"], raw["context"], raw["final_decision"]):
        if isinstance(ctx, dict):
            ctx = " ".join(ctx.get("contexts", []))
        elif isinstance(ctx, list):
            ctx = " ".join(ctx)
        texts.append(f"Question: {q} Context: {ctx}")
        labels.append(lab.get(str(ans).lower(), 2))
    return texts, labels


def pubmedqa_split_indices(n, seed=42):
    """80/10/10 train/val/test permutation split."""
    perm = np.random.RandomState(seed).permutation(n)
    n_train, n_val = int(n * 0.8), int(n * 0.1)
    return perm[:n_train], perm[n_train:n_train + n_val], perm[n_train + n_val:]
