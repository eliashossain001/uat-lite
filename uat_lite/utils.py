import json
import random
from pathlib import Path

import numpy as np
import torch


def default_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_native(obj):
    """Recursively convert NumPy / torch scalars and arrays to JSON-serializable types."""
    if isinstance(obj, dict):
        return {str(k): to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_native(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(to_native(obj), f, indent=2)
    print(f"[OK] saved {path}")


def ckpt(task):
    """Default checkpoint path written by scripts/train/train.py."""
    return f"./experiments/{task}/bert-base-uncased/checkpoints/best_model.pt"


def add_common_args(parser, ckpt_path, output, M=10):
    parser.add_argument("--ckpt_path", default=ckpt_path)
    parser.add_argument("--M", type=int, default=M, help="MC samples")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--output_json", default=output)
    add_uat_arg(parser)
    return parser


def add_uat_arg(parser, choices=("attention", "embedding", "mc_logits")):
    from .inference import DEFAULT_UAT, UAT_HELP
    parser.add_argument("--uat", default=DEFAULT_UAT, choices=list(choices), help=UAT_HELP)
    return parser
