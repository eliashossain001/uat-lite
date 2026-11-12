"""IMDb test OOD evaluation of the SST-2 checkpoint (temperature fitted on the full
SST-2 validation set). SST-2 = short snippets, IMDb = full reviews.

    python scripts/ood/imdb.py
"""
import argparse

from transformers import AutoTokenizer

from uat_lite import data
from uat_lite.inference import load_checkpoint
from uat_lite.protocol import (method_notes, print_fixed_coverage_table, print_latex_rows,
                               print_threshold_table, run_five_methods)
from uat_lite.utils import add_common_args, ckpt, save_json


def main():
    args = add_common_args(argparse.ArgumentParser(description=__doc__), ckpt("sst2"),
                           "./results/imdb_ood.json").parse_args()
    model = load_checkpoint(args.ckpt_path, args.device)
    assert model.num_classes == 2, "expected the 2-class SST-2 checkpoint"
    tok = AutoTokenizer.from_pretrained(model.base_model_name)
    dev = data.sst2_full_dev_loader(tok, args.max_length, args.batch_size)
    test, labels = data.load_imdb_test(tok, args.max_length, args.batch_size)

    results, _ = run_five_methods(model, test, dev, args.M, args.device, uat=args.uat)
    print_threshold_table("IMDb OOD -- threshold-based selective prediction", results, len(labels))
    print_fixed_coverage_table("IMDb OOD", results, len(labels))
    print_latex_rows("IMDb", results)
    save_json({"dataset": "IMDb test", "N": len(labels), "M": args.M,
               "label_mapping": "2-class: negative=0, positive=1 (same as SST-2)",
               "uat": args.uat, "method_notes": method_notes(args.uat), "methods": results}, args.output_json)


if __name__ == "__main__":
    main()
