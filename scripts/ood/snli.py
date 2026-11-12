"""SNLI test OOD evaluation of the MNLI checkpoint (temperature fitted on MNLI dev).

    python scripts/ood/snli.py
"""
import argparse

from transformers import AutoTokenizer

from uat_lite import data
from uat_lite.inference import load_checkpoint
from uat_lite.protocol import (method_notes, print_fixed_coverage_table, print_latex_rows,
                               print_threshold_table, run_five_methods)
from uat_lite.utils import add_common_args, ckpt, save_json


def main():
    args = add_common_args(argparse.ArgumentParser(description=__doc__), ckpt("mnli"),
                           "./results/snli_ood.json").parse_args()
    model = load_checkpoint(args.ckpt_path, args.device)
    tok = AutoTokenizer.from_pretrained(model.base_model_name)
    dev = data.mnli_dev_loader(tok, args.max_length, args.batch_size)
    test, labels = data.load_snli_test(tok, args.max_length, args.batch_size)

    results, _ = run_five_methods(model, test, dev, args.M, args.device, uat=args.uat)
    print_threshold_table("SNLI OOD -- threshold-based selective prediction", results, len(labels))
    print_fixed_coverage_table("SNLI OOD", results, len(labels))
    print_latex_rows("SNLI", results)
    save_json({"dataset": "SNLI test", "N": len(labels), "M": args.M,
               "label_mapping": "3-class: entailment=0, neutral=1, contradiction=2",
               "uat": args.uat, "method_notes": method_notes(args.uat), "methods": results}, args.output_json)


if __name__ == "__main__":
    main()
