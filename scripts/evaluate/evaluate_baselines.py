"""
Evaluate Baseline Models

Evaluates Vanilla BERT and MC Dropout baselines on test data.

Usage:
    python scripts/evaluate/evaluate_baselines.py \
        --checkpoint experiments/baselines/squad/vanilla_bert_best.pt \
        --baseline vanilla \
        --dataset squad

Author: Elias Hossain
"""

import argparse
from pathlib import Path


import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import pandas as pd
import json

from transformers import AutoModel

from uat_lite.train_data import get_dataset
from uat_lite.metrics import training_summary as compute_all_metrics
from uat_lite.baselines import MCDropoutBERT, VanillaBERT
from uat_lite.utils import to_native as convert_to_native


def evaluate_baseline(model, data_loader, device):
    """Evaluate baseline model"""
    model.eval()
    
    all_probabilities = []
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask, training_mode=False) if isinstance(model, MCDropoutBERT) else model(input_ids, attention_mask)
            
            all_probabilities.append(outputs['probabilities'].cpu().numpy())
            all_predictions.append(outputs['predictions'].cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    # Concatenate
    probabilities = np.concatenate(all_probabilities, axis=0)
    predictions = np.concatenate(all_predictions, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    # Get confidences (max probability)
    confidences = np.max(probabilities, axis=1)
    
    # Compute metrics
    metrics = compute_all_metrics(
        probabilities=probabilities,
        predictions=predictions,
        labels=labels,
        confidences=confidences,
        num_bins=15
    )
    
    return metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--baseline', type=str, required=True,
                       choices=['vanilla', 'mc_dropout'])
    parser.add_argument('--dataset', type=str, default='squad')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_length', type=int, default=384)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default='results/baselines')
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("\n" + "="*60)
    print(f"Evaluating {args.baseline.upper()} Baseline")
    print("="*60 + "\n")
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"✓ Device: {device}")
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint['config']
    
    # Initialize model
    if args.baseline == 'vanilla':
        model = VanillaBERT(config['model_name'], config['num_classes'])
    elif args.baseline == 'mc_dropout':
        model = MCDropoutBERT(config['model_name'], config['num_classes'], mc_samples=10)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print("✓ Model loaded")
    
    # Load dataset
    print(f"\nLoading {args.dataset.upper()} dataset...")
    _, val_loader, _ = get_dataset(
        dataset_name=args.dataset,
        tokenizer_name=config['model_name'],
        max_length=args.max_length,
        batch_size=args.batch_size,
        num_workers=4
    )
    
    # Evaluate
    print("\n" + "="*60)
    print("Evaluation")
    print("="*60 + "\n")
    
    metrics = evaluate_baseline(model, val_loader, device)
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60 + "\n")
    
    print("Calibration Metrics:")
    print(f"  ECE:         {metrics['ece']:.4f}")
    print(f"  Brier Score: {metrics['brier']:.4f}")
    print(f"  NLL:         {metrics['nll']:.4f}")
    print(f"  AURC:        {metrics['aurc']:.4f}")
    
    print("\nAccuracy Metrics:")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  F1 Score:    {metrics['f1']:.4f}")
    
    # Save results
    output_dir = Path(args.output_dir) / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_native = convert_to_native(metrics)
    
    # Save JSON
    metrics_file = output_dir / f'{args.baseline}_metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics_native, f, indent=2)
    print(f"\n✓ Metrics saved to: {metrics_file}")
    
    # Save CSV
    df = pd.DataFrame([metrics_native])
    csv_file = output_dir / f'{args.baseline}_results.csv'
    df.to_csv(csv_file, index=False)
    print(f"✓ Results saved to: {csv_file}")
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)


if __name__ == '__main__':
    main()