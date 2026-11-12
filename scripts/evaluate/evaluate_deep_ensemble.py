"""
Deep Ensemble Evaluation

Evaluates ensemble of 5 models by averaging their predictions.

Usage:
    python scripts/evaluate/evaluate_deep_ensemble.py \
        --dataset squad \
        --num_models 5 \
        --batch_size 32

Author: Elias Hossain
"""

import argparse
import sys
from pathlib import Path


import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import json
from transformers import AutoModel

from uat_lite.train_data import get_dataset
from uat_lite.metrics import training_summary as compute_all_metrics
from uat_lite.baselines import VanillaBERT
from uat_lite.utils import to_native as convert_to_native


def evaluate_ensemble(models, data_loader, device):
    """
    Evaluate ensemble by averaging predictions
    """
    for model in models:
        model.eval()
    
    all_ensemble_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating Ensemble"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Collect predictions from all models
            batch_probs = []
            for model in models:
                logits = model(input_ids, attention_mask)['logits']
                probs = torch.softmax(logits, dim=-1)
                batch_probs.append(probs)
            
            # Average probabilities
            ensemble_probs = torch.stack(batch_probs, dim=0).mean(dim=0)
            
            all_ensemble_probs.append(ensemble_probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    # Concatenate
    probabilities = np.concatenate(all_ensemble_probs, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    # Get predictions and confidences
    predictions = np.argmax(probabilities, axis=1)
    confidences = np.max(probabilities, axis=1)
    
    # Compute metrics
    metrics = compute_all_metrics(probabilities, predictions, labels, confidences)
    
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Deep Ensemble')
    
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['squad', 'squad_v2', 'mnli', 'sst2'])
    parser.add_argument('--num_models', type=int, default=5,
                       help='Number of models in ensemble')
    parser.add_argument('--base_dir', type=str, default='experiments/baselines',
                       help='Base directory containing ensemble models')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_length', type=int, default=384)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default='results/baselines')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("\n" + "="*60)
    print(f"Deep Ensemble Evaluation ({args.num_models} models)")
    print("="*60 + "\n")
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"✓ Device: {device}")
    
    # Load all ensemble models
    print(f"\nLoading {args.num_models} ensemble models...")
    models = []
    base_dir = Path(args.base_dir) / args.dataset
    
    for i in range(1, args.num_models + 1):
        checkpoint_path = base_dir / f"ensemble_{i}" / "vanilla_bert_best.pt"
        
        if not checkpoint_path.exists():
            print(f"✗ Model {i} not found at: {checkpoint_path}")
            print(f"  Please train ensemble models first using:")
            print(f"  ./scripts/train/train_deep_ensemble.sh {args.dataset} <num_classes> <batch_size> <max_length>")
            sys.exit(1)
        
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        config = checkpoint['config']
        
        model = VanillaBERT(config['model_name'], config['num_classes'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        models.append(model)
        print(f"✓ Loaded model {i}/{args.num_models}")
    
    # Load dataset
    print(f"\nLoading {args.dataset.upper()} validation set...")
    _, val_loader, _ = get_dataset(
        dataset_name=args.dataset,
        tokenizer_name=config['model_name'],
        max_length=args.max_length,
        batch_size=args.batch_size,
        num_workers=4
    )
    
    # Evaluate ensemble
    print("\n" + "="*60)
    print("Evaluating Ensemble")
    print("="*60 + "\n")
    
    metrics = evaluate_ensemble(models, val_loader, device)
    
    # Print results
    print("\n" + "="*60)
    print("ENSEMBLE RESULTS")
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
    metrics_file = output_dir / 'deep_ensemble_metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics_native, f, indent=2)
    print(f"\n✓ Results saved to: {metrics_file}")
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)


if __name__ == '__main__':
    main()