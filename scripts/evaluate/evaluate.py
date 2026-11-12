"""
Evaluation Script for UAT-LITE

Computes all metrics and generates result tables.

Usage:
    python scripts/evaluate/evaluate.py \
        --checkpoint experiments/squad/bert-base-uncased/checkpoints/best_model.pt \
        --dataset squad
    
Author: Elias Hossain
"""

import argparse
from pathlib import Path


import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
import json

from uat_lite.models import UATLite
from uat_lite.train_data import get_dataset
from uat_lite.metrics import training_summary as compute_all_metrics
from uat_lite.utils import to_native as convert_to_native


def parse_args():
    """Parse arguments"""
    parser = argparse.ArgumentParser(description='Evaluate UAT-LITE')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['squad', 'squad_v2', 'mnli', 'sst2', 'medqa'])
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Evaluation batch size')
    parser.add_argument('--max_length', type=int, default=384)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--num_workers', type=int, default=4)
    
    return parser.parse_args()


def evaluate_model(
    model,
    data_loader,
    device,
    return_layer_decomposition: bool = False
):
    """
    Evaluate model on dataset
    
    Returns:
        Dictionary with predictions and metrics
    """
    model.eval()
    
    all_probabilities = []
    all_predictions = []
    all_labels = []
    all_confidences = []
    all_epistemic = []
    all_aleatoric = []
    all_abstained = []
    
    # Layer-wise uncertainties (if requested)
    all_layer_uncertainties = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass with uncertainty
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_uncertainty=True,
                return_layer_decomposition=return_layer_decomposition
            )
            
            # Collect outputs
            all_probabilities.append(outputs['probabilities'].cpu().numpy())
            all_predictions.append(outputs['predictions'].cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_confidences.append(outputs['confidence'].cpu().numpy())
            all_epistemic.append(outputs['epistemic_uncertainty'].cpu().numpy())
            all_aleatoric.append(outputs['aleatoric_uncertainty'].cpu().numpy())
            all_abstained.append(outputs['should_abstain'].cpu().numpy())
            
            if return_layer_decomposition and 'layer_uncertainties' in outputs:
                all_layer_uncertainties.append(outputs['layer_uncertainties'].cpu().numpy())
    
    # Concatenate all batches
    probabilities = np.concatenate(all_probabilities, axis=0)
    predictions = np.concatenate(all_predictions, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    confidences = np.concatenate(all_confidences, axis=0)
    epistemic = np.concatenate(all_epistemic, axis=0)
    aleatoric = np.concatenate(all_aleatoric, axis=0)
    abstained = np.concatenate(all_abstained, axis=0)
    
    # Compute metrics
    metrics = compute_all_metrics(
        probabilities=probabilities,
        predictions=predictions,
        labels=labels,
        confidences=confidences,
        num_bins=15
    )
    
    # Add uncertainty metrics
    metrics['mean_epistemic'] = float(epistemic.mean())
    metrics['mean_aleatoric'] = float(aleatoric.mean())
    metrics['abstention_rate'] = float(abstained.mean())
    
    result = {
        'metrics': metrics,
        'probabilities': probabilities,
        'predictions': predictions,
        'labels': labels,
        'confidences': confidences,
        'epistemic': epistemic,
        'aleatoric': aleatoric,
        'abstained': abstained,
    }
    
    if all_layer_uncertainties:
        result['layer_uncertainties'] = np.concatenate(all_layer_uncertainties, axis=0)
    
    return result


def main():
    """Main evaluation"""
    args = parse_args()
    
    print("\n" + "="*60)
    print("UAT-LITE Evaluation")
    print("="*60 + "\n")
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"✓ Device: {device}")
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    model = UATLite.from_pretrained(args.checkpoint, device=str(device))
    model.eval()
    print("✓ Model loaded")
    
    # Get model config
    config = model.get_config()
    print(f"\nModel Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Load dataset
    print(f"\n{'='*60}")
    print(f"Loading {args.dataset.upper()} dataset...")
    print(f"{'='*60}\n")
    
    _, val_loader, test_loader = get_dataset(
        dataset_name=args.dataset,
        tokenizer_name=config['base_model_name'],
        max_length=args.max_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Evaluate on validation set
    print(f"\n{'='*60}")
    print("Evaluating on Validation Set")
    print(f"{'='*60}\n")
    
    val_results = evaluate_model(
        model, val_loader, device,
        return_layer_decomposition=False
    )
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60 + "\n")
    
    metrics = val_results['metrics']
    
    print("Calibration Metrics:")
    print(f"  ECE:         {metrics['ece']:.4f}")
    print(f"  Brier Score: {metrics['brier']:.4f}")
    print(f"  NLL:         {metrics['nll']:.4f}")
    print(f"  AURC:        {metrics['aurc']:.4f}")
    
    print("\nAccuracy Metrics:")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  F1 Score:    {metrics['f1']:.4f}")
    
    print("\nSelective Prediction:")
    print(f"  Acc @ 70%:   {metrics['selective_acc_70']:.4f}")
    print(f"  Acc @ 80%:   {metrics['selective_acc_80']:.4f}")
    print(f"  Acc @ 90%:   {metrics['selective_acc_90']:.4f}")
    print(f"  Acc @ 100%:  {metrics['selective_acc_100']:.4f}")
    
    print("\nUncertainty Statistics:")
    print(f"  Epistemic:   {metrics['mean_epistemic']:.4f}")
    print(f"  Aleatoric:   {metrics['mean_aleatoric']:.4f}")
    print(f"  Abstention:  {metrics['abstention_rate']:.2%}")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert all metrics to native Python types
    metrics_native = convert_to_native(metrics)
    
    # Save metrics as JSON
    metrics_file = output_dir / f'{args.dataset}_metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics_native, f, indent=2)
    print(f"\n✓ Metrics saved to: {metrics_file}")
    
    # Save as CSV table
    df = pd.DataFrame([metrics_native])
    csv_file = output_dir / f'{args.dataset}_results.csv'
    df.to_csv(csv_file, index=False)
    print(f"✓ Results table saved to: {csv_file}")
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)


if __name__ == '__main__':
    main()