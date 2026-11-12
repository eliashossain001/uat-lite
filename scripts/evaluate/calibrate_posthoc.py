"""
Post-Hoc Calibration Methods (FIXED)

Implements:
1. Temperature Scaling
2. Isotonic Regression

Author: Elias Hossain
"""

import argparse
from pathlib import Path


import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.isotonic import IsotonicRegression
import json

from uat_lite.train_data import get_dataset
from uat_lite.metrics import training_summary as compute_all_metrics
from transformers import AutoModel
from uat_lite.baselines import VanillaBERT
from uat_lite.utils import to_native as convert_to_native


class TemperatureScaling(nn.Module):
    """Temperature Scaling for calibration"""
    
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
    
    def forward(self, logits):
        return logits / self.temperature
    
    def fit(self, logits, labels, max_iter=50, lr=0.01):
        """Optimize temperature on validation set"""
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        criterion = nn.CrossEntropyLoss()
        
        def eval_loss():
            optimizer.zero_grad()
            loss = criterion(self.forward(logits), labels)
            loss.backward()
            return loss
        
        optimizer.step(eval_loss)
        return self.temperature.item()


class IsotonicCalibration:
    """Isotonic Regression for calibration"""
    
    def __init__(self):
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
    
    def fit(self, confidences, labels):
        """Fit isotonic regression on validation set"""
        self.calibrator.fit(confidences, labels)
    
    def predict(self, confidences):
        """Apply calibration to confidence scores"""
        return self.calibrator.predict(confidences)


def collect_logits_and_labels(model, data_loader, device):
    """Collect all logits and labels from validation set"""
    model.eval()
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Collecting predictions"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids, attention_mask)['logits']
            
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
    
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    return all_logits, all_labels


def apply_temperature_scaling(checkpoint_path, data_loader, device, config):
    """Apply temperature scaling calibration (FIXED)"""
    print("\n" + "="*60)
    print("Temperature Scaling Calibration")
    print("="*60 + "\n")
    
    # Load model
    model = VanillaBERT(config['model_name'], config['num_classes'])
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print("✓ Model loaded")
    
    # Collect validation predictions
    print("\nCollecting validation predictions...")
    val_logits, val_labels = collect_logits_and_labels(model, data_loader, device)
    val_logits = val_logits.to(device)
    val_labels = val_labels.to(device)
    
    # Fit temperature
    print("\nOptimizing temperature...")
    temp_scaler = TemperatureScaling().to(device)
    optimal_temp = temp_scaler.fit(val_logits, val_labels)
    
    print(f"✓ Optimal temperature: {optimal_temp:.4f}")
    
    # Apply temperature scaling (with no_grad for inference)
    with torch.no_grad():
        scaled_logits = temp_scaler(val_logits)
        probabilities = torch.softmax(scaled_logits, dim=-1).cpu().numpy()
        predictions = torch.argmax(scaled_logits, dim=-1).cpu().numpy()
    
    labels = val_labels.cpu().numpy()
    confidences = np.max(probabilities, axis=1)
    
    # Compute metrics
    metrics = compute_all_metrics(probabilities, predictions, labels, confidences)
    
    return metrics, optimal_temp


def apply_isotonic_regression(checkpoint_path, data_loader, device, config):
    """Apply isotonic regression calibration"""
    print("\n" + "="*60)
    print("Isotonic Regression Calibration")
    print("="*60 + "\n")
    
    # Load model
    model = VanillaBERT(config['model_name'], config['num_classes'])
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print("✓ Model loaded")
    
    # Collect validation predictions
    print("\nCollecting validation predictions...")
    val_logits, val_labels = collect_logits_and_labels(model, data_loader, device)
    
    # Get probabilities and confidences
    probabilities = torch.softmax(val_logits, dim=-1).numpy()
    predictions = np.argmax(probabilities, axis=1)
    confidences = np.max(probabilities, axis=1)
    labels = val_labels.numpy()
    
    # Create binary correctness labels
    correctness = (predictions == labels).astype(float)
    
    # Fit isotonic regression
    print("\nFitting isotonic regression...")
    iso_calibrator = IsotonicCalibration()
    iso_calibrator.fit(confidences, correctness)
    
    print("✓ Isotonic regression fitted")
    
    # Apply calibration (adjust confidences)
    calibrated_confidences = iso_calibrator.predict(confidences)
    
    # Adjust probabilities proportionally
    confidence_ratio = calibrated_confidences / (confidences + 1e-10)
    calibrated_probabilities = probabilities * confidence_ratio[:, np.newaxis]
    
    # Renormalize
    calibrated_probabilities = calibrated_probabilities / calibrated_probabilities.sum(axis=1, keepdims=True)
    
    # Compute metrics
    metrics = compute_all_metrics(
        calibrated_probabilities, 
        predictions, 
        labels, 
        calibrated_confidences
    )
    
    return metrics, iso_calibrator


def parse_args():
    parser = argparse.ArgumentParser(description='Post-hoc Calibration')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--method', type=str, required=True,
                       choices=['temperature_scaling', 'isotonic_regression'],
                       help='Calibration method')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['squad', 'squad_v2', 'mnli', 'sst2'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_length', type=int, default=384)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default='results/baselines')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("\n" + "="*60)
    print(f"Post-Hoc Calibration: {args.method.upper()}")
    print("="*60 + "\n")
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"✓ Device: {device}")
    
    # Load checkpoint config
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint['config']
    
    # Load validation dataset
    print(f"\nLoading {args.dataset.upper()} validation set...")
    _, val_loader, _ = get_dataset(
        dataset_name=args.dataset,
        tokenizer_name=config['model_name'],
        max_length=args.max_length,
        batch_size=args.batch_size,
        num_workers=4
    )
    
    # Apply calibration
    if args.method == 'temperature_scaling':
        metrics, extra_info = apply_temperature_scaling(
            args.checkpoint, val_loader, device, config
        )
        method_name = 'temperature_scaling'
        print(f"\nOptimal Temperature: {extra_info:.4f}")
    else:
        metrics, extra_info = apply_isotonic_regression(
            args.checkpoint, val_loader, device, config
        )
        method_name = 'isotonic_regression'
    
    # Print results
    print("\n" + "="*60)
    print("CALIBRATION RESULTS")
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
    metrics_file = output_dir / f'{method_name}_metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics_native, f, indent=2)
    print(f"\n✓ Results saved to: {metrics_file}")
    
    print("\n" + "="*60)
    print("CALIBRATION COMPLETE!")
    print("="*60)


if __name__ == '__main__':
    main()