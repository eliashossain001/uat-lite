"""
SWAG: Stochastic Weight Averaging-Gaussian

Implements SWAG for uncertainty quantification.

Reference: https://arxiv.org/abs/1902.02476

Usage:
    python scripts/train/train_swag.py \
        --dataset squad \
        --num_classes 2 \
        --epochs 3

Author: Elias Hossain
"""

import argparse
from pathlib import Path


import torch
import torch.nn as nn
from torch.optim import SGD
from transformers import AutoModel
from tqdm import tqdm
import copy
import json

from uat_lite.train_data import get_dataset
from uat_lite.baselines import VanillaBERT


class SWAG:
    """
    Stochastic Weight Averaging-Gaussian
    
    Collects weight snapshots during training and fits Gaussian.
    """
    def __init__(self, model, max_snapshots=20):
        self.model = model
        self.max_snapshots = max_snapshots
        self.snapshots = []
        self.mean_params = None
        self.sq_mean_params = None
        self.n_snapshots = 0
    
    def collect_snapshot(self):
        """Collect current model weights as snapshot"""
        snapshot = {}
        for name, param in self.model.named_parameters():
            snapshot[name] = param.data.clone()
        
        self.snapshots.append(snapshot)
        
        # Keep only last max_snapshots
        if len(self.snapshots) > self.max_snapshots:
            self.snapshots.pop(0)
        
        # Update running statistics
        if self.mean_params is None:
            self.mean_params = copy.deepcopy(snapshot)
            self.sq_mean_params = {k: v**2 for k, v in snapshot.items()}
            self.n_snapshots = 1
        else:
            self.n_snapshots += 1
            for name in self.mean_params:
                self.mean_params[name] = (
                    (self.n_snapshots - 1) * self.mean_params[name] + snapshot[name]
                ) / self.n_snapshots
                self.sq_mean_params[name] = (
                    (self.n_snapshots - 1) * self.sq_mean_params[name] + snapshot[name]**2
                ) / self.n_snapshots
    
    def sample_parameters(self, scale=1.0):
        """Sample parameters from SWAG posterior"""
        if self.mean_params is None:
            return
        
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                mean = self.mean_params[name]
                var = self.sq_mean_params[name] - mean**2
                std = torch.sqrt(var + 1e-8)
                
                # Sample from Gaussian
                noise = torch.randn_like(param) * std * scale
                param.copy_(mean + noise)


def train_swag(args):
    """Train model with SWAG"""
    print("\n" + "="*60)
    print("Training SWAG")
    print("="*60 + "\n")
    
    device = torch.device(args.device)
    
    # Load data
    print(f"Loading {args.dataset.upper()} dataset...")
    train_loader, val_loader, _ = get_dataset(
        dataset_name=args.dataset,
        tokenizer_name=args.model,
        max_length=args.max_length,
        batch_size=args.batch_size,
        num_workers=4
    )
    
    # Initialize model
    print("Initializing model...")
    model = VanillaBERT(args.model, args.num_classes).to(device)
    
    # Optimizer (SGD for SWAG)
    optimizer = SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
    
    # SWAG
    swag = SWAG(model, max_snapshots=20)
    
    print(f"Total training steps: {len(train_loader) * args.epochs}")
    
    # Training loop
    best_val_loss = float('inf')
    snapshot_freq = len(train_loader) // 5  # Collect 5 snapshots per epoch
    
    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*60}")
        
        # Training
        model.train()
        train_loss = 0
        
        for step, batch in enumerate(tqdm(train_loader, desc="Training")):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask, labels)
            loss = outputs['loss']
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            train_loss += loss.item()
            
            # Collect SWAG snapshot
            if (step + 1) % snapshot_freq == 0:
                swag.collect_snapshot()
                print(f"  Collected snapshot (total: {swag.n_snapshots})")
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask, labels)
                val_loss += outputs['loss'].item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"\nResults:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f}")
        print(f"  SWAG Snapshots: {swag.n_snapshots}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_dir = Path(args.output_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Save SWAG statistics
            save_path = save_dir / 'swag_best.pt'
            torch.save({
                'model_state_dict': model.state_dict(),
                'swag_mean': swag.mean_params,
                'swag_sq_mean': swag.sq_mean_params,
                'n_snapshots': swag.n_snapshots,
                'config': {
                    'model_name': args.model,
                    'num_classes': args.num_classes,
                }
            }, save_path)
            print(f"✓ Best model saved: {save_path}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    
    return save_path


def parse_args():
    parser = argparse.ArgumentParser(description='Train SWAG')
    
    parser.add_argument('--dataset', type=str, default='squad')
    parser.add_argument('--model', type=str, default='bert-base-uncased')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--max_length', type=int, default=384)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default='experiments/baselines')
    
    return parser.parse_args()


def main():
    args = parse_args()
    train_swag(args)


if __name__ == '__main__':
    main()