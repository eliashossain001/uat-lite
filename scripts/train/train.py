"""
Complete Training Script for UAT-LITE (MEMORY OPTIMIZED)

Author: Elias Hossain
"""

import argparse
from pathlib import Path


import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import random
import numpy as np

from uat_lite.models import UATLite
from uat_lite.train_data import get_dataset
from uat_lite.training import Trainer
from uat_lite.utils import set_seed


def parse_args():
    """Parse arguments"""
    parser = argparse.ArgumentParser(description='Train UAT-LITE')
    
    # Dataset
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['squad', 'squad_v2', 'mnli', 'sst2', 'medqa'])
    
    # Model
    parser.add_argument('--model', type=str, default='bert-base-uncased')
    parser.add_argument('--num_classes', type=int, default=None)
    parser.add_argument('--mc_samples', type=int, default=5,
                       help='MC samples (reduced from 10 for memory)')
    parser.add_argument('--dropout_rate', type=float, default=0.3)
    parser.add_argument('--uncertainty_penalty', type=float, default=0.5)
    parser.add_argument('--confidence_threshold', type=float, default=0.7)
    parser.add_argument('--no_layer_decomposition', action='store_true',
                       help='Disable the layer-wise variance decomposition to save memory')
    
    # Training
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Reduced for memory')
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--max_length', type=int, default=384,
                       help='Reduced from 512 for memory')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                       help='Effective batch = batch_size * this')
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--early_stopping_patience', type=int, default=3)
    
    # Output
    parser.add_argument('--output_dir', type=str, default=None)
    
    # Hardware
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Reproducibility
    parser.add_argument('--seed', type=int, default=42)
    
    # Debug
    parser.add_argument('--debug', action='store_true')
    
    return parser.parse_args()


def get_num_classes(dataset_name: str) -> int:
    """Auto-detect classes"""
    num_classes_map = {
        'squad': 2,
        'squad_v2': 2,
        'mnli': 3,
        'sst2': 2,
        'medqa': 4,
    }
    return num_classes_map.get(dataset_name, 2)


def main():
    """Main function"""
    args = parse_args()
    
    print("\n" + "="*60)
    print("UAT-LITE Training (Memory Optimized)")
    print("="*60 + "\n")
    
    # Set seed
    set_seed(args.seed)
    print(f"✓ Seed: {args.seed}")
    
    # Device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠ CUDA not available, using CPU")
        args.device = 'cpu'
    device = torch.device(args.device)
    print(f"✓ Device: {device}")
    
    # Clear GPU cache
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Auto-detect classes
    if args.num_classes is None:
        args.num_classes = get_num_classes(args.dataset)
    print(f"✓ Classes: {args.num_classes}")
    
    # Output dir
    if args.output_dir is None:
        args.output_dir = f"experiments/{args.dataset}/{args.model.replace('/', '_')}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Output: {output_dir}")
    
    # Load dataset
    print(f"\n{'='*60}")
    print(f"Loading {args.dataset.upper()}...")
    print(f"{'='*60}\n")
    
    train_loader, val_loader, test_loader = get_dataset(
        dataset_name=args.dataset,
        tokenizer_name=args.model,
        max_length=args.max_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Debug mode
    if args.debug:
        print("⚠ DEBUG MODE")
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(train_loader.dataset, range(100)),
            batch_size=args.batch_size,
            shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(val_loader.dataset, range(50)),
            batch_size=args.batch_size*2,
            shuffle=False
        )
    
    # Initialize model
    print(f"\n{'='*60}")
    print("Initializing Model...")
    print(f"{'='*60}\n")
    
    model = UATLite(
        base_model_name=args.model,
        num_classes=args.num_classes,
        mc_samples=args.mc_samples,
        dropout_rate=args.dropout_rate,
        uncertainty_penalty=args.uncertainty_penalty,
        confidence_threshold=args.confidence_threshold,
        use_layer_decomposition=not args.no_layer_decomposition
    )
    
    print(f"✓ Model: {args.model}")
    print(f"✓ Hidden size: {model.hidden_size}")
    print(f"✓ Layers: {model.num_layers}")
    print(f"✓ MC samples: {model.mc_samples}")
    print(f"✓ Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Scheduler
    total_steps = len(train_loader) * args.epochs // args.gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )
    
    print(f"\n✓ Optimizer: AdamW (lr={args.learning_rate})")
    print(f"✓ Scheduler: Linear warmup ({args.warmup_steps} steps)")
    print(f"✓ Total steps: {total_steps}")
    print(f"✓ Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    
    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=args.output_dir,
        max_epochs=args.epochs,
        early_stopping_patience=args.early_stopping_patience,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        log_interval=100
    )
    
    # Train
    history = trainer.train()
    
    # Save config
    import json
    config_file = output_dir / 'model_config.json'
    with open(config_file, 'w') as f:
        json.dump({
            'model_name': args.model,
            'num_classes': args.num_classes,
            'mc_samples': args.mc_samples,
            'dropout_rate': args.dropout_rate,
            'dataset': args.dataset,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'epochs': args.epochs,
            'seed': args.seed,
        }, f, indent=2)
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"\nBest checkpoint: {output_dir}/checkpoints/best_model.pt")
    print(f"\n🎉 Evaluate with:")
    print(f"   python scripts/evaluate/evaluate.py --checkpoint {output_dir}/checkpoints/best_model.pt --dataset {args.dataset}")


if __name__ == '__main__':
    main()