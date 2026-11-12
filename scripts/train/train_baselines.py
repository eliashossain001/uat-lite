"""Train the baselines: vanilla BERT (no uncertainty), output-layer MC dropout and
deep-ensemble members, with per-seed runs."""

import argparse
from pathlib import Path


import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoModel, AutoConfig, get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
import json

from uat_lite.train_data import get_dataset
from uat_lite.baselines import MCDropoutBERT, VanillaBERT


def train_vanilla_bert(args):
    """Train Baseline 1: Vanilla BERT"""
    print(f"\n{'='*60}")
    print("Training Baseline 1: Vanilla BERT")
    print(f"{'='*60}\n")
    
    # SET RANDOM SEED FOR REPRODUCIBILITY
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    print(f"Random seed: {args.seed}")
    
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
    print("Initializing Vanilla BERT...")
    model = VanillaBERT(args.model, args.num_classes).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model initialized ({total_params:,} parameters)")
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    
    # Scheduler
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=500, 
        num_training_steps=total_steps
    )
    
    print(f"Total training steps: {total_steps}")
    
    # Training loop
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
    
    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*60}")
        
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask, labels)
            loss = outputs['loss']
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            train_loss += loss.item()
            train_correct += (outputs['predictions'] == labels).sum().item()
            train_total += labels.size(0)
        
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask, labels)
                val_loss += outputs['loss'].item()
                
                val_correct += (outputs['predictions'] == labels).sum().item()
                val_total += labels.size(0)
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / val_total
        
        print(f"\nResults:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f}, Val Acc:   {val_accuracy:.4f}")
        
        # Save history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_accuracy'].append(val_accuracy)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_dir = Path(args.output_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            save_path = save_dir / 'vanilla_bert_best.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'val_accuracy': val_accuracy,
                'seed': args.seed,
                'config': {
                    'model_name': args.model,
                    'num_classes': args.num_classes,
                    'baseline': 'vanilla_bert'
                }
            }, save_path)
            print(f"✓ Best model saved: {save_path}")
    
    # Save training history
    history_file = save_dir / 'vanilla_bert_history.json'
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved: {save_path}")
    
    return save_path


def train_mc_dropout(args):
    """Train Baseline 2: MC Dropout"""
    print(f"\n{'='*60}")
    print("Training Baseline 2: MC Dropout (Output Only)")
    print(f"{'='*60}\n")
    
    # SET RANDOM SEED FOR REPRODUCIBILITY
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    print(f"Random seed: {args.seed}")
    
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
    print("Initializing MC Dropout BERT...")
    model = MCDropoutBERT(
        args.model, 
        args.num_classes, 
        mc_samples=10, 
        dropout_rate=0.3
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model initialized ({total_params:,} parameters)")
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    
    # Scheduler
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=500, 
        num_training_steps=total_steps
    )
    
    print(f"Total training steps: {total_steps}")
    
    # Training loop
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
    
    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*60}")
        
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask, labels, training_mode=True)
            loss = outputs['loss']
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            train_loss += loss.item()
            train_correct += (outputs['predictions'] == labels).sum().item()
            train_total += labels.size(0)
        
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / train_total
        
        # Validation with MC sampling
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating (MC)"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask, labels, training_mode=False)
                val_loss += outputs['loss'].item()
                
                val_correct += (outputs['predictions'] == labels).sum().item()
                val_total += labels.size(0)
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / val_total
        
        print(f"\nResults:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f}, Val Acc:   {val_accuracy:.4f}")
        
        # Save history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_accuracy'].append(val_accuracy)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_dir = Path(args.output_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            save_path = save_dir / 'mc_dropout_best.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'val_accuracy': val_accuracy,
                'seed': args.seed,
                'config': {
                    'model_name': args.model,
                    'num_classes': args.num_classes,
                    'mc_samples': 10,
                    'baseline': 'mc_dropout'
                }
            }, save_path)
            print(f"✓ Best model saved: {save_path}")
    
    # Save training history
    history_file = save_dir / 'mc_dropout_history.json'
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved: {save_path}")
    
    return save_path


def parse_args():
    parser = argparse.ArgumentParser(description='Train Baseline Models')
    
    parser.add_argument('--baseline', type=str, required=True,
                       choices=['vanilla', 'mc_dropout', 'deep_ensemble'],
                       help='Which baseline to train')
    parser.add_argument('--dataset', type=str, default='squad',
                       choices=['squad', 'squad_v2', 'mnli', 'sst2', 'medqa'])
    parser.add_argument('--model', type=str, default='bert-base-uncased')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--max_length', type=int, default=384)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default='experiments/baselines')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.baseline == 'vanilla':
        train_vanilla_bert(args)
    elif args.baseline == 'mc_dropout':
        train_mc_dropout(args)
    elif args.baseline == 'deep_ensemble':
        print("\n" + "="*60)
        print("Deep Ensemble Training")
        print("="*60)
        print("\nDeep Ensemble requires training 5 separate models.")
        print("Run the following commands:\n")
        for i in range(1, 6):
            print(f"# Model {i}:")
            print(f"python scripts/train/train_baselines.py \\")
            print(f"    --baseline vanilla \\")
            print(f"    --dataset {args.dataset} \\")
            print(f"    --seed {42 + i} \\")
            print(f"    --output_dir experiments/baselines/{args.dataset}/ensemble_{i}\n")
    else:
        raise ValueError(f"Unknown baseline: {args.baseline}")


if __name__ == '__main__':
    main()