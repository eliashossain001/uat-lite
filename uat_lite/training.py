"""Training loop for UAT-LITE: gradient accumulation, early stopping on validation
loss, best/last checkpointing. Optimizer and scheduler are supplied by the caller."""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import json
import time


class Trainer:
    """
    Trainer for UAT-LITE
    
    Args:
        model: UATLite model
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on
        output_dir: Directory to save checkpoints
        max_epochs: Maximum number of epochs
        early_stopping_patience: Patience for early stopping
        gradient_accumulation_steps: Steps to accumulate gradients
        max_grad_norm: Maximum gradient norm for clipping
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer,
        scheduler,
        device: str = 'cuda',
        output_dir: str = 'experiments',
        max_epochs: int = 3,
        early_stopping_patience: int = 3,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        log_interval: int = 100,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = Path(output_dir)
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.log_interval = log_interval
        
        # Create output directories
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.log_dir = self.output_dir / 'logs'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
        # History
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rate': []
        }
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}/{self.max_epochs}")
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_uncertainty=False,  # Skip uncertainty during training for speed
                return_layer_decomposition=False
            )
            
            loss = outputs['loss']
            
            # Normalize loss for gradient accumulation
            loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation
            if (step + 1) % self.gradient_accumulation_steps == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.max_grad_norm
                )
                
                # Update weights
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                self.global_step += 1
            
            # Track metrics
            total_loss += loss.item() * self.gradient_accumulation_steps
            predictions = outputs['predictions']
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            
            # Update progress bar
            current_lr = self.scheduler.get_last_lr()[0]
            progress_bar.set_postfix({
                'loss': f"{total_loss / (step + 1):.4f}",
                'acc': f"{total_correct / total_samples:.4f}",
                'lr': f"{current_lr:.2e}"
            })
            
            # Log periodically
            if (step + 1) % self.log_interval == 0:
                self.log_step(step, total_loss / (step + 1), current_lr)
        
        # Epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        avg_accuracy = total_correct / total_samples
        
        return {
            'loss': avg_loss,
            'accuracy': avg_accuracy
        }
    
    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    return_uncertainty=False,
                    return_layer_decomposition=False
                )
                
                # Track metrics
                total_loss += outputs['loss'].item()
                predictions = outputs['predictions']
                total_correct += (predictions == labels).sum().item()
                total_samples += labels.size(0)
        
        # Compute averages
        avg_loss = total_loss / len(self.val_loader)
        avg_accuracy = total_correct / total_samples
        
        return {
            'loss': avg_loss,
            'accuracy': avg_accuracy
        }
    
    def save_checkpoint(self, filename: str = 'checkpoint.pt', is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history,
            'config': self.model.get_config()
        }
        
        save_path = self.checkpoint_dir / filename
        torch.save(checkpoint, save_path)
        print(f"✓ Checkpoint saved: {save_path}")
        
        # Save as best model if applicable
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"✓ Best model saved: {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']
        
        print(f"✓ Checkpoint loaded from: {checkpoint_path}")
    
    def log_step(self, step: int, loss: float, lr: float):
        """Log training step"""
        log_data = {
            'epoch': self.current_epoch,
            'step': step,
            'global_step': self.global_step,
            'loss': loss,
            'learning_rate': lr,
            'timestamp': time.time()
        }
        
        # Save to log file
        log_file = self.log_dir / f'train_log_epoch_{self.current_epoch}.jsonl'
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_data) + '\n')
    
    def log_epoch(self, train_metrics: Dict, val_metrics: Dict):
        """Log epoch results"""
        print(f"\nEpoch {self.current_epoch + 1}/{self.max_epochs} Results:")
        print(f"  Train Loss: {train_metrics['loss']:.4f}")
        print(f"  Train Acc:  {train_metrics['accuracy']:.4f}")
        print(f"  Val Loss:   {val_metrics['loss']:.4f}")
        print(f"  Val Acc:    {val_metrics['accuracy']:.4f}")
        
        # Update history
        self.history['train_loss'].append(train_metrics['loss'])
        self.history['val_loss'].append(val_metrics['loss'])
        self.history['val_accuracy'].append(val_metrics['accuracy'])
        self.history['learning_rate'].append(self.scheduler.get_last_lr()[0])
        
        # Save history
        history_file = self.log_dir / 'history.json'
        with open(history_file, 'w') as f:
            json.dumps(self.history, indent=2)
    
    def check_early_stopping(self, val_loss: float) -> bool:
        """Check if training should stop early"""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.epochs_without_improvement = 0
            return False
        else:
            self.epochs_without_improvement += 1
            if self.epochs_without_improvement >= self.early_stopping_patience:
                print(f"\n⚠ Early stopping triggered! No improvement for {self.early_stopping_patience} epochs.")
                return True
            return False
    
    def train(self):
        """Main training loop"""
        print("\n" + "="*60)
        print("Starting Training")
        print("="*60 + "\n")
        
        start_time = time.time()
        
        for epoch in range(self.max_epochs):
            self.current_epoch = epoch
            
            # Train for one epoch
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Log results
            self.log_epoch(train_metrics, val_metrics)
            
            # Save checkpoint
            is_best = val_metrics['loss'] < self.best_val_loss
            self.save_checkpoint(
                filename=f'checkpoint_epoch_{epoch+1}.pt',
                is_best=is_best
            )
            
            # Check early stopping
            if self.check_early_stopping(val_metrics['loss']):
                break
        
        # Training complete
        elapsed_time = time.time() - start_time
        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60)
        print(f"Total time: {elapsed_time/3600:.2f} hours")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Best model saved at: {self.checkpoint_dir / 'best_model.pt'}")
        
        return self.history
