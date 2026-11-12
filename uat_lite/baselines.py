"""Baseline classifiers: a plain fine-tuned BERT and MC dropout on the output layer only.
Both return {logits, probabilities, predictions[, loss]}."""

import torch
import torch.nn as nn
from transformers import AutoModel


class VanillaBERT(nn.Module):
    """Baseline 1: Standard BERT without uncertainty"""
    
    def __init__(self, model_name: str, num_classes: int):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        self.num_classes = num_classes
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        
        result = {
            'logits': logits,
            'probabilities': torch.softmax(logits, dim=-1),
            'predictions': torch.argmax(logits, dim=-1)
        }
        
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            result['loss'] = loss
        
        return result


class MCDropoutBERT(nn.Module):
    """Baseline 2: MC Dropout (output layer only)"""
    
    def __init__(self, model_name: str, num_classes: int, mc_samples: int = 10, dropout_rate: float = 0.3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        self.mc_samples = mc_samples
        self.num_classes = num_classes
    
    def forward(self, input_ids, attention_mask=None, labels=None, training_mode=True):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]
        
        if training_mode or self.training:
            # During training: standard forward
            pooled = self.dropout(pooled)
            logits = self.classifier(pooled)
        else:
            # During evaluation: MC sampling
            logits_samples = []
            self.dropout.train()  # Enable dropout
            for _ in range(self.mc_samples):
                dropped = self.dropout(pooled)
                logits = self.classifier(dropped)
                logits_samples.append(logits)
            logits = torch.stack(logits_samples, dim=0).mean(dim=0)
            self.dropout.eval()
        
        result = {
            'logits': logits,
            'probabilities': torch.softmax(logits, dim=-1),
            'predictions': torch.argmax(logits, dim=-1)
        }
        
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            result['loss'] = loss
        
        return result
