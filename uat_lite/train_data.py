"""Training-time data loaders (train / val / test) for SQuAD 2.0 (answerable vs.
unanswerable), MNLI, SST-2 and MedQA. Used by ``scripts/train``."""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from typing import Optional, Dict, Tuple
import numpy as np


class TextClassificationDataset(Dataset):
    """
    Generic dataset for text classification tasks
    """
    
    def __init__(
        self,
        texts: list,
        labels: list,
        tokenizer,
        max_length: int = 512
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def load_squad_v2(
    tokenizer_name: str = 'bert-base-uncased',
    max_length: int = 512,
    batch_size: int = 16,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load SQuAD 2.0 dataset
    
    Task: Question Answering (Answerable vs Unanswerable)
    
    Returns:
        train_loader, val_loader, test_loader
    """
    print("Loading SQuAD 2.0 dataset...")
    
    # Load dataset from HuggingFace
    dataset = load_dataset("squad_v2")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Process dataset
    def process_squad(examples):
        # Combine question and context
        texts = [
            f"Question: {q} Context: {c}" 
            for q, c in zip(examples['question'], examples['context'])
        ]
        
        # Label: 1 if answerable (has answer), 0 if unanswerable
        labels = [1 if len(a['text']) > 0 else 0 for a in examples['answers']]
        
        return {'text': texts, 'label': labels}
    
    # Process splits
    train_data = dataset['train'].map(process_squad, batched=True, remove_columns=dataset['train'].column_names)
    val_data = dataset['validation'].map(process_squad, batched=True, remove_columns=dataset['validation'].column_names)
    
    # Create datasets
    train_dataset = TextClassificationDataset(
        train_data['text'], train_data['label'], tokenizer, max_length
    )
    val_dataset = TextClassificationDataset(
        val_data['text'], val_data['label'], tokenizer, max_length
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size*2, shuffle=False, num_workers=num_workers
    )
    
    print(f"✓ SQuAD 2.0 loaded: {len(train_dataset)} train, {len(val_dataset)} val")
    
    return train_loader, val_loader, val_loader  # Use val as test


def load_mnli(
    tokenizer_name: str = 'bert-base-uncased',
    max_length: int = 256,
    batch_size: int = 32,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load MNLI (Multi-Genre Natural Language Inference) dataset
    
    Task: 3-way classification (entailment, neutral, contradiction)
    
    Returns:
        train_loader, val_loader, test_loader
    """
    print("Loading MNLI dataset...")
    
    # Load dataset
    dataset = load_dataset("glue", "mnli")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Process dataset
    def process_mnli(examples):
        texts = [
            f"Premise: {p} Hypothesis: {h}"
            for p, h in zip(examples['premise'], examples['hypothesis'])
        ]
        return {'text': texts, 'label': examples['label']}
    
    # Process splits
    train_data = dataset['train'].map(process_mnli, batched=True, remove_columns=dataset['train'].column_names)
    val_data = dataset['validation_matched'].map(process_mnli, batched=True, remove_columns=dataset['validation_matched'].column_names)
    test_data = dataset['validation_mismatched'].map(process_mnli, batched=True, remove_columns=dataset['validation_mismatched'].column_names)
    
    # Create datasets
    train_dataset = TextClassificationDataset(
        train_data['text'], train_data['label'], tokenizer, max_length
    )
    val_dataset = TextClassificationDataset(
        val_data['text'], val_data['label'], tokenizer, max_length
    )
    test_dataset = TextClassificationDataset(
        test_data['text'], test_data['label'], tokenizer, max_length
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size*2, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size*2, shuffle=False, num_workers=num_workers
    )
    
    print(f"✓ MNLI loaded: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test")
    
    return train_loader, val_loader, test_loader


def load_sst2(
    tokenizer_name: str = 'bert-base-uncased',
    max_length: int = 128,
    batch_size: int = 32,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load SST-2 (Stanford Sentiment Treebank) dataset
    
    Task: Binary sentiment classification (positive/negative)
    
    Returns:
        train_loader, val_loader, test_loader
    """
    print("Loading SST-2 dataset...")
    
    # Load dataset
    dataset = load_dataset("glue", "sst2")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Process dataset
    train_data = dataset['train']
    val_data = dataset['validation']
    
    # Create datasets
    train_dataset = TextClassificationDataset(
        train_data['sentence'], train_data['label'], tokenizer, max_length
    )
    val_dataset = TextClassificationDataset(
        val_data['sentence'], val_data['label'], tokenizer, max_length
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size*2, shuffle=False, num_workers=num_workers
    )
    
    print(f"✓ SST-2 loaded: {len(train_dataset)} train, {len(val_dataset)} val")
    
    return train_loader, val_loader, val_loader  # Use val as test


def load_medqa(
    tokenizer_name: str = 'emilyalsentzer/Bio_ClinicalBERT',
    max_length: int = 512,
    batch_size: int = 8,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load MedQA dataset
    
    Task: Medical question answering (4-way multiple choice)
    
    Returns:
        train_loader, val_loader, test_loader
    """
    print("Loading MedQA dataset...")
    
    # Load dataset
    dataset = load_dataset("GBaker/MedQA-USMLE-4-options")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Process dataset
    def process_medqa(examples):
        texts = []
        labels = []
        
        for question, options, answer_idx in zip(
            examples['question'], examples['options'], examples['answer_idx']
        ):
            # Combine question with options
            options_text = " ".join([f"({chr(65+i)}) {opt}" for i, opt in enumerate(options)])
            text = f"Question: {question} Options: {options_text}"
            texts.append(text)
            labels.append(answer_idx)
        
        return {'text': texts, 'label': labels}
    
    # Process splits
    train_data = dataset['train'].map(process_medqa, batched=True, remove_columns=dataset['train'].column_names)
    val_data = dataset['validation'].map(process_medqa, batched=True, remove_columns=dataset['validation'].column_names)
    test_data = dataset['test'].map(process_medqa, batched=True, remove_columns=dataset['test'].column_names)
    
    # Create datasets
    train_dataset = TextClassificationDataset(
        train_data['text'], train_data['label'], tokenizer, max_length
    )
    val_dataset = TextClassificationDataset(
        val_data['text'], val_data['label'], tokenizer, max_length
    )
    test_dataset = TextClassificationDataset(
        test_data['text'], test_data['label'], tokenizer, max_length
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size*2, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size*2, shuffle=False, num_workers=num_workers
    )
    
    print(f"✓ MedQA loaded: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test")
    
    return train_loader, val_loader, test_loader


def get_dataset(
    dataset_name: str,
    tokenizer_name: str,
    max_length: int = 512,
    batch_size: int = 16,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Universal dataset loader
    
    Args:
        dataset_name: Name of dataset ('squad', 'mnli', 'sst2', 'medqa', etc.)
        tokenizer_name: HuggingFace tokenizer name
        max_length: Max sequence length
        batch_size: Batch size
        num_workers: Number of data loading workers
        
    Returns:
        train_loader, val_loader, test_loader
    """
    
    if dataset_name == 'squad' or dataset_name == 'squad_v2':
        return load_squad_v2(tokenizer_name, max_length, batch_size, num_workers)
    elif dataset_name == 'mnli':
        return load_mnli(tokenizer_name, max_length, batch_size, num_workers)
    elif dataset_name == 'sst2':
        return load_sst2(tokenizer_name, max_length, batch_size, num_workers)
    elif dataset_name == 'medqa':
        return load_medqa(tokenizer_name, max_length, batch_size, num_workers)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
