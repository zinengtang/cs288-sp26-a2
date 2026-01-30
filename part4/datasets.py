"""
Dataset classes for pre-training and fine-tuning.

Provides:
- PretrainingDataset: For language model pre-training on TinyStories
- MultipleChoiceQADataset: For fine-tuning on multiple choice QA
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple


class PretrainingDataset(Dataset):
    """
    Dataset for language model pre-training.
    
    Loads tokenized sequences and returns fixed-length chunks for training.
    Uses <|endoftext|> as document separator.
    """
    
    def __init__(
        self,
        file_path: str | Path,
        tokenizer,
        max_length: int = 256,
        stride: int | None = None,
    ):
        """
        Initialize pre-training dataset.
        
        Args:
            file_path: Path to text file for pre-training
            tokenizer: Tokenizer instance with encode method
            max_length: Maximum sequence length
            stride: Stride for creating overlapping sequences (default: max_length)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride or max_length
        
        # Read and tokenize the entire file
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        # Tokenize the entire text
        self.token_ids = tokenizer.encode(text)
        
        # Calculate the number of sequences
        if len(self.token_ids) <= max_length:
            self.num_sequences = 1
        else:
            self.num_sequences = (len(self.token_ids) - max_length) // self.stride + 1
    
    def __len__(self) -> int:
        return self.num_sequences
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a training sequence for language model pre-training.
        
        Returns:
            Dictionary with:
            - input_ids: Token IDs of shape (max_length,)
            - labels: Target token IDs (shifted by 1) of shape (max_length,)
        
        Implementation notes:
            - Calculate start and end indices based on idx and stride
            - Extract sequence of length max_length + 1 (need extra token for shifting)
            - Pad sequence if it's shorter than max_length + 1
            - For language modeling: input_ids = sequence[:-1], labels = sequence[1:]
              This creates the next-token prediction task
        """
        # TODO: Implement __getitem__ for pre-training dataset
        # Step 1: Calculate start_idx = idx * self.stride
        # Step 2: Calculate end_idx (don't exceed len(self.token_ids))
        # Step 3: Extract sequence from self.token_ids[start_idx:end_idx]
        # Step 4: Pad sequence if len < max_length + 1
        # Step 5: Create input_ids = sequence[:-1] and labels = sequence[1:]
        # Step 6: Convert to torch tensors with dtype=torch.long
        # Step 7: Return dict with "input_ids" and "labels"
        
        raise NotImplementedError("Implement PretrainingDataset.__getitem__")


class MultipleChoiceQADataset(Dataset):
    """
    Dataset for multiple-choice question answering.
    
    Each example has:
    - context: Background text
    - question: The question being asked
    - choices: List of answer options (typically 4)
    - answer: Index of correct answer (0-indexed)
    """
    
    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer,
        max_length: int = 256,
        num_choices: int = 4,
    ):
        """
        Initialize QA dataset.
        
        Args:
            data: List of QA examples, each with keys:
                - context: str
                - question: str  
                - choices: List[str] (length num_choices)
                - answer: int (optional, -1 if not provided)
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length per choice
            num_choices: Number of choices per question
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_choices = num_choices
    
    def __len__(self) -> int:
        return len(self.data)
    
    def _format_choice_input(self, context: str, question: str, choice: str) -> str:
        """Format a single choice as input text."""
        return f"{context}\n\nQuestion: {question}\n\nAnswer: {choice}"
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a QA example for multiple-choice fine-tuning.
        
        Returns:
            Dictionary with:
            - input_ids: Token IDs of shape (num_choices, max_length)
            - attention_mask: Mask of shape (num_choices, max_length)
            - labels: Correct answer index (scalar)
        
        Implementation notes:
            - For each choice, format the input as: "{context}\n\nQuestion: {question}\n\nAnswer: {choice}"
            - Tokenize each formatted input
            - Truncate to max_length if needed
            - Create attention_mask (1 for real tokens, 0 for padding)
            - Pad all sequences to max_length
            - Stack all choices into tensors
        """
        example = self.data[idx]
        
        context = example["context"]
        question = example["question"]
        choices = example["choices"]
        answer = example.get("answer", -1)  # -1 for test set without labels
        
        all_input_ids = []
        all_attention_masks = []
        
        # TODO: Implement __getitem__ for QA dataset
        # For each choice in choices:
        #   Step 1: Format input text using self._format_choice_input(context, question, choice)
        #   Step 2: Tokenize using self.tokenizer.encode(text)
        #   Step 3: Truncate if len(token_ids) > self.max_length
        #   Step 4: Create attention_mask = [1] * len(token_ids)
        #   Step 5: Pad token_ids and attention_mask to self.max_length
        #   Step 6: Append to all_input_ids and all_attention_masks
        #
        # Return dict with:
        #   - "input_ids": tensor of shape (num_choices, max_length)
        #   - "attention_mask": tensor of shape (num_choices, max_length)
        #   - "labels": tensor containing answer index
        
        raise NotImplementedError("Implement MultipleChoiceQADataset.__getitem__")
    
    @classmethod
    def from_json(cls, file_path: str | Path, tokenizer, **kwargs) -> "MultipleChoiceQADataset":
        """Load dataset from JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data, tokenizer, **kwargs)


def create_pretraining_dataloader(
    file_path: str | Path,
    tokenizer,
    batch_size: int = 8,
    max_length: int = 256,
    stride: int | None = None,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create a DataLoader for pre-training.
    
    Args:
        file_path: Path to text file
        tokenizer: Tokenizer instance
        batch_size: Batch size
        max_length: Maximum sequence length
        stride: Stride for overlapping sequences
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
    
    Returns:
        DataLoader instance
    """
    dataset = PretrainingDataset(file_path, tokenizer, max_length, stride)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


def create_qa_dataloader(
    data: List[Dict[str, Any]] | str | Path,
    tokenizer,
    batch_size: int = 4,
    max_length: int = 256,
    num_choices: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create a DataLoader for QA fine-tuning.
    
    Args:
        data: List of QA examples or path to JSON file
        tokenizer: Tokenizer instance
        batch_size: Batch size
        max_length: Maximum sequence length
        num_choices: Number of choices per question
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
    
    Returns:
        DataLoader instance
    """
    if isinstance(data, (str, Path)):
        dataset = MultipleChoiceQADataset.from_json(data, tokenizer, max_length=max_length, num_choices=num_choices)
    else:
        dataset = MultipleChoiceQADataset(data, tokenizer, max_length=max_length, num_choices=num_choices)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
