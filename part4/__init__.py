"""
Part 4: Pre-training + Fine-tuning + Prompting

This module provides utilities for:
- Pre-training a Transformer LM on TinyStories
- Fine-tuning on Multiple-choice QA
- Prompting experiments
"""

from .sampling import greedy_decode, top_k_decode
from .datasets import PretrainingDataset, MultipleChoiceQADataset
from .trainer import Trainer, TrainingConfig
from .qa_model import TransformerForMultipleChoice
from .prompting import PromptTemplate, PromptingPipeline

__all__ = [
    "greedy_decode",
    "top_k_decode",
    "PretrainingDataset",
    "MultipleChoiceQADataset",
    "Trainer",
    "TrainingConfig",
    "TransformerForMultipleChoice",
    "PromptTemplate",
    "PromptingPipeline",
]
