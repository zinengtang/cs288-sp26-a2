"""
Training utilities for pre-training and fine-tuning.

Provides a unified Trainer class for both pre-training and fine-tuning.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Callable
from pathlib import Path
import time
import sys

# Add parent directory to path for imports
_parent = str(Path(__file__).parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from part3.nn_utils import cross_entropy, gradient_clipping


@dataclass
class TrainingConfig:
    """Configuration for training."""
    
    # Training parameters
    num_epochs: int = 3
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    
    # Batch size (can also be set in dataloader)
    batch_size: int = 8
    
    # Logging and checkpointing
    log_interval: int = 10
    save_interval: int = 500
    checkpoint_dir: Optional[str] = None
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Mixed precision (optional)
    use_amp: bool = False
    
    # Early stopping (optional)
    patience: Optional[int] = None
    
    def __post_init__(self):
        if self.checkpoint_dir is not None:
            Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)


class Trainer:
    """
    Unified trainer for pre-training and fine-tuning.
    
    Supports:
    - Language model pre-training (next token prediction)
    - Multiple-choice QA fine-tuning
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        compute_loss_fn: Optional[Callable] = None,
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            config: Training configuration
            train_dataloader: Training data loader
            val_dataloader: Optional validation data loader
            compute_loss_fn: Optional custom loss function
        """
        self.model = model.to(config.device)
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.compute_loss_fn = compute_loss_fn or self._default_lm_loss
        
        # Setup optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # Setup learning rate scheduler
        total_steps = len(train_dataloader) * config.num_epochs
        
        if config.warmup_steps > 0:
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=config.warmup_steps,
            )
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps - config.warmup_steps,
            )
            self.scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[config.warmup_steps],
            )
        else:
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps,
            )
        
        # Training state
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        
        # Mixed precision
        self.scaler = torch.amp.GradScaler("cuda") if config.use_amp else None
        
        # Logging
        self.train_losses = []
        self.val_losses = []
    
    def _default_lm_loss(self, batch: Dict[str, torch.Tensor], model: nn.Module) -> torch.Tensor:
        """
        Default language model loss (next token prediction).
        
        Args:
            batch: Dictionary with "input_ids" and "labels" tensors
            model: TransformerLM model
        
        Returns:
            Scalar loss tensor
        
        Implementation notes:
            - Move input_ids and labels to device
            - Forward pass: logits = model(input_ids) -> shape (batch, seq_len, vocab_size)
            - Flatten logits to (batch * seq_len, vocab_size) for cross_entropy
            - Flatten labels to (batch * seq_len,)
            - Compute cross_entropy loss
        """
        input_ids = batch["input_ids"].to(self.config.device)
        labels = batch["labels"].to(self.config.device)
        
        # TODO: Implement language model loss
        # Step 1: Forward pass through model to get logits
        # Step 2: Get shape (batch_size, seq_len, vocab_size) from logits
        # Step 3: Flatten logits to (batch * seq_len, vocab_size)
        # Step 4: Flatten labels to (batch * seq_len,)
        # Step 5: Compute cross_entropy loss using the imported cross_entropy function
        # Step 6: Return the loss
        
        raise NotImplementedError("Implement _default_lm_loss")
    
    def train_epoch(self) -> float:
        """
        Train for one epoch.
        
        Returns:
            Average training loss for the epoch
        
        Implementation notes:
            Standard PyTorch training loop:
            1. Set model to train mode
            2. For each batch:
               a. Zero gradients
               b. Compute loss using self.compute_loss_fn
               c. Backward pass
               d. Gradient clipping using gradient_clipping function
               e. Optimizer step
               f. Scheduler step
               g. Log progress periodically
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        epoch_start_time = time.time()
        
        for batch_idx, batch in enumerate(self.train_dataloader):
            # TODO: Implement training loop for one epoch
            # Step 1: Zero gradients with self.optimizer.zero_grad()
            # Step 2: Compute loss using self.compute_loss_fn(batch, self.model)
            # Step 3: Backward pass with loss.backward()
            # Step 4: Gradient clipping with gradient_clipping(self.model.parameters(), self.config.max_grad_norm)
            # Step 5: Optimizer step with self.optimizer.step()
            # Step 6: Scheduler step with self.scheduler.step()
            # Step 7: Update total_loss, num_batches, self.global_step
            # Step 8: Log progress if self.global_step % self.config.log_interval == 0
            
            raise NotImplementedError("Implement train_epoch")
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def evaluate(self) -> float:
        """
        Evaluate on validation set.
        
        Returns:
            Average validation loss
        """
        if self.val_dataloader is None:
            return 0.0
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.val_dataloader:
            loss = self.compute_loss_fn(batch, self.model)
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        
        return avg_loss
    
    def train(self) -> Dict[str, Any]:
        """
        Full training loop.
        
        Returns:
            Dictionary with training history
        """
        print(f"Starting training for {self.config.num_epochs} epochs...")
        print(f"  Device: {self.config.device}")
        print(f"  Total batches per epoch: {len(self.train_dataloader)}")
        print(f"  Learning rate: {self.config.learning_rate}")
        
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            
            # Train
            train_loss = self.train_epoch()
            print(f"  Train loss: {train_loss:.4f}")
            
            # Evaluate
            if self.val_dataloader is not None:
                val_loss = self.evaluate()
                print(f"  Val loss: {val_loss:.4f}")
                
                # Early stopping
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    if self.config.checkpoint_dir:
                        self.save_checkpoint("best_model.pt")
                else:
                    self.patience_counter += 1
                    if self.config.patience and self.patience_counter >= self.config.patience:
                        print(f"  Early stopping triggered after {epoch + 1} epochs")
                        break
        
        total_time = time.time() - start_time
        print(f"\nTraining complete in {total_time:.1f}s")
        
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss,
            "total_time": total_time,
        }
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = Path(self.config.checkpoint_dir) / filename
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }, path)
        print(f"  Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str | Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.config.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]
        print(f"Loaded checkpoint from {path}")


def compute_qa_loss(batch: Dict[str, torch.Tensor], model: nn.Module, device: str = "cuda") -> torch.Tensor:
    """
    Compute loss for multiple-choice QA.
    
    The model should return logits for each choice. We compute cross-entropy
    over the choice dimension.
    
    Args:
        batch: Dictionary with input_ids, attention_mask, labels
        model: TransformerForMultipleChoice model
        device: Device to use
    
    Returns:
        Loss tensor
    
    Implementation notes:
        - Move input_ids, attention_mask, labels to device
        - Forward pass: logits = model(input_ids, attention_mask) -> shape (batch, num_choices)
        - Compute cross_entropy(logits, labels) where labels are choice indices
    """
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    
    # TODO: Implement QA loss computation
    # Step 1: Forward pass through model to get logits (batch, num_choices)
    # Step 2: Compute cross_entropy loss between logits and labels
    # Step 3: Return the loss
    
    raise NotImplementedError("Implement compute_qa_loss")


def create_qa_loss_fn(device: str = "cuda") -> Callable:
    """Create a QA loss function for the trainer."""
    def loss_fn(batch, model):
        return compute_qa_loss(batch, model, device)
    return loss_fn
