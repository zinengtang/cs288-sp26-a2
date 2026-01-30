"""
Multiple-choice QA model built on top of the Transformer LM.

Uses the transformer backbone with a classification head for multiple-choice QA.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
import sys
from pathlib import Path

# Add parent directory to path for imports
_parent = str(Path(__file__).parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from part2.model import Linear


class TransformerForMultipleChoice(nn.Module):
    """
    Transformer model for multiple-choice question answering.
    
    Uses a pre-trained TransformerLM as backbone and adds a classification head.
    For each choice, we:
    1. Encode the (context, question, choice) sequence
    2. Pool the hidden states (last token, mean, or max)
    3. Project to a single logit
    
    The final prediction is the choice with the highest logit.
    """
    
    def __init__(
        self,
        transformer_lm,
        hidden_size: int,
        num_choices: int = 4,
        pooling: str = "last",
        freeze_backbone: bool = False,
    ):
        """
        Initialize QA model.
        
        Args:
            transformer_lm: Pre-trained TransformerLM model
            hidden_size: Hidden size of the transformer (d_model)
            num_choices: Number of answer choices
            pooling: Pooling method ("last", "mean", or "max")
            freeze_backbone: Whether to freeze the transformer backbone
        """
        super().__init__()
        
        self.transformer = transformer_lm
        self.hidden_size = hidden_size
        self.num_choices = num_choices
        self.pooling = pooling
        
        # Classification head: projects pooled representation to single logit
        self.classifier = Linear(hidden_size, 1)
        
        # Optionally freeze backbone
        if freeze_backbone:
            for param in self.transformer.parameters():
                param.requires_grad = False
    
    def _get_hidden_states(self, input_ids: Tensor) -> Tensor:
        """
        Get hidden states from transformer (before output projection).
        
        Args:
            input_ids: Token IDs of shape (batch, seq_len)
        
        Returns:
            Hidden states of shape (batch, seq_len, hidden_size)
        
        Implementation notes:
            - Get token embeddings from self.transformer.token_embeddings
            - Apply each transformer layer from self.transformer.layers
            - Apply final layer norm from self.transformer.final_ln
            - Do NOT apply the output projection (we want hidden states, not logits)
        """
        # TODO: Implement hidden state extraction
        # Step 1: Get token embeddings: x = self.transformer.token_embeddings(input_ids)
        # Step 2: Apply each transformer layer: for layer in self.transformer.layers: x = layer(x)
        # Step 3: Apply final layer norm: x = self.transformer.final_ln(x)
        # Step 4: Return x (hidden states)
        
        raise NotImplementedError("Implement _get_hidden_states")
    
    def _pool(self, hidden_states: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        """
        Pool hidden states to a single vector.
        
        Args:
            hidden_states: Hidden states of shape (batch, seq_len, hidden_size)
            attention_mask: Optional mask of shape (batch, seq_len)
        
        Returns:
            Pooled representation of shape (batch, hidden_size)
        
        Implementation notes:
            Implement three pooling strategies based on self.pooling:
            
            "last": Use the last non-padded token
                - If attention_mask provided: find last 1 in each sequence
                - seq_lengths = attention_mask.sum(dim=1).long() - 1
                - Index hidden_states at those positions
            
            "mean": Mean pooling over non-padded tokens
                - Expand mask to (batch, seq_len, 1)
                - Multiply hidden_states by mask, sum over seq_len
                - Divide by number of non-padded tokens
            
            "max": Max pooling over non-padded tokens
                - Set padded positions to -inf before taking max
        """
        # TODO: Implement pooling strategies
        # Check self.pooling and implement the appropriate strategy:
        #
        # if self.pooling == "last":
        #     Use the last non-padded token's hidden state
        #     Hint: seq_lengths = attention_mask.sum(dim=1).long() - 1
        #
        # elif self.pooling == "mean":
        #     Average hidden states over non-padded positions
        #     Hint: mask out padded positions before averaging
        #
        # elif self.pooling == "max":
        #     Take element-wise max over non-padded positions
        #     Hint: set padded positions to -inf before max
        
        raise NotImplementedError("Implement _pool")
    
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass for multiple-choice QA.
        
        Args:
            input_ids: Token IDs of shape (batch, num_choices, seq_len)
            attention_mask: Optional mask of shape (batch, num_choices, seq_len)
        
        Returns:
            Logits of shape (batch, num_choices)
        
        Implementation notes:
            1. Flatten input_ids from (batch, num_choices, seq_len) to (batch*num_choices, seq_len)
            2. Also flatten attention_mask if provided
            3. Get hidden states using _get_hidden_states
            4. Pool hidden states using _pool
            5. Apply classifier to get logit per choice
            6. Reshape back to (batch, num_choices)
        """
        batch_size, num_choices, seq_len = input_ids.shape
        
        # TODO: Implement forward pass for multiple-choice QA
        # Step 1: Flatten input_ids to (batch * num_choices, seq_len) using view(-1, seq_len)
        # Step 2: Flatten attention_mask if provided (same reshaping)
        # Step 3: Get hidden states using self._get_hidden_states(input_ids_flat)
        # Step 4: Pool hidden states using self._pool(hidden_states, attention_mask_flat)
        # Step 5: Apply classifier: logits = self.classifier(pooled).squeeze(-1)
        # Step 6: Reshape logits to (batch_size, num_choices)
        # Step 7: Return logits
        
        raise NotImplementedError("Implement forward")
    
    @torch.no_grad()
    def predict(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Predict the correct choice for each example.
        
        Args:
            input_ids: Token IDs of shape (batch, num_choices, seq_len)
            attention_mask: Optional mask of shape (batch, num_choices, seq_len)
        
        Returns:
            Predicted choice indices of shape (batch,)
        """
        self.eval()
        logits = self.forward(input_ids, attention_mask)
        return logits.argmax(dim=-1)
    
    @torch.no_grad()
    def predict_probs(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Get probability distribution over choices.
        
        Args:
            input_ids: Token IDs of shape (batch, num_choices, seq_len)
            attention_mask: Optional mask of shape (batch, num_choices, seq_len)
        
        Returns:
            Probabilities of shape (batch, num_choices)
        """
        self.eval()
        logits = self.forward(input_ids, attention_mask)
        return torch.softmax(logits, dim=-1)


def evaluate_qa_model(
    model: TransformerForMultipleChoice,
    dataloader,
    device: str = "cuda",
) -> dict:
    """
    Evaluate QA model on a dataset.
    
    Args:
        model: QA model
        dataloader: DataLoader with QA examples
        device: Device to use
    
    Returns:
        Dictionary with accuracy and predictions
    """
    model.eval()
    model.to(device)
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            predictions = model.predict(input_ids, attention_mask)
            
            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    
    # Calculate accuracy (ignoring -1 labels for test set)
    correct = 0
    total = 0
    for pred, label in zip(all_predictions, all_labels):
        if label >= 0:
            total += 1
            if pred == label:
                correct += 1
    
    accuracy = correct / total if total > 0 else 0.0
    
    return {
        "accuracy": accuracy,
        "predictions": all_predictions,
        "labels": all_labels,
    }
