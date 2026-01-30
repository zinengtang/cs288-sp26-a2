"""
Sampling utilities for text generation.

Implements greedy decoding and top-k sampling for autoregressive generation.
"""

import torch
from torch import Tensor
import sys
from pathlib import Path

# Add parent directory to path for imports
_parent = str(Path(__file__).parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from part3.nn_utils import softmax


def greedy_decode(
    model,
    input_ids: Tensor,
    max_new_tokens: int,
    eos_token_id: int | None = None,
    pad_token_id: int | None = None,
) -> Tensor:
    """
    Generate tokens using greedy decoding (always pick the highest probability token).
    
    Args:
        model: TransformerLM model
        input_ids: Input token IDs of shape (batch, seq_len)
        max_new_tokens: Maximum number of new tokens to generate
        eos_token_id: Optional end-of-sequence token ID to stop generation
        pad_token_id: Optional padding token ID (unused in greedy, for compatibility)
    
    Returns:
        Generated token IDs of shape (batch, seq_len + generated_len)
    
    Algorithm:
        1. Set model to eval mode and move input to correct device
        2. For each new token to generate:
           a. Get logits from model for current sequence
           b. Take logits for the last position only
           c. Select token with highest logit (argmax)
           d. Append selected token to sequence
           e. Stop early if EOS token is generated
        3. Return the full generated sequence
    """
    model.eval()
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    
    generated = input_ids.clone()
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # TODO: Implement greedy decoding
            # Step 1: Get model predictions (logits) for the current sequence
            # Step 2: Extract logits for the last position only (shape: batch, vocab_size)
            # Step 3: Select the token with highest probability using argmax
            # Step 4: Append the selected token to the generated sequence
            # Step 5: Check for EOS token and break if all sequences have generated it
            
            raise NotImplementedError("Implement greedy decoding")
    
    return generated


def top_k_decode(
    model,
    input_ids: Tensor,
    max_new_tokens: int,
    k: int = 50,
    temperature: float = 1.0,
    eos_token_id: int | None = None,
    pad_token_id: int | None = None,
) -> Tensor:
    """
    Generate tokens using top-k sampling.
    
    At each step, sample from the top-k most likely tokens, with probabilities
    optionally scaled by temperature.
    
    Args:
        model: TransformerLM model
        input_ids: Input token IDs of shape (batch, seq_len)
        max_new_tokens: Maximum number of new tokens to generate
        k: Number of top tokens to sample from
        temperature: Sampling temperature (higher = more random, lower = more deterministic)
        eos_token_id: Optional end-of-sequence token ID to stop generation
        pad_token_id: Optional padding token ID (unused here, for compatibility)
    
    Returns:
        Generated token IDs of shape (batch, seq_len + generated_len)
    
    Algorithm:
        1. Set model to eval mode and move input to correct device
        2. For each new token to generate:
           a. Get logits from model for current sequence
           b. Take logits for the last position only
           c. Apply temperature scaling: logits = logits / temperature
           d. Get top-k logits and their indices using torch.topk
           e. Convert top-k logits to probabilities using softmax
           f. Sample from the top-k distribution using torch.multinomial
           g. Map sampled index back to vocabulary index using gather
           h. Append selected token to sequence
           i. Stop early if EOS token is generated
        3. Return the full generated sequence
    """
    model.eval()
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    
    generated = input_ids.clone()
    batch_size = input_ids.shape[0]
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # TODO: Implement top-k sampling
            # Step 1: Get model predictions (logits) for the current sequence
            # Step 2: Extract logits for the last position only
            # Step 3: Apply temperature scaling (divide logits by temperature)
            # Step 4: Get top-k logits and indices using torch.topk
            # Step 5: Convert top-k logits to probabilities using softmax
            # Step 6: Sample from the distribution using torch.multinomial
            # Step 7: Map the sampled index back to vocabulary index using gather
            # Step 8: Append the selected token to the generated sequence
            # Step 9: Check for EOS token and break if needed
            
            raise NotImplementedError("Implement top-k sampling")
    
    return generated


def nucleus_decode(
    model,
    input_ids: Tensor,
    max_new_tokens: int,
    p: float = 0.9,
    temperature: float = 1.0,
    eos_token_id: int | None = None,
    pad_token_id: int | None = None,
) -> Tensor:
    """
    Generate tokens using nucleus (top-p) sampling.
    
    At each step, sample from the smallest set of tokens whose cumulative
    probability exceeds p.
    
    Args:
        model: TransformerLM model
        input_ids: Input token IDs of shape (batch, seq_len)
        max_new_tokens: Maximum number of new tokens to generate
        p: Cumulative probability threshold (nucleus)
        temperature: Sampling temperature
        eos_token_id: Optional end-of-sequence token ID to stop generation
        pad_token_id: Optional padding token ID (unused here, for compatibility)
    
    Returns:
        Generated token IDs of shape (batch, seq_len + generated_len)
    """
    model.eval()
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    
    generated = input_ids.clone()
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Get model predictions for the last position
            logits = model(generated)
            next_token_logits = logits[:, -1, :]
            
            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Sort logits in descending order
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True, dim=-1)
            sorted_probs = softmax(sorted_logits, dim=-1)
            
            # Compute cumulative probabilities
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            # Find the cutoff index where cumulative probability exceeds p
            # We want to keep tokens until cumsum > p, so shift by 1
            sorted_indices_to_remove = cumulative_probs > p
            # Shift right to keep at least one token
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            
            # Zero out removed tokens
            sorted_probs[sorted_indices_to_remove] = 0
            
            # Renormalize
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
            
            # Sample
            sampled_indices = torch.multinomial(sorted_probs, num_samples=1)
            next_tokens = sorted_indices.gather(dim=-1, index=sampled_indices)
            
            generated = torch.cat([generated, next_tokens], dim=1)
            
            if eos_token_id is not None:
                if (next_tokens == eos_token_id).all():
                    break
    
    return generated


def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    method: str = "greedy",
    k: int = 50,
    p: float = 0.9,
    temperature: float = 1.0,
    eos_token_id: int | None = None,
) -> str:
    """
    Generate text from a prompt using the specified decoding method.
    
    Args:
        model: TransformerLM model
        tokenizer: Tokenizer instance
        prompt: Input text prompt
        max_new_tokens: Maximum number of new tokens to generate
        method: Decoding method ("greedy", "top_k", or "nucleus")
        k: Top-k parameter (for top_k method)
        p: Nucleus parameter (for nucleus method)
        temperature: Sampling temperature
        eos_token_id: Optional EOS token ID
    
    Returns:
        Generated text (including the prompt)
    """
    # Encode prompt
    input_ids = torch.tensor([tokenizer.encode(prompt)])
    
    # Select decoding method
    if method == "greedy":
        output_ids = greedy_decode(model, input_ids, max_new_tokens, eos_token_id)
    elif method == "top_k":
        output_ids = top_k_decode(model, input_ids, max_new_tokens, k, temperature, eos_token_id)
    elif method == "nucleus":
        output_ids = nucleus_decode(model, input_ids, max_new_tokens, p, temperature, eos_token_id)
    else:
        raise ValueError(f"Unknown decoding method: {method}")
    
    # Decode output
    generated_text = tokenizer.decode(output_ids[0].tolist())
    
    return generated_text
