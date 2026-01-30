"""
Transformer model implementation from scratch.
Implements all components needed for a decoder-only transformer language model.
"""
import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from nn_utils import softmax


# =============================================================================
# Problem (linear): Implementing the linear module
# =============================================================================

class Linear(nn.Module):
    """
    Linear transformation layer: y = xW^T
    
    Note: We don't use bias in modern transformer implementations (like LLaMA).
    """
    
    def __init__(self, d_in: int, d_out: int):
        """
        Initialize linear layer.
        
        Args:
            d_in: Input dimension
            d_out: Output dimension
        """
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        # Weight matrix of shape (d_out, d_in)
        self.weight = nn.Parameter(torch.empty(d_out, d_in))
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply linear transformation: y = x @ W^T
        
        Args:
            x: Input tensor of shape (..., d_in)
        
        Returns:
            Output tensor of shape (..., d_out)
        """
        # TODO: Implement linear transformation
        # Return x @ self.weight.T
        
        raise NotImplementedError("Implement Linear.forward")


# =============================================================================
# Problem (embedding): Implement the embedding module
# =============================================================================

class Embedding(nn.Module):
    """
    Token embedding layer that maps token indices to dense vectors.
    """
    
    def __init__(self, vocab_size: int, d_model: int):
        """
        Initialize embedding layer.
        
        Args:
            vocab_size: Size of vocabulary
            d_model: Embedding dimension
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        # Embedding weight matrix of shape (vocab_size, d_model)
        self.weight = nn.Parameter(torch.empty(vocab_size, d_model))
        self._init_weights()
    
    def _init_weights(self):
        """Initialize embeddings from normal distribution."""
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
    
    def forward(self, token_ids: Tensor) -> Tensor:
        """
        Look up embeddings for token IDs.
        
        Args:
            token_ids: Tensor of token indices of shape (batch, seq_len)
        
        Returns:
            Tensor of embeddings of shape (batch, seq_len, d_model)
        """
        # TODO: Implement embedding lookup
        # Use token_ids to index into self.weight
        # Return self.weight[token_ids]
        
        raise NotImplementedError("Implement Embedding.forward")


# =============================================================================
# Problem (rmsnorm): Root Mean Square Layer Normalization
# =============================================================================

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    RMSNorm is a simplification of LayerNorm that removes the mean centering
    and only normalizes by the root mean square of the activations.
    
    RMSNorm(x) = x / RMS(x) * gamma
    where RMS(x) = sqrt(mean(x^2) + eps)
    """
    
    def __init__(self, d_model: int, eps: float = 1e-5):
        """
        Initialize RMSNorm.
        
        Args:
            d_model: Model dimension (size of last dimension)
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        # Learnable scale parameter (gamma)
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply RMS normalization.
        
        RMSNorm(x) = x / RMS(x) * gamma
        where RMS(x) = sqrt(mean(x^2) + eps)
        
        Args:
            x: Input tensor of shape (..., d_model)
        
        Returns:
            Normalized tensor of same shape
        """
        # TODO: Implement RMS normalization
        # Step 1: Compute RMS = sqrt(mean(x^2, dim=-1, keepdim=True) + self.eps)
        # Step 2: Normalize: x_norm = x / rms
        # Step 3: Scale by learnable weight: return x_norm * self.weight
        
        raise NotImplementedError("Implement RMSNorm.forward")


# =============================================================================
# Problem (softmax): Implement softmax (used in attention)
# =============================================================================

# softmax is implemented in nn_utils.py


# =============================================================================
# SiLU activation (helper for SwiGLU)
# =============================================================================

def silu(x: Tensor) -> Tensor:
    """
    SiLU (Sigmoid Linear Unit) activation function.
    Also known as Swish: silu(x) = x * sigmoid(x)
    
    Args:
        x: Input tensor
    
    Returns:
        Tensor with SiLU applied element-wise
    """
    # TODO: Implement SiLU activation
    # Return x * torch.sigmoid(x)
    
    raise NotImplementedError("Implement silu")


# =============================================================================
# Problem (positionwise_feedforward): Implement the position-wise feed-forward network
# =============================================================================

class SwiGLU(nn.Module):
    """
    SwiGLU Feed-Forward Network.
    
    SwiGLU is a variant of the GLU (Gated Linear Unit) that uses SiLU activation.
    It has three linear projections: gate, up, and down.
    
    SwiGLU(x) = (SiLU(x @ W1^T) * (x @ W3^T)) @ W2^T
    
    Where:
        - W1 is the gate projection
        - W3 is the up projection  
        - W2 is the down projection
    """
    
    def __init__(self, d_model: int, d_ff: int):
        """
        Initialize SwiGLU layer.
        
        Args:
            d_model: Model dimension
            d_ff: Hidden dimension of feed-forward layer
        """
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        
        # Gate projection: d_model -> d_ff
        self.w1 = Linear(d_model, d_ff)
        # Down projection: d_ff -> d_model
        self.w2 = Linear(d_ff, d_model)
        # Up projection: d_model -> d_ff
        self.w3 = Linear(d_model, d_ff)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply SwiGLU transformation.
        
        SwiGLU(x) = (SiLU(x @ W1^T) * (x @ W3^T)) @ W2^T
        
        Args:
            x: Input tensor of shape (..., d_model)
        
        Returns:
            Output tensor of shape (..., d_model)
        """
        # TODO: Implement SwiGLU
        # Step 1: gate = silu(self.w1(x))  # Gate projection with SiLU
        # Step 2: up = self.w3(x)          # Up projection
        # Step 3: hidden = gate * up       # Element-wise multiplication
        # Step 4: return self.w2(hidden)   # Down projection
        
        raise NotImplementedError("Implement SwiGLU.forward")


# =============================================================================
# Problem (rope): Implement RoPE (Rotary Position Embedding)
# =============================================================================

class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).
    
    RoPE encodes position information by rotating the query and key vectors
    in a way that makes the dot product depend on relative position.
    
    For each pair of dimensions (2i, 2i+1), we rotate by angle m * theta_i,
    where m is the position and theta_i = 1 / (base ^ (2i / d)).
    """
    
    def __init__(self, d_model: int, max_seq_len: int, theta: float = 10000.0):
        """
        Initialize RoPE.
        
        Args:
            d_model: Model dimension (head dimension for attention)
            max_seq_len: Maximum sequence length
            theta: Base for frequency computation (default: 10000.0)
        """
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        # Precompute frequencies
        # inv_freq shape: (d_model // 2,)
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)
        
        # Precompute cos and sin for all positions
        self._precompute_cache(max_seq_len)
    
    def _precompute_cache(self, seq_len: int):
        """Precompute cos and sin values for positions up to seq_len."""
        # positions shape: (seq_len,)
        positions = torch.arange(seq_len, device=self.inv_freq.device)
        
        # freqs shape: (seq_len, d_model // 2)
        freqs = torch.outer(positions, self.inv_freq)
        
        # Duplicate each frequency for the pair of dimensions
        # emb shape: (seq_len, d_model)
        emb = torch.cat([freqs, freqs], dim=-1)
        
        self.register_buffer("cos_cached", torch.cos(emb), persistent=False)
        self.register_buffer("sin_cached", torch.sin(emb), persistent=False)
    
    def _rotate_half(self, x: Tensor) -> Tensor:
        """
        Rotate half the hidden dims of the input.
        
        For input [..., [x1, x2, x3, x4, ...]], return [..., [-x3, -x4, ..., x1, x2, ...]]
        """
        # TODO: Implement rotate_half
        # Split x into two halves along last dimension
        # x1 = x[..., :d // 2], x2 = x[..., d // 2:]
        # Return torch.cat([-x2, x1], dim=-1)
        
        raise NotImplementedError("Implement _rotate_half")
    
    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        """
        Apply rotary position embedding.
        
        Args:
            x: Input tensor of shape (..., seq_len, d_model) or (batch, num_heads, seq_len, d_k)
            token_positions: Position indices of shape (batch, seq_len) or (seq_len,)
        
        Returns:
            Tensor with rotary position embedding applied, same shape as input
        
        Formula: x_rotated = x * cos(theta) + rotate_half(x) * sin(theta)
        """
        # TODO: Implement RoPE forward pass
        # Step 1: Get cos and sin for given positions from cached buffers
        #         cos = self.cos_cached[token_positions]
        #         sin = self.sin_cached[token_positions]
        # Step 2: Handle dimensions for broadcasting:
        #         - If x is 4D (batch, heads, seq, d_k), expand cos/sin with unsqueeze
        # Step 3: Apply rotation formula:
        #         return x * cos + self._rotate_half(x) * sin
        
        raise NotImplementedError("Implement RotaryPositionEmbedding.forward")


def apply_rope(x: Tensor, d_model: int, theta: float, max_seq_len: int, token_positions: Tensor) -> Tensor:
    """
    Functional interface for applying RoPE.
    
    Args:
        x: Input tensor of shape (..., seq_len, d_model)
        d_model: Dimension of the model/head
        theta: RoPE base frequency
        max_seq_len: Maximum sequence length
        token_positions: Position indices
    
    Returns:
        Tensor with RoPE applied
    """
    rope = RotaryPositionEmbedding(d_model, max_seq_len, theta)
    rope = rope.to(x.device)
    return rope(x, token_positions)


# =============================================================================
# Problem (scaled_dot_product_attention): Implement scaled dot-product attention
# =============================================================================

def scaled_dot_product_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    mask: Optional[Tensor] = None,
) -> Tensor:
    """
    Compute scaled dot-product attention.
    
    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
    
    Args:
        Q: Query tensor of shape (..., seq_len_q, d_k)
        K: Key tensor of shape (..., seq_len_k, d_k)
        V: Value tensor of shape (..., seq_len_k, d_v)
        mask: Optional boolean mask of shape (..., seq_len_q, seq_len_k)
              True values indicate positions to attend to, False positions are masked
    
    Returns:
        Attention output of shape (..., seq_len_q, d_v)
    """
    d_k = Q.shape[-1]
    
    # TODO: Implement scaled dot-product attention
    # Step 1: Compute attention scores = Q @ K^T / sqrt(d_k)
    #         scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    # Step 2: Apply mask if provided (set masked positions to -inf)
    #         if mask is not None: scores = scores.masked_fill(~mask, float('-inf'))
    # Step 3: Apply softmax to get attention weights
    #         attn_weights = softmax(scores, dim=-1)
    # Step 4: Handle NaN from all-masked rows: attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
    # Step 5: Compute output = attn_weights @ V
    # Step 6: Return output
    
    raise NotImplementedError("Implement scaled_dot_product_attention")


# =============================================================================
# Problem (multihead_self_attention): Implement causal multi-head self-attention
# =============================================================================

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention layer with causal masking.
    
    This implements the attention mechanism used in decoder-only transformers
    like GPT and LLaMA. It projects the input into queries, keys, and values,
    applies scaled dot-product attention with causal masking, and projects back.
    """
    
    def __init__(self, d_model: int, num_heads: int):
        """
        Initialize multi-head self-attention.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head
        
        # Projection layers
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)
    
    def _create_causal_mask(self, seq_len: int, device: torch.device) -> Tensor:
        """Create causal (lower triangular) attention mask."""
        # mask[i, j] = True if j <= i (can attend to position j from position i)
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
        return mask
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply multi-head self-attention.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
        
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # TODO: Implement multi-head self-attention
        # Step 1: Project x to Q, K, V using self.q_proj, self.k_proj, self.v_proj
        # Step 2: Reshape from (batch, seq_len, d_model) to (batch, num_heads, seq_len, d_k)
        #         Use view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        # Step 3: Create causal mask using self._create_causal_mask(seq_len, x.device)
        #         Expand with unsqueeze(0).unsqueeze(0) for batch and heads
        # Step 4: Apply scaled_dot_product_attention(Q, K, V, mask)
        # Step 5: Reshape back to (batch, seq_len, d_model)
        #         Use transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        # Step 6: Return self.output_proj(attn_output)
        
        raise NotImplementedError("Implement MultiHeadSelfAttention.forward")


class MultiHeadSelfAttentionWithRoPE(nn.Module):
    """
    Multi-Head Self-Attention with Rotary Position Embedding (RoPE).
    
    This extends the basic multi-head attention by applying RoPE to the
    query and key vectors before computing attention scores.
    """
    
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, theta: float = 10000.0):
        """
        Initialize multi-head self-attention with RoPE.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            max_seq_len: Maximum sequence length for RoPE
            theta: RoPE base frequency
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        # Projection layers
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)
        
        # RoPE for query/key rotation
        self.rope = RotaryPositionEmbedding(self.d_k, max_seq_len, theta)
    
    def _create_causal_mask(self, seq_len: int, device: torch.device) -> Tensor:
        """Create causal (lower triangular) attention mask."""
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
        return mask
    
    def forward(self, x: Tensor, token_positions: Optional[Tensor] = None) -> Tensor:
        """
        Apply multi-head self-attention with RoPE.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            token_positions: Optional position indices of shape (batch, seq_len)
                           If None, uses sequential positions [0, 1, 2, ...]
        
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # Default to sequential positions
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        
        # TODO: Implement multi-head self-attention with RoPE
        # Step 1: Project x to Q, K, V using projection layers
        # Step 2: Reshape to (batch, seq_len, num_heads, d_k)
        # Step 3: Transpose to (batch, num_heads, seq_len, d_k)
        # Step 4: Apply RoPE to Q and K: Q_rope = self.rope(Q, token_positions)
        # Step 5: Create causal mask and expand for batch/heads
        # Step 6: Apply scaled_dot_product_attention(Q_rope, K_rope, V, mask)
        # Step 7: Reshape back to (batch, seq_len, d_model)
        # Step 8: Return self.output_proj(attn_output)
        
        raise NotImplementedError("Implement MultiHeadSelfAttentionWithRoPE.forward")


# =============================================================================
# Problem (transformer_block): Implement the Transformer block
# =============================================================================

class TransformerBlock(nn.Module):
    """
    A single Transformer decoder block.
    
    Structure (Pre-LN / LLaMA-style):
        x = x + Attention(RMSNorm(x))
        x = x + FFN(RMSNorm(x))
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float = 10000.0,
        eps: float = 1e-5,
    ):
        """
        Initialize Transformer block.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
            max_seq_len: Maximum sequence length
            theta: RoPE base frequency
            eps: Epsilon for layer normalization
        """
        super().__init__()
        
        # Layer norms (Pre-LN)
        self.ln1 = RMSNorm(d_model, eps)
        self.ln2 = RMSNorm(d_model, eps)
        
        # Self-attention with RoPE
        self.attn = MultiHeadSelfAttentionWithRoPE(d_model, num_heads, max_seq_len, theta)
        
        # Feed-forward network
        self.ffn = SwiGLU(d_model, d_ff)
    
    def forward(self, x: Tensor, token_positions: Optional[Tensor] = None) -> Tensor:
        """
        Apply Transformer block (Pre-LN style).
        
        Structure:
            x = x + Attention(RMSNorm(x))
            x = x + FFN(RMSNorm(x))
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            token_positions: Optional position indices
        
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        # TODO: Implement Transformer block forward pass
        # Step 1: Apply attention with residual: x = x + self.attn(self.ln1(x), token_positions)
        # Step 2: Apply FFN with residual: x = x + self.ffn(self.ln2(x))
        # Step 3: Return x
        
        raise NotImplementedError("Implement TransformerBlock.forward")


# =============================================================================
# Problem (transformer_lm): Implementing the Transformer LM
# =============================================================================

class TransformerLM(nn.Module):
    """
    Transformer Language Model (decoder-only, like GPT/LLaMA).
    
    Architecture:
        1. Token embedding
        2. N x Transformer blocks
        3. Final layer norm
        4. Output projection to vocabulary
    """
    
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float = 10000.0,
        eps: float = 1e-5,
    ):
        """
        Initialize Transformer LM.
        
        Args:
            vocab_size: Size of vocabulary
            context_length: Maximum sequence/context length
            d_model: Model dimension
            num_layers: Number of Transformer blocks
            num_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
            rope_theta: RoPE base frequency
            eps: Epsilon for layer normalization
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Token embeddings
        self.token_embeddings = Embedding(vocab_size, d_model)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, eps)
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.final_ln = RMSNorm(d_model, eps)
        
        # Output projection (to vocab size)
        self.output = Linear(d_model, vocab_size)
    
    def forward(self, token_ids: Tensor, token_positions: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass of the Transformer LM.
        
        Args:
            token_ids: Input token indices of shape (batch, seq_len)
            token_positions: Optional position indices of shape (batch, seq_len)
                           If None, uses sequential positions [0, 1, 2, ...]
        
        Returns:
            Logits of shape (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = token_ids.shape
        
        # Default to sequential positions
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # TODO: Implement TransformerLM forward pass
        # Step 1: Get token embeddings: x = self.token_embeddings(token_ids)
        # Step 2: Apply each transformer layer: for layer in self.layers: x = layer(x, token_positions)
        # Step 3: Apply final layer norm: x = self.final_ln(x)
        # Step 4: Apply output projection: logits = self.output(x)
        # Step 5: Return logits
        
        raise NotImplementedError("Implement TransformerLM.forward")
    
    def load_weights(self, state_dict: dict):
        """
        Load weights from a state dict.
        
        Args:
            state_dict: Dictionary mapping weight names to tensors
        """
        # Token embeddings
        if "token_embeddings.weight" in state_dict:
            self.token_embeddings.weight.data.copy_(state_dict["token_embeddings.weight"])
        
        # Output projection
        if "output.weight" in state_dict:
            self.output.weight.data.copy_(state_dict["output.weight"])
        
        # Final layer norm
        if "final_ln.weight" in state_dict:
            self.final_ln.weight.data.copy_(state_dict["final_ln.weight"])
        
        # Layer weights
        for layer_idx, layer in enumerate(self.layers):
            prefix = f"layers.{layer_idx}"
            
            # Layer norms
            if f"{prefix}.ln1.weight" in state_dict:
                layer.ln1.weight.data.copy_(state_dict[f"{prefix}.ln1.weight"])
            if f"{prefix}.ln2.weight" in state_dict:
                layer.ln2.weight.data.copy_(state_dict[f"{prefix}.ln2.weight"])
            
            # Attention projections
            if f"{prefix}.attn.q_proj.weight" in state_dict:
                layer.attn.q_proj.weight.data.copy_(state_dict[f"{prefix}.attn.q_proj.weight"])
            if f"{prefix}.attn.k_proj.weight" in state_dict:
                layer.attn.k_proj.weight.data.copy_(state_dict[f"{prefix}.attn.k_proj.weight"])
            if f"{prefix}.attn.v_proj.weight" in state_dict:
                layer.attn.v_proj.weight.data.copy_(state_dict[f"{prefix}.attn.v_proj.weight"])
            if f"{prefix}.attn.output_proj.weight" in state_dict:
                layer.attn.output_proj.weight.data.copy_(state_dict[f"{prefix}.attn.output_proj.weight"])
            
            # FFN weights
            if f"{prefix}.ffn.w1.weight" in state_dict:
                layer.ffn.w1.weight.data.copy_(state_dict[f"{prefix}.ffn.w1.weight"])
            if f"{prefix}.ffn.w2.weight" in state_dict:
                layer.ffn.w2.weight.data.copy_(state_dict[f"{prefix}.ffn.w2.weight"])
            if f"{prefix}.ffn.w3.weight" in state_dict:
                layer.ffn.w3.weight.data.copy_(state_dict[f"{prefix}.ffn.w3.weight"])


# =============================================================================
# Problem (transformer_accounting): Transformer LM resource accounting
# =============================================================================

def count_parameters(model: nn.Module) -> int:
    """
    Count the total number of parameters in a model.
    
    Args:
        model: PyTorch model
    
    Returns:
        Total number of parameters
    """
    return sum(p.numel() for p in model.parameters())


def count_flops_per_token(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
) -> int:
    """
    Estimate the number of FLOPs per token for a forward pass.
    
    This is an approximation that counts multiply-accumulate operations (MACs).
    Each MAC is typically counted as 2 FLOPs.
    
    Args:
        vocab_size: Size of vocabulary
        context_length: Maximum sequence length (used for attention)
        d_model: Model dimension
        num_layers: Number of Transformer blocks
        num_heads: Number of attention heads
        d_ff: Feed-forward hidden dimension
    
    Returns:
        Approximate FLOPs per token
    """
    # TODO: Implement FLOPs counting
    # Count FLOPs for each component (each MAC = 2 FLOPs):
    # 
    # Per layer:
    #   - Q, K, V projections: 3 * 2 * d_model * d_model
    #   - Attention scores (Q @ K^T): 2 * num_heads * d_k * context_length
    #   - Attention output (attn @ V): 2 * num_heads * context_length * d_k
    #   - Output projection: 2 * d_model * d_model
    #   - FFN (SwiGLU with w1, w2, w3): 3 * 2 * d_model * d_ff
    #   - Layer norms: 2 * 2 * d_model
    #
    # Final:
    #   - Final layer norm: 2 * d_model
    #   - Output projection: 2 * d_model * vocab_size
    #
    # Sum all components and return
    
    raise NotImplementedError("Implement count_flops_per_token")


def estimate_memory_bytes(
    vocab_size: int,
    d_model: int,
    num_layers: int,
    d_ff: int,
    dtype_bytes: int = 4,  # float32 = 4 bytes
) -> int:
    """
    Estimate the memory required to store model parameters.
    
    Args:
        vocab_size: Size of vocabulary
        d_model: Model dimension
        num_layers: Number of Transformer blocks
        d_ff: Feed-forward hidden dimension
        dtype_bytes: Bytes per parameter (4 for float32, 2 for float16)
    
    Returns:
        Approximate memory in bytes
    """
    # TODO: Implement memory estimation
    # Count parameters for each component:
    #
    # - Token embeddings: vocab_size * d_model
    # - Output projection: d_model * vocab_size
    # - Final layer norm: d_model
    #
    # Per layer:
    #   - Layer norms (2): 2 * d_model
    #   - Attention projections (Q, K, V, O): 4 * d_model * d_model
    #   - FFN (w1, w2, w3): 3 * d_model * d_ff
    #
    # Return total_params * dtype_bytes
    
    raise NotImplementedError("Implement estimate_memory_bytes")
