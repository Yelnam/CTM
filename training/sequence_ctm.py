"""
Sequence CTM - Predict the next number in an arithmetic sequence

Task: Given [2, 4, 6, 8], predict 10
      Given [5, 10, 15, 20], predict 25

This is a good fit for CTM because:
- The model needs to process the sequence over ticks
- It needs to detect a pattern (the step size)
- Then apply that pattern to predict

Constraints:
- Arithmetic sequences only (constant step)
- Steps: 1, 2, 3, 4, 5
- Numbers: 1-50
- Sequence length: 4 inputs → predict 5th
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SequenceCTM(nn.Module):
    """
    Tiny CTM for arithmetic sequence prediction.
    
    Architecture mirrors TinyCharacterCTM but adapted for numbers:
    - Number embeddings (1-50) instead of character embeddings
    - Output is a number (1-55, to allow predictions beyond training range)
    - No query embedding needed - just predict next in sequence
    """
    
    def __init__(
        self,
        max_number=110,       # Highest number we can output (100 + max step of 10)
        seq_length=4,         # Input sequence length
        d_embed=16,           # Number embedding dimension
        d_model=16,           # Number of neurons
        d_input=16,           # Attention output dimension
        memory_length=8,      # NLM history length
        n_ticks=10,           # Thinking steps
        n_heads=2,            # Attention heads
        nlm_hidden=8,         # Hidden dim per NLM
    ):
        super().__init__()
        
        self.max_number = max_number
        self.seq_length = seq_length
        self.d_model = d_model
        self.d_embed = d_embed
        self.d_input = d_input
        self.memory_length = memory_length
        self.n_ticks = n_ticks
        self.n_heads = n_heads
        self.n_output = max_number  # Output is a number from 1 to max_number
        self.head_dim = d_embed // n_heads
        self.nlm_hidden = nlm_hidden
        
        # Number embeddings (0 = padding, 1-55 = actual numbers)
        self.num_embed = nn.Embedding(max_number + 1, d_embed)
        self.pos_embed = nn.Embedding(seq_length, d_embed)
        
        # Project embeddings to KV for attention
        self.kv_proj = nn.Linear(d_embed, n_heads * self.head_dim * 2)
        
        # Learnable initial states
        self.z_init = nn.Parameter(torch.randn(d_model) * 0.1)
        self.pre_act_history_init = nn.Parameter(torch.randn(d_model, memory_length) * 0.1)
        
        # Synapse: combines neuron state with attention output
        self.synapse = nn.Sequential(
            nn.Linear(d_model + d_input, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
        )
        
        # NLMs - each neuron has private weights
        self.nlm_w1 = nn.Parameter(torch.randn(d_model, memory_length, nlm_hidden) * 0.1)
        self.nlm_b1 = nn.Parameter(torch.zeros(d_model, nlm_hidden))
        self.nlm_w2 = nn.Parameter(torch.randn(d_model, nlm_hidden, 1) * 0.1)
        self.nlm_b2 = nn.Parameter(torch.zeros(d_model, 1))
        
        # Synchronization indices
        self.d_sync = d_model * (d_model + 1) // 2
        idx_i, idx_j = torch.triu_indices(d_model, d_model)
        self.register_buffer('sync_idx_i', idx_i)
        self.register_buffer('sync_idx_j', idx_j)
        self.decay = nn.Parameter(torch.zeros(self.d_sync))
        
        # Attention query - built from sync (no external query needed)
        self.q_proj = nn.Linear(self.d_sync, n_heads * self.head_dim)
        self.attn_out_proj = nn.Linear(n_heads * self.head_dim, d_input)
        
        # Output projection: sync → predicted number
        self.output_proj = nn.Linear(self.d_sync, self.n_output)
        
    def nlm_forward(self, history):
        """Each neuron processes its own history."""
        h = torch.einsum('bdm,dmh->bdh', history, self.nlm_w1) + self.nlm_b1
        h = F.relu(h)
        out = torch.einsum('bdh,dho->bdo', h, self.nlm_w2) + self.nlm_b2
        return out.squeeze(-1)
    
    def compute_sync(self, post_act_history):
        """Compute synchronization between neuron pairs."""
        batch, T, D = post_act_history.shape
        device = post_act_history.device
        
        z_i = post_act_history[:, :, self.sync_idx_i]
        z_j = post_act_history[:, :, self.sync_idx_j]
        
        t_back = torch.arange(T - 1, -1, -1, device=device, dtype=torch.float32)
        decay_rates = F.softplus(self.decay)
        weights = torch.exp(-decay_rates.unsqueeze(0) * t_back.unsqueeze(-1))
        
        products = z_i * z_j
        weighted = products * weights.unsqueeze(0)
        sync = weighted.sum(dim=1) / (weights.sum(dim=0, keepdim=True) + 1e-8)
        
        return sync
    
    def forward(self, numbers, return_internals=False):
        """
        Forward pass.
        
        Args:
            numbers: (batch, seq_length) - sequence of numbers (1-50)
            return_internals: whether to return attention weights, activations, etc.
            
        Returns:
            outputs: (batch, n_output, n_ticks) - predicted number distribution at each tick
        """
        batch, seq_len = numbers.shape
        device = numbers.device
        
        # Embed numbers with positions
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, -1)
        num_emb = self.num_embed(numbers) + self.pos_embed(positions)
        
        # Project to keys and values
        kv_features = self.kv_proj(num_emb)
        KV = kv_features.view(batch, seq_len, self.n_heads, self.head_dim * 2)
        KV = KV.permute(0, 2, 1, 3)
        K, V = KV.chunk(2, dim=-1)
        
        # Initialize neuron states
        z = self.z_init.unsqueeze(0).expand(batch, -1).clone()
        pre_act_history = self.pre_act_history_init.unsqueeze(0).expand(batch, -1, -1).clone()
        
        # Track post-activation history for sync
        post_act_history = torch.zeros(batch, self.n_ticks + 1, self.d_model, device=device)
        post_act_history[:, 0] = z
        
        outputs = torch.zeros(batch, self.n_output, self.n_ticks, device=device)
        
        if return_internals:
            internals = {
                'post_activations': [],
                'sync': [],
                'attention_weights': [],
            }
        else:
            internals = None
        
        for t in range(self.n_ticks):
            # Compute sync from history
            sync = self.compute_sync(post_act_history[:, :t + 1])
            
            # Build attention query from sync
            Q = self.q_proj(sync).view(batch, self.n_heads, 1, self.head_dim)
            
            # Attention over sequence
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
            attn_weights = F.softmax(scores, dim=-1)
            
            attn_out = torch.matmul(attn_weights, V)
            attn_out = attn_out.squeeze(2).reshape(batch, -1)
            attn_out = self.attn_out_proj(attn_out)
            
            # Synapse
            pre_act = self.synapse(torch.cat([z, attn_out], dim=-1))
            
            # Update pre-activation history
            pre_act_history = torch.cat([
                pre_act_history[:, :, 1:],
                pre_act.unsqueeze(-1)
            ], dim=-1)
            
            # NLM
            z = self.nlm_forward(pre_act_history)
            post_act_history[:, t + 1] = z
            
            # Output from sync
            final_sync = self.compute_sync(post_act_history[:, :t + 2])
            logits = self.output_proj(final_sync)
            outputs[:, :, t] = logits
            
            if return_internals:
                internals['post_activations'].append(z.detach().cpu())
                internals['sync'].append(final_sync.detach().cpu())
                internals['attention_weights'].append(attn_weights.squeeze(2).detach().cpu())
        
        return (outputs, internals) if return_internals else outputs


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("Testing SequenceCTM...")
    
    model = SequenceCTM(
        max_number=55,
        seq_length=4,
        d_embed=16,
        d_model=16,
        d_input=16,
        memory_length=8,
        n_ticks=10,
        n_heads=2,
        nlm_hidden=8,
    )
    
    print(f"Parameters: {count_parameters(model):,}")
    
    # Test forward pass
    # Sequence: [2, 4, 6, 8] -> should predict 10
    test_seq = torch.tensor([[2, 4, 6, 8]])
    
    outputs, internals = model(test_seq, return_internals=True)
    
    print(f"Output shape: {outputs.shape}")  # (1, 55, 10)
    
    # Get prediction at final tick
    pred = outputs[0, :, -1].argmax().item() + 1  # +1 because index 0 = number 1
    print(f"Input: [2, 4, 6, 8]")
    print(f"Predicted next: {pred}")
    print(f"Expected: 10")
    
    print("\n✓ SequenceCTM working!")
