"""
Export trained SequenceCTM weights to JSON for browser inference.

Run this locally:
    python export_model.py models/sequence_ctm_best.pt model_weights.json

The output JSON contains all weights and model configuration.
"""

import torch
import json
import sys
import os

# Add parent directory to path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sequence_ctm import SequenceCTM


def tensor_to_list(t):
    """Convert tensor to nested Python lists."""
    return t.detach().cpu().tolist()


def export_model(model_path, output_path, config=None):
    """
    Export model weights to JSON.
    
    Args:
        model_path: Path to .pt checkpoint
        output_path: Path for output JSON
        config: Model config dict (if None, uses defaults from training log)
    """
    
    # Default config matching the trained model
    if config is None:
        config = {
            'max_number': 55,
            'seq_length': 3,
            'd_embed': 12,
            'd_model': 12,
            'd_input': 12,
            'memory_length': 4,
            'n_ticks': 6,
            'n_heads': 2,
            'nlm_hidden': 6,
        }
    
    # Create model with correct architecture
    model = SequenceCTM(
        max_number=config['max_number'],
        seq_length=config['seq_length'],
        d_embed=config['d_embed'],
        d_model=config['d_model'],
        d_input=config['d_input'],
        memory_length=config['memory_length'],
        n_ticks=config['n_ticks'],
        n_heads=config['n_heads'],
        nlm_hidden=config['nlm_hidden'],
    )
    
    # Load weights
    state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Compute derived values
    head_dim = config['d_embed'] // config['n_heads']
    d_sync = config['d_model'] * (config['d_model'] + 1) // 2
    
    # Build export structure
    export = {
        'config': {
            **config,
            'head_dim': head_dim,
            'd_sync': d_sync,
            'n_output': config['max_number'],
        },
        'weights': {
            # Embeddings
            'num_embed': tensor_to_list(model.num_embed.weight),
            'pos_embed': tensor_to_list(model.pos_embed.weight),
            
            # KV projection
            'kv_proj_weight': tensor_to_list(model.kv_proj.weight),
            'kv_proj_bias': tensor_to_list(model.kv_proj.bias),
            
            # Initial states
            'z_init': tensor_to_list(model.z_init),
            'pre_act_history_init': tensor_to_list(model.pre_act_history_init),
            
            # Synapse (Sequential with Linear, LayerNorm, ReLU, Linear)
            'synapse_0_weight': tensor_to_list(model.synapse[0].weight),
            'synapse_0_bias': tensor_to_list(model.synapse[0].bias),
            'synapse_1_weight': tensor_to_list(model.synapse[1].weight),
            'synapse_1_bias': tensor_to_list(model.synapse[1].bias),
            'synapse_3_weight': tensor_to_list(model.synapse[3].weight),
            'synapse_3_bias': tensor_to_list(model.synapse[3].bias),
            
            # NLM weights (per-neuron)
            'nlm_w1': tensor_to_list(model.nlm_w1),
            'nlm_b1': tensor_to_list(model.nlm_b1),
            'nlm_w2': tensor_to_list(model.nlm_w2),
            'nlm_b2': tensor_to_list(model.nlm_b2),
            
            # Sync decay
            'decay': tensor_to_list(model.decay),
            
            # Attention projections
            'q_proj_weight': tensor_to_list(model.q_proj.weight),
            'q_proj_bias': tensor_to_list(model.q_proj.bias),
            'attn_out_proj_weight': tensor_to_list(model.attn_out_proj.weight),
            'attn_out_proj_bias': tensor_to_list(model.attn_out_proj.bias),
            
            # Output projection
            'output_proj_weight': tensor_to_list(model.output_proj.weight),
            'output_proj_bias': tensor_to_list(model.output_proj.bias),
        }
    }
    
    # Save JSON
    with open(output_path, 'w') as f:
        json.dump(export, f)
    
    # Print summary
    print(f"Exported model to {output_path}")
    print(f"Config: {config}")
    print(f"File size: {os.path.getsize(output_path) / 1024:.1f} KB")
    
    # Verification: run a test input
    print("\nVerification:")
    test_input = torch.tensor([[2, 4, 6]])
    with torch.no_grad():
        outputs = model(test_input)
        pred = outputs[0, :, -1].argmax().item() + 1
        probs = torch.softmax(outputs[0, :, -1], dim=0)
        conf = probs[pred - 1].item()
    print(f"  Input: [2, 4, 6]")
    print(f"  Prediction: {pred} (confidence: {conf:.1%})")
    print(f"  Expected: 8")
    
    return export


def verify_against_pytorch(export, model_path):
    """
    Run a few test cases and print expected outputs for JS verification.
    """
    config = export['config']
    
    model = SequenceCTM(
        max_number=config['max_number'],
        seq_length=config['seq_length'],
        d_embed=config['d_embed'],
        d_model=config['d_model'],
        d_input=config['d_input'],
        memory_length=config['memory_length'],
        n_ticks=config['n_ticks'],
        n_heads=config['n_heads'],
        nlm_hidden=config['nlm_hidden'],
    )
    
    state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    
    test_cases = [
        [2, 4, 6],    # Step 2 -> 8
        [5, 10, 15],  # Step 5 -> 20
        [1, 2, 3],    # Step 1 -> 4
        [10, 13, 16], # Step 3 -> 19
    ]
    
    print("\nTest cases for JS verification:")
    print("-" * 50)
    
    for seq in test_cases:
        test_input = torch.tensor([seq])
        with torch.no_grad():
            outputs, internals = model(test_input, return_internals=True)
        
        # Final prediction
        final_logits = outputs[0, :, -1]
        pred = final_logits.argmax().item() + 1
        
        # Top 3 predictions
        probs = torch.softmax(final_logits, dim=0)
        top_probs, top_idx = torch.topk(probs, 3)
        
        print(f"\nInput: {seq}")
        print(f"Prediction: {pred}")
        print(f"Top 3: {[(idx.item() + 1, f'{prob.item():.4f}') for idx, prob in zip(top_idx, top_probs)]}")
        
        # Print some internal values for verification
        print(f"First tick sync[0:5]: {internals['sync'][0][0, :5].tolist()}")
        print(f"Last tick sync[0:5]: {internals['sync'][-1][0, :5].tolist()}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python export_model.py <model_path> [output_path]")
        print("Example: python export_model.py models/sequence_ctm_best.pt model_weights.json")
        sys.exit(1)
    
    model_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "model_weights.json"
    
    export = export_model(model_path, output_path)
    verify_against_pytorch(export, model_path)