"""
Train SequenceCTM on arithmetic sequence prediction

Task: Given [a, a+d, a+2d, a+3d], predict a+4d
Where:
- a (start): 1-45
- d (step): 1, 2, 3, 4, 5
- All numbers in sequence must be 1-50

Example sequences:
- [2, 4, 6, 8] -> 10 (step=2)
- [5, 10, 15, 20] -> 25 (step=5)
- [7, 8, 9, 10] -> 11 (step=1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import json
import os
from collections import Counter

from sequence_ctm import SequenceCTM, count_parameters


class ArithmeticSequenceDataset(Dataset):
    """
    Dataset of arithmetic sequences, loaded from JSON file.
    
    Each sample: ([a, a+d, ...], target)
    """
    
    def __init__(self, filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.metadata = data['metadata']
        self.samples = [(s['input'], s['target'], s['step']) for s in data['sequences']]
        self.max_number = self.metadata['max_target']
        self.seq_length = self.metadata['seq_length']
        
        print(f"Loaded {len(self.samples)} samples from {filepath}")
        print(f"  Sequence length: {self.seq_length}, Max number: {self.max_number}")
        
        # Analyze distribution
        step_counts = Counter(s[2] for s in self.samples)
        print(f"  Samples by step: {dict(sorted(step_counts.items()))}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        seq, target, step = self.samples[idx]
        return (
            torch.tensor(seq, dtype=torch.long),
            torch.tensor(target - 1, dtype=torch.long),  # -1 because output indices are 0-based
            step,  # For analysis
        )


def collate_fn(batch):
    seqs = torch.stack([b[0] for b in batch])
    targets = torch.stack([b[1] for b in batch])
    steps = [b[2] for b in batch]
    return seqs, targets, steps


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    for seqs, targets, steps in dataloader:
        seqs = seqs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(seqs)  # (batch, n_output, n_ticks)
        
        # Average loss across all ticks
        batch, n_classes, n_ticks = outputs.shape
        loss = 0
        for t in range(n_ticks):
            loss += F.cross_entropy(outputs[:, :, t], targets)
        loss = loss / n_ticks
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item() * seqs.size(0)
        
        # Accuracy at final tick
        preds = outputs[:, :, -1].argmax(dim=1)
        total_correct += (preds == targets).sum().item()
        total_samples += seqs.size(0)
    
    return total_loss / total_samples, total_correct / total_samples


def evaluate(model, dataloader, device, verbose=False):
    model.eval()
    total_correct = 0
    total_samples = 0
    
    # Track accuracy by step size
    correct_by_step = Counter()
    total_by_step = Counter()
    
    examples = []
    
    with torch.no_grad():
        for seqs, targets, steps in dataloader:
            seqs = seqs.to(device)
            targets = targets.to(device)
            
            outputs = model(seqs)
            preds = outputs[:, :, -1].argmax(dim=1)
            
            for i in range(len(targets)):
                step = steps[i]
                target = targets[i].item()
                pred = preds[i].item()
                
                total_by_step[step] += 1
                if pred == target:
                    correct_by_step[step] += 1
                    total_correct += 1
                total_samples += 1
                
                if len(examples) < 15:
                    prob = F.softmax(outputs[i, :, -1], dim=0)
                    conf = prob[pred].item()
                    examples.append({
                        'seq': seqs[i].tolist(),
                        'target': target + 1,  # +1 to get actual number
                        'pred': pred + 1,
                        'step': step,
                        'conf': conf,
                        'correct': pred == target,
                    })
    
    accuracy = total_correct / total_samples if total_samples > 0 else 0
    
    if verbose:
        print("\nAccuracy by step size:")
        for step in sorted(total_by_step.keys()):
            acc = correct_by_step[step] / total_by_step[step] if total_by_step[step] > 0 else 0
            print(f"  Step {step}: {acc:.1%} ({correct_by_step[step]}/{total_by_step[step]})")
        
        print("\nExample predictions:")
        for ex in examples[:10]:
            status = "✓" if ex['correct'] else "✗"
            print(f"  {status} {ex['seq']} → pred={ex['pred']} (conf={ex['conf']:.1%}), "
                  f"target={ex['target']}, step={ex['step']}")
    
    return accuracy, correct_by_step, total_by_step


def test_sequences(model, device, max_number=110, seq_length=4):
    """Test on specific sequences."""
    
    # Generate test cases dynamically based on seq_length
    test_cases_by_length = {
        3: [
            # Step 1
            ([1, 2, 3], 4),
            ([10, 11, 12], 13),
            # Step 2
            ([2, 4, 6], 8),
            ([1, 3, 5], 7),
            ([10, 12, 14], 16),
            # Step 3
            ([3, 6, 9], 12),
            ([1, 4, 7], 10),
            # Step 4
            ([4, 8, 12], 16),
            ([2, 6, 10], 14),
            # Step 5
            ([5, 10, 15], 20),
            ([1, 6, 11], 16),
            # Higher numbers
            ([30, 32, 34], 36),
            ([40, 41, 42], 43),
        ],
        4: [
            # Step 1
            ([1, 2, 3, 4], 5),
            ([10, 11, 12, 13], 14),
            # Step 2
            ([2, 4, 6, 8], 10),
            ([1, 3, 5, 7], 9),
            ([10, 12, 14, 16], 18),
            # Step 3
            ([3, 6, 9, 12], 15),
            ([1, 4, 7, 10], 13),
            # Step 4
            ([4, 8, 12, 16], 20),
            ([2, 6, 10, 14], 18),
            # Step 5
            ([5, 10, 15, 20], 25),
            ([1, 6, 11, 16], 21),
            # Higher numbers
            ([30, 32, 34, 36], 38),
            ([40, 41, 42, 43], 44),
        ],
        5: [
            # Step 1
            ([1, 2, 3, 4, 5], 6),
            ([10, 11, 12, 13, 14], 15),
            # Step 2
            ([2, 4, 6, 8, 10], 12),
            ([1, 3, 5, 7, 9], 11),
            # Step 3
            ([3, 6, 9, 12, 15], 18),
            # Step 5
            ([5, 10, 15, 20, 25], 30),
        ],
    }
    
    # Default to length-4 tests if we don't have specific ones
    test_cases = test_cases_by_length.get(seq_length, test_cases_by_length[4])
    
    print("\n" + "=" * 60)
    print("SEQUENCE PREDICTION TEST")
    print("=" * 60)
    
    # Filter test cases to only include those within vocab AND correct length
    valid_test_cases = []
    for seq, expected in test_cases:
        if len(seq) == seq_length and all(1 <= n <= max_number for n in seq) and 1 <= expected <= max_number:
            step = seq[1] - seq[0]
            valid_test_cases.append((seq, expected, step))
    
    if not valid_test_cases:
        print(f"No valid test cases for seq_length={seq_length}, max_number={max_number}")
        return False
    
    model.eval()
    correct = 0
    
    with torch.no_grad():
        for seq, expected, step in valid_test_cases:
            seq_tensor = torch.tensor([seq], device=device)
            
            outputs, internals = model(seq_tensor, return_internals=True)
            pred = outputs[0, :, -1].argmax().item() + 1  # +1 for actual number
            probs = F.softmax(outputs[0, :, -1], dim=0)
            confidence = probs[pred - 1].item()
            
            is_correct = pred == expected
            correct += is_correct
            
            status = "✓" if is_correct else "✗"
            print(f"{status} {seq} → {pred} (conf={confidence:.1%}), expected={expected}, step={step}")
    
    print("=" * 60)
    print(f"Score: {correct}/{len(valid_test_cases)}")
    if correct == len(valid_test_cases):
        print("🎉 ALL TESTS PASSED!")
    print("=" * 60)
    
    return correct == len(valid_test_cases)


def analyze_attention(model, device, seq_length=4):
    """See where the model attends for different sequences."""
    
    test_seqs_by_length = {
        3: [
            [2, 4, 6],     # Step 2
            [5, 10, 15],   # Step 5
            [1, 2, 3],     # Step 1
        ],
        4: [
            [2, 4, 6, 8],    # Step 2
            [5, 10, 15, 20], # Step 5
            [1, 2, 3, 4],    # Step 1
        ],
        5: [
            [2, 4, 6, 8, 10],    # Step 2
            [5, 10, 15, 20, 25], # Step 5
            [1, 2, 3, 4, 5],     # Step 1
        ],
    }
    
    test_seqs = test_seqs_by_length.get(seq_length, test_seqs_by_length[4])
    
    print("\n" + "=" * 60)
    print("ATTENTION ANALYSIS")
    print("=" * 60)
    
    model.eval()
    
    with torch.no_grad():
        for seq in test_seqs:
            seq_tensor = torch.tensor([seq], device=device)
            _, internals = model(seq_tensor, return_internals=True)
            
            print(f"\nSequence: {seq} (step={seq[1]-seq[0]})")
            
            # Show attention at first and last tick
            for tick_idx, tick_name in [(0, "First"), (-1, "Last")]:
                attn = internals['attention_weights'][tick_idx][0]  # (n_heads, seq_len)
                
                print(f"  {tick_name} tick attention:")
                for head in range(attn.shape[0]):
                    weights = attn[head].numpy()
                    bars = ''.join(['█' if w > 0.3 else '▄' if w > 0.2 else '▁' for w in weights])
                    print(f"    Head {head}: {bars} ({weights.round(2).tolist()})")


def prompt_config():
    """Interactive configuration with sensible defaults."""
    
    config = {
        'd_model': 16,
        'n_ticks': 10,
        'n_heads': 2,
        'memory_length': 8,
        'nlm_hidden': 8,
        'epochs': 100,
        'batch_size': 64,
        'lr': 3e-4,
    }
    
    while True:
        print("\n" + "=" * 60)
        print("Sequence CTM - Configuration")
        print("Press Enter to accept [current values]")
        print("=" * 60)
        
        # === Model Architecture ===
        print("\n--- Model Architecture ---")
        
        val = input(f"Neurons (d_model) [{config['d_model']}]: ").strip()
        if val:
            config['d_model'] = int(val)
        config['d_embed'] = config['d_model']
        config['d_input'] = config['d_model']
        
        val = input(f"Thinking ticks [{config['n_ticks']}]: ").strip()
        if val:
            config['n_ticks'] = int(val)
        
        val = input(f"Attention heads [{config['n_heads']}]: ").strip()
        if val:
            config['n_heads'] = int(val)
        
        # === Per-Neuron NLM ===
        print("\n--- Per-Neuron NLM ---")
        
        val = input(f"Memory length [{config['memory_length']}]: ").strip()
        if val:
            config['memory_length'] = int(val)
        
        val = input(f"NLM hidden dim [{config['nlm_hidden']}]: ").strip()
        if val:
            config['nlm_hidden'] = int(val)
        
        # === Training ===
        print("\n--- Training ---")
        
        val = input(f"Epochs [{config['epochs']}]: ").strip()
        if val:
            config['epochs'] = int(val)
        
        val = input(f"Batch size [{config['batch_size']}]: ").strip()
        if val:
            config['batch_size'] = int(val)
        
        val = input(f"Learning rate [{config['lr']}]: ").strip()
        if val:
            config['lr'] = float(val)
        
        # === Fixed/Derived ===
        config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Calculate parameter count estimate (using max_number=110 as rough estimate)
        max_num_est = 110  # Rough estimate for parameter counting
        d = config['d_model']
        n_sync = d * (d + 1) // 2
        est_params = (
            55 * d +                          # num_embed
            4 * d +                           # pos_embed
            d * config['n_heads'] * (d // config['n_heads']) * 2 +  # kv_proj
            d +                               # z_init
            d * config['memory_length'] +     # pre_act_history_init
            (d + d) * d * 2 + d * 2 + d +     # synapse (rough)
            d * config['memory_length'] * config['nlm_hidden'] +  # nlm_w1
            d * config['nlm_hidden'] +        # nlm_b1
            d * config['nlm_hidden'] +        # nlm_w2
            d +                               # nlm_b2
            n_sync +                          # decay
            n_sync * config['n_heads'] * (d // config['n_heads']) +  # q_proj
            config['n_heads'] * (d // config['n_heads']) * d +  # attn_out_proj
            n_sync * 55                       # output_proj
        )
        
        # === Summary ===
        print("\n" + "=" * 60)
        print("Configuration Summary:")
        print("=" * 60)
        print(f"  Neurons: {config['d_model']}")
        print(f"  Ticks: {config['n_ticks']}")
        print(f"  Attention heads: {config['n_heads']}")
        print(f"  Memory length: {config['memory_length']}")
        print(f"  NLM hidden: {config['nlm_hidden']}")
        print(f"  Epochs: {config['epochs']}")
        print(f"  Batch size: {config['batch_size']}")
        print(f"  Learning rate: {config['lr']}")
        print(f"  Device: {config['device']}")
        print(f"  Est. parameters: ~{est_params:,}")
        
        confirm = input("\nProceed? [Y/n]: ").strip().lower()
        if confirm != 'n':
            break
        print("\n--- Let's try again ---")
    
    return config


def main():
    print("=" * 60)
    print("Sequence CTM - Arithmetic Sequence Prediction")
    print("=" * 60)
    
    # First, get data file paths
    print("\n--- Dataset Selection ---")
    
    # List available datasets
    data_dir = 'data'
    if os.path.exists(data_dir):
        files = [f for f in os.listdir(data_dir) if f.endswith('_train.json')]
        if files:
            print("Available datasets:")
            for i, f in enumerate(files):
                name = f.replace('_train.json', '')
                print(f"  {i+1}. {name}")
    
    train_path = input("Train data path [data/seq_small_train.json]: ").strip() or "data/seq_small_train.json"
    test_path = train_path.replace('_train.json', '_test.json')
    
    if not os.path.exists(train_path):
        print(f"\n⚠ File not found: {train_path}")
        print("Run generate_sequence_data.py first to create datasets.")
        return
    
    # Load datasets to get metadata
    print(f"\nLoading datasets...")
    train_dataset = ArithmeticSequenceDataset(train_path)
    test_dataset = ArithmeticSequenceDataset(test_path)
    
    # Now get model config, with dataset info
    config = prompt_config()
    
    # Add dataset info to config
    config['max_number'] = train_dataset.max_number
    config['seq_length'] = train_dataset.seq_length
    
    print(f"\nDevice: {config['device']}")
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                              shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], 
                             shuffle=False, collate_fn=collate_fn)
    
    # Create model
    print("\nCreating model...")
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
    ).to(config['device'])
    
    print(f"Parameters: {count_parameters(model):,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, config['epochs'])
    
    print("\nStarting training...")
    print("-" * 60)
    
    best_accuracy = 0
    
    for epoch in range(config['epochs']):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, config['device'])
        test_acc, _, _ = evaluate(model, test_loader, config['device'], verbose=False)
        scheduler.step()
        
        marker = ""
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            marker = " ★"
            torch.save(model.state_dict(), "models/sequence_ctm_best.pt")
        
        print(f"Epoch {epoch+1:3d}/{config['epochs']}: "
              f"Loss={train_loss:.4f} Train={train_acc:.1%} Test={test_acc:.1%}{marker}")
        
        if (epoch + 1) % 20 == 0:
            evaluate(model, test_loader, config['device'], verbose=True)
            test_sequences(model, config['device'], config['max_number'], config['seq_length'])
    
    print("-" * 60)
    print(f"Training complete! Best accuracy: {best_accuracy:.1%}")
    
    # Final evaluation
    print("\nFinal evaluation:")
    model.load_state_dict(torch.load("models/sequence_ctm_best.pt", weights_only=True))
    final_acc, correct_by_step, total_by_step = evaluate(model, test_loader, config['device'], verbose=True)
    test_sequences(model, config['device'], config['max_number'], config['seq_length'])
    analyze_attention(model, config['device'], config['seq_length'])
    
    # Save log
    os.makedirs('logs', exist_ok=True)
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"logs/train_{timestamp}.txt"
    
    with open(log_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("SEQUENCE CTM TRAINING LOG\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("DATASET\n")
        f.write("-" * 30 + "\n")
        f.write(f"Train file: {train_path}\n")
        f.write(f"Sequence length: {config['seq_length']}\n")
        f.write(f"Max number: {config['max_number']}\n")
        f.write(f"Train samples: {len(train_dataset)}\n")
        f.write(f"Test samples: {len(test_dataset)}\n\n")
        
        f.write("MODEL\n")
        f.write("-" * 30 + "\n")
        f.write(f"Neurons (d_model): {config['d_model']}\n")
        f.write(f"Ticks: {config['n_ticks']}\n")
        f.write(f"Attention heads: {config['n_heads']}\n")
        f.write(f"Memory length: {config['memory_length']}\n")
        f.write(f"NLM hidden: {config['nlm_hidden']}\n")
        f.write(f"Parameters: {count_parameters(model):,}\n\n")
        
        f.write("TRAINING\n")
        f.write("-" * 30 + "\n")
        f.write(f"Epochs: {config['epochs']}\n")
        f.write(f"Batch size: {config['batch_size']}\n")
        f.write(f"Learning rate: {config['lr']}\n")
        f.write(f"Device: {config['device']}\n\n")
        
        f.write("RESULTS\n")
        f.write("-" * 30 + "\n")
        f.write(f"Best test accuracy: {best_accuracy:.1%}\n")
        f.write(f"Final test accuracy: {final_acc:.1%}\n\n")
        
        f.write("Accuracy by step size:\n")
        for step in sorted(total_by_step.keys()):
            acc = correct_by_step[step] / total_by_step[step] if total_by_step[step] > 0 else 0
            f.write(f"  Step {step}: {acc:.1%} ({correct_by_step[step]}/{total_by_step[step]})\n")
        
        f.write("\n" + "=" * 60 + "\n")
    
    print(f"\n✓ Log saved to {log_path}")


if __name__ == "__main__":
    main()
