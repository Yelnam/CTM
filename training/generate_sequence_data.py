"""
Generate Arithmetic Sequence Datasets

Creates JSON files with arithmetic sequences for training/testing.
Separates data generation from training so we can:
- Reproduce exact results
- Compare models on identical data
- Quickly switch between dataset configurations
"""

import json
import random
import os
from datetime import datetime
from collections import Counter


def generate_sequences(seq_length=4, max_step=10, max_number=100):
    """
    Generate all valid arithmetic sequences.
    
    Args:
        seq_length: Number of input elements (output is seq_length+1th element)
        max_step: Maximum step size (1 to max_step inclusive)
        max_number: Maximum value for any number in input sequence
    
    Returns:
        List of (input_sequence, target, step) tuples
    """
    all_sequences = []
    
    for step in range(1, max_step + 1):
        # Start values where all inputs stay <= max_number
        # Last input is: start + (seq_length-1) * step <= max_number
        # Target is: start + seq_length * step (can exceed max_number slightly)
        max_start = max_number - (seq_length - 1) * step
        
        for start in range(1, max_start + 1):
            seq = [start + i * step for i in range(seq_length)]
            target = start + seq_length * step
            
            # Allow target to go up to max_number + max_step
            if target <= max_number + max_step:
                all_sequences.append({
                    'input': seq,
                    'target': target,
                    'step': step
                })
    
    return all_sequences


def create_dataset(sequences, num_samples, seed=None):
    """
    Create a dataset by sampling from sequences.
    """
    if seed is not None:
        random.seed(seed)
    
    if num_samples >= len(sequences):
        dataset = sequences.copy()
        while len(dataset) < num_samples:
            dataset.extend(random.sample(sequences, min(len(sequences), num_samples - len(dataset))))
    else:
        dataset = random.sample(sequences, num_samples)
    
    random.shuffle(dataset)
    return dataset


def save_dataset(dataset, filepath, metadata=None):
    """Save dataset to JSON file."""
    output = {
        'metadata': metadata or {},
        'sequences': dataset
    }
    
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"  Saved {len(dataset)} sequences to {filepath}")


def main():
    print("=" * 60)
    print("Arithmetic Sequence Dataset Generator")
    print("=" * 60)
    print()
    print("This creates training/test data for the sequence CTM.")
    print("You'll configure what kinds of sequences to generate.")
    print()
    
    # === Sequence Length ===
    print("-" * 60)
    print("SEQUENCE LENGTH")
    print("-" * 60)
    print("How many numbers in the input sequence?")
    print("  3 inputs: [2, 4, 6] → 8")
    print("  4 inputs: [2, 4, 6, 8] → 10")
    print("  5 inputs: [2, 4, 6, 8, 10] → 12")
    print()
    
    while True:
        val = input("Sequence length [4]: ").strip()
        seq_length = int(val) if val else 4
        if 2 <= seq_length <= 10:
            break
        print("Please enter a number between 2 and 10")
    
    print()
    
    # === Step Sizes ===
    print("-" * 60)
    print("STEP SIZES")
    print("-" * 60)
    print("What step sizes to include? (1 to max_step)")
    print("  max_step=5:  steps 1,2,3,4,5  (e.g., [2,4,6,8]→10 is step 2)")
    print("  max_step=10: steps 1,2,3,...,10  (e.g., [10,20,30,40]→50 is step 10)")
    print()
    
    while True:
        val = input("Maximum step size [5]: ").strip()
        max_step = int(val) if val else 5
        if 1 <= max_step <= 20:
            break
        print("Please enter a number between 1 and 20")
    
    print()
    
    # === Number Range ===
    print("-" * 60)
    print("NUMBER RANGE")
    print("-" * 60)
    print("What's the maximum number in the INPUT sequence?")
    print("(Output/target can go slightly higher: max_number + max_step)")
    print("  max_number=50:  inputs use 1-50, outputs up to 50+step")
    print("  max_number=100: inputs use 1-100, outputs up to 100+step")
    print()
    
    while True:
        val = input("Maximum input number [50]: ").strip()
        max_number = int(val) if val else 50
        if 10 <= max_number <= 200:
            break
        print("Please enter a number between 10 and 200")
    
    print()
    
    # === Preview ===
    print("-" * 60)
    print("PREVIEW")
    print("-" * 60)
    
    all_sequences = generate_sequences(seq_length, max_step, max_number)
    
    print(f"Configuration: {seq_length} inputs, steps 1-{max_step}, numbers 1-{max_number}")
    print(f"Total unique sequences: {len(all_sequences)}")
    print()
    
    # Distribution by step
    step_counts = Counter(s['step'] for s in all_sequences)
    print("Sequences per step size:")
    for step in sorted(step_counts.keys()):
        print(f"  Step {step}: {step_counts[step]}")
    print()
    
    # Show examples
    print("Example sequences:")
    examples = random.sample(all_sequences, min(5, len(all_sequences)))
    for ex in examples:
        print(f"  {ex['input']} → {ex['target']} (step={ex['step']})")
    print()
    
    # === Confirm or Adjust ===
    confirm = input("Look good? [Y/n]: ").strip().lower()
    if confirm == 'n':
        print("\nRestarting...\n")
        return main()
    
    print()
    
    # === Dataset Name ===
    print("-" * 60)
    print("DATASET NAME")
    print("-" * 60)
    default_name = f"seq{seq_length}_step{max_step}_num{max_number}"
    print(f"This will create: data/<name>_train.json and data/<name>_test.json")
    print()
    
    name = input(f"Dataset name [{default_name}]: ").strip() or default_name
    
    print()
    
    # === Sample Counts ===
    print("-" * 60)
    print("SAMPLE COUNTS")
    print("-" * 60)
    print("How many samples for training and testing?")
    print(f"(You have {len(all_sequences)} unique sequences)")
    print("  - More samples = more repetition = better memorization")
    print("  - Typical: 10000 train, 2000 test")
    print()
    
    default_train = min(10000, len(all_sequences) * 20)
    default_test = min(2000, len(all_sequences) * 4)
    
    val = input(f"Training samples [{default_train}]: ").strip()
    train_samples = int(val) if val else default_train
    
    val = input(f"Test samples [{default_test}]: ").strip()
    test_samples = int(val) if val else default_test
    
    print()
    
    # === Generate and Save ===
    print("-" * 60)
    print("GENERATING")
    print("-" * 60)
    
    train_dataset = create_dataset(all_sequences, train_samples, seed=42)
    test_dataset = create_dataset(all_sequences, test_samples, seed=123)
    
    metadata = {
        'seq_length': seq_length,
        'max_step': max_step,
        'max_number': max_number,
        'max_target': max_number + max_step,
        'unique_sequences': len(all_sequences),
        'generated': datetime.now().isoformat(),
    }
    
    os.makedirs('data', exist_ok=True)
    train_path = f"data/{name}_train.json"
    test_path = f"data/{name}_test.json"
    
    save_dataset(train_dataset, train_path, {**metadata, 'split': 'train', 'samples': train_samples})
    save_dataset(test_dataset, test_path, {**metadata, 'split': 'test', 'samples': test_samples})
    
    print()
    print("=" * 60)
    print("✓ DONE!")
    print("=" * 60)
    print()
    print("Files created:")
    print(f"  {train_path}")
    print(f"  {test_path}")
    print()
    print("Next step: run train_sequence.py and enter the train path when asked.")
    print()


if __name__ == "__main__":
    main()

