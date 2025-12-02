"""
Prepare CodeAlpaca Dataset for Training
Formats the data and splits into train/validation sets
"""

import json
import random
from pathlib import Path
from collections import Counter

def format_instruction_prompt(instruction, input_text=""):
    """Format instruction and input into a proper prompt"""
    if input_text and input_text.strip():
        return f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""
    else:
        return f"""### Instruction:
{instruction}

### Response:
"""

def prepare_codealpaca_dataset(
    input_file="data/codealpaca/codealpaca_full.jsonl",
    output_dir="data/codealpaca",
    train_split=0.95,
    max_length=1024,
    seed=42
):
    """Prepare CodeAlpaca dataset for training"""
    
    print("=" * 60)
    print("Preparing CodeAlpaca Dataset for Training")
    print("=" * 60)
    
    input_path = Path(input_file)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"\n📂 Loading data from {input_path}...")
    examples = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            examples.append(json.loads(line))
    
    print(f"✅ Loaded {len(examples)} examples")
    
    # Filter by length (to avoid OOM on 4GB VRAM)
    print(f"\n🔍 Filtering examples (max {max_length} chars)...")
    filtered_examples = []
    
    for ex in examples:
        prompt = format_instruction_prompt(ex["instruction"], ex.get("input", ""))
        response = ex["output"]
        total_length = len(prompt) + len(response)
        
        if total_length <= max_length * 4:  # Rough estimate (chars to tokens ~= 4)
            filtered_examples.append({
                "prompt": prompt,
                "response": response,
                "instruction": ex["instruction"],
                "input": ex.get("input", ""),
                "output": ex["output"]
            })
    
    print(f"✅ Kept {len(filtered_examples)} examples (removed {len(examples) - len(filtered_examples)} too long)")
    
    # Shuffle
    random.seed(seed)
    random.shuffle(filtered_examples)
    
    # Split into train/validation
    split_idx = int(len(filtered_examples) * train_split)
    train_data = filtered_examples[:split_idx]
    val_data = filtered_examples[split_idx:]
    
    print(f"\n📊 Split Statistics:")
    print(f"  Training examples: {len(train_data)}")
    print(f"  Validation examples: {len(val_data)}")
    
    # Save training data
    train_file = output_path / "train.jsonl"
    print(f"\n💾 Saving training data to {train_file}...")
    with open(train_file, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # Save validation data
    val_file = output_path / "validation.jsonl"
    print(f"💾 Saving validation data to {val_file}...")
    with open(val_file, 'w', encoding='utf-8') as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # Show statistics
    print("\n📊 Final Dataset Statistics:")
    print("-" * 60)
    print(f"Train set: {len(train_data)} examples")
    print(f"Validation set: {len(val_data)} examples")
    print(f"Total: {len(filtered_examples)} examples")
    
    # Average lengths
    avg_prompt = sum(len(ex['prompt']) for ex in train_data) / len(train_data)
    avg_response = sum(len(ex['response']) for ex in train_data) / len(train_data)
    
    print(f"\nAverage prompt length: {avg_prompt:.0f} chars")
    print(f"Average response length: {avg_response:.0f} chars")
    print(f"Average total length: {avg_prompt + avg_response:.0f} chars")
    
    # Show sample
    print("\n📋 Sample Training Example:")
    print("-" * 60)
    sample = train_data[0]
    print("PROMPT:")
    print(sample['prompt'][:200] + "..." if len(sample['prompt']) > 200 else sample['prompt'])
    print("\nRESPONSE:")
    print(sample['response'][:200] + "..." if len(sample['response']) > 200 else sample['response'])
    print("-" * 60)
    
    print("\n✅ Dataset preparation complete!")
    print(f"\nFiles created:")
    print(f"  📄 {train_file}")
    print(f"  📄 {val_file}")
    print(f"\nNext step: Run training script with these files")
    
    return train_file, val_file

if __name__ == "__main__":
    try:
        prepare_codealpaca_dataset()
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\n⚠️ Please run download_codealpaca.py first!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise
