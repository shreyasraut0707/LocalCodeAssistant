"""
Download CodeAlpaca-20k Dataset from Hugging Face
This is a high-quality code instruction dataset better suited for fine-tuning
"""

import os
import json
from datasets import load_dataset
from pathlib import Path

def download_codealpaca():
    """Download and save CodeAlpaca-20k dataset"""
    
    print("=" * 60)
    print("Downloading CodeAlpaca-20k Dataset")
    print("=" * 60)
    
    # Create data directory
    data_dir = Path("data/codealpaca")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n📥 Downloading from Hugging Face...")
    print("Dataset: sahil2801/CodeAlpaca-20k")
    
    try:
        # Load dataset from Hugging Face
        dataset = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
        
        print(f"\n✅ Downloaded {len(dataset)} examples")
        
        # Show example
        print("\n📋 Sample example:")
        print("-" * 60)
        example = dataset[0]
        print(f"Instruction: {example['instruction'][:100]}...")
        if example.get('input'):
            print(f"Input: {example['input'][:100]}...")
        print(f"Output: {example['output'][:100]}...")
        print("-" * 60)
        
        # Save as JSONL
        output_file = data_dir / "codealpaca_full.jsonl"
        print(f"\n💾 Saving to {output_file}...")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in dataset:
                # Format: instruction, input (optional), output
                json_line = {
                    "instruction": item["instruction"],
                    "input": item.get("input", ""),
                    "output": item["output"]
                }
                f.write(json.dumps(json_line, ensure_ascii=False) + '\n')
        
        print(f"✅ Saved {len(dataset)} examples to {output_file}")
        
        # Statistics
        print("\n📊 Dataset Statistics:")
        print(f"  Total examples: {len(dataset)}")
        print(f"  File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
        
        # Check average lengths
        avg_instruction_len = sum(len(item['instruction']) for item in dataset) / len(dataset)
        avg_output_len = sum(len(item['output']) for item in dataset) / len(dataset)
        
        print(f"  Avg instruction length: {avg_instruction_len:.0f} chars")
        print(f"  Avg output length: {avg_output_len:.0f} chars")
        
        print("\n✅ Download complete!")
        print(f"\nNext step: Run prepare_codealpaca.py to format for training")
        
        return output_file
        
    except Exception as e:
        print(f"\n❌ Error downloading dataset: {e}")
        print("\nTroubleshooting:")
        print("1. Check internet connection")
        print("2. Install datasets: pip install datasets")
        print("3. Try again in a few minutes")
        raise

if __name__ == "__main__":
    try:
        download_codealpaca()
    except KeyboardInterrupt:
        print("\n\n⚠️ Download cancelled by user")
    except Exception as e:
        print(f"\n❌ Failed: {e}")
        exit(1)
