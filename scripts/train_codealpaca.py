"""
Fine-tune Model on CodeAlpaca-20k Dataset
Optimized for GTX 1650 (4GB VRAM)
"""

import os
import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CodeAlpacaDataset:
    """Dataset handler for CodeAlpaca format"""
    
    def __init__(self, file_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        # Load data
        logger.info(f"Loading data from {file_path}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.examples.append(json.loads(line))
        
        logger.info(f"Loaded {len(self.examples)} examples")
    
    def tokenize_example(self, example):
        """Tokenize a single example"""
        # Combine prompt and response
        full_text = example['prompt'] + example['response']
        
        # Tokenize
        tokenized = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None
        )
        
        # Create labels (same as input_ids for causal LM)
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    def get_dataset(self):
        """Convert to HuggingFace Dataset"""
        # Tokenize all examples
        tokenized_data = [self.tokenize_example(ex) for ex in self.examples]
        
        # Convert to Dataset
        dataset = Dataset.from_list(tokenized_data)
        return dataset

def train_on_codealpaca(
    train_file="data/codealpaca/train.jsonl",
    val_file="data/codealpaca/validation.jsonl",
    model_name="Salesforce/codegen-350M-mono",
    output_dir="models/codealpaca-finetuned",
    max_length=512,  # Restored to 512 for better learning
    batch_size=1,
    gradient_accumulation_steps=8,  # Reduced for faster iterations
    num_epochs=3,
    learning_rate=2e-4,
    save_steps=500,
):
    """Fine-tune on CodeAlpaca dataset"""
    
    print("=" * 70)
    print("Fine-Tuning on CodeAlpaca-20k Dataset (OPTIMIZED)")
    print("=" * 70)
    
    # Check CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🔧 Device: {device}")
    
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"   GPU: {gpu_name}")
        print(f"   VRAM: {gpu_memory:.1f} GB")
        
        # Check if BF16 is supported (better than FP16)
        bf16_support = torch.cuda.is_bf16_supported()
        print(f"   BF16 support: {bf16_support}")
    else:
        bf16_support = False
    
    # Load tokenizer
    print(f"\n📥 Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    print(f"📥 Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True
    )
    
    # Move model to device with mixed precision
    if device == "cuda":
        model = model.to(device)
        # Use BF16 if available, otherwise FP16
        if bf16_support:
            model = model.to(torch.bfloat16)
            print("   Using BF16 mixed precision (optimal!)")
        else:
            model = model.to(torch.float16)
            print("   Using FP16 mixed precision")
    
    model_params = sum(p.numel() for p in model.parameters()) / 1_000_000
    print(f"   Model size: {model_params:.0f}M parameters")
    
    # Prepare datasets
    print(f"\n📊 Preparing datasets...")
    train_dataset_handler = CodeAlpacaDataset(train_file, tokenizer, max_length)
    val_dataset_handler = CodeAlpacaDataset(val_file, tokenizer, max_length)
    
    train_dataset = train_dataset_handler.get_dataset()
    val_dataset = val_dataset_handler.get_dataset()
    
    print(f"   Training examples: {len(train_dataset)}")
    print(f"   Validation examples: {len(val_dataset)}")
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM, not masked LM
    )
    
    # Training arguments (OPTIMIZED for speed + quality)
    print(f"\n⚙️ Training Configuration (OPTIMIZED):")
    print(f"   Precision: {'BF16' if bf16_support else 'FP16'} (mixed precision)")
    print(f"   Batch size: {batch_size}")
    print(f"   Gradient accumulation: {gradient_accumulation_steps}")
    print(f"   Effective batch size: {batch_size * gradient_accumulation_steps}")
    print(f"   Epochs: {num_epochs}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Max length: {max_length}")
    print(f"   Expected speedup: 4-5x faster (6-8 hours instead of 28)")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_steps=100,
        logging_steps=50,
        save_steps=save_steps,
        eval_steps=save_steps,
        save_total_limit=3,
        eval_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=bf16_support if device == "cuda" else False,  # Use BF16 if available
        fp16=not bf16_support if device == "cuda" else False,  # Otherwise FP16
        fp16_full_eval=False,  # Avoid FP16 in evaluation to prevent errors
        gradient_checkpointing=False,  # Keep disabled for stability
        optim="adamw_torch",
        report_to=None,
        remove_unused_columns=False,
        dataloader_num_workers=0,  # Avoid multiprocessing issues
        # Better learning rate schedule
        lr_scheduler_type="cosine",
        save_safetensors=True,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("\n" + "=" * 70)
    
    # Check for existing checkpoints to resume from
    checkpoint_path = None
    output_path = Path(output_dir)
    if output_path.exists():
        checkpoints = sorted(output_path.glob("checkpoint-*"))
        if checkpoints:
            checkpoint_path = str(checkpoints[-1])
            print(f"🔄 Found checkpoint: {checkpoint_path}")
            print(f"   Resuming training from this checkpoint...")
    
    if checkpoint_path:
        print("🚀 Resuming Training...")
    else:
        print("🚀 Starting Training from Beginning...")
    print("=" * 70)
    print("\n⏰ Training will auto-save every 500 steps")
    print("💡 You can safely stop (Ctrl+C) and resume later!")
    print("📊 Progress will be shown below\n")
    
    try:
        # Resume from checkpoint if available, otherwise start fresh
        trainer.train(resume_from_checkpoint=checkpoint_path)
        
        print("\n" + "=" * 70)
        print("✅ Training Complete!")
        print("=" * 70)
        
        # Save final model
        final_model_dir = Path(output_dir) / "final"
        print(f"\n💾 Saving final model to {final_model_dir}...")
        trainer.save_model(final_model_dir)
        tokenizer.save_pretrained(final_model_dir)
        
        print(f"\n✅ Model saved successfully!")
        print(f"\nTo use your fine-tuned model:")
        print(f"  1. Update model path in your code to: {final_model_dir.absolute()}")
        print(f"  2. Run your inference script")
        print(f"  3. Test with code generation prompts")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        raise

if __name__ == "__main__":
    try:
        # Check if data files exist
        train_file = Path("data/codealpaca/train.jsonl")
        val_file = Path("data/codealpaca/validation.jsonl")
        
        if not train_file.exists() or not val_file.exists():
            print("❌ Dataset files not found!")
            print("\nPlease run these scripts first:")
            print("  1. python scripts/download_codealpaca.py")
            print("  2. python scripts/prepare_codealpaca.py")
            exit(1)
        
        # Start training
        train_on_codealpaca()
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise
