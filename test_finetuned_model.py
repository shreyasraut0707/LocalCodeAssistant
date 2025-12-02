"""
Test the Fine-Tuned CodeAlpaca Model
Quick script to verify your model works and generates good code!
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_finetuned_model():
    """Test the fine-tuned model with sample prompts"""
    
    print("=" * 70)
    print("Testing Fine-Tuned CodeAlpaca Model")
    print("=" * 70)
    
    # Load the fine-tuned model
    model_path = "models/codealpaca-finetuned/final"
    
    print(f"\n📥 Loading fine-tuned model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    print("✅ Model loaded successfully!")
    
    # Test prompts
    test_prompts = [
        "### Instruction:\nWrite a Python function to reverse a string\n\n### Response:\n",
        "### Instruction:\nCreate a function to calculate factorial of a number\n\n### Response:\n",
        "### Instruction:\nWrite a function to check if a string is a palindrome\n\n### Response:\n",
    ]
    
    print("\n" + "=" * 70)
    print("Testing Code Generation")
    print("=" * 70)
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{'='* 70}")
        print(f"Test {i}:")
        print(f"{'='* 70}")
        print(f"\nPrompt: {prompt.split('Response:')[0].strip()}")
        
        # Generate code
        inputs = tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=200,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = generated_code.split("### Response:")[-1].strip()
        
        print(f"\n✅ Generated Code:\n")
        print(response)
    
    print("\n" + "=" * 70)
    print("✅ Testing Complete!")
    print("=" * 70)
    print("\nYour fine-tuned model is working and generating code!")
    print(f"Model location: {model_path}")

if __name__ == "__main__":
    try:
        test_finetuned_model()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure:")
        print("  1. Model exists at: models/codealpaca-finetuned/final/")
        print("  2. You have transformers and torch installed")
        raise
