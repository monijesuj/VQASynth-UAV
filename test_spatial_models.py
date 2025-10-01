#!/usr/bin/env python3
"""
Test local spatial reasoning models
Quick verification that SpaceOm and SpaceThinker work
"""
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image
import os

def test_model(model_path):
    """Test loading and basic inference with a spatial model"""
    print(f"\n🔄 Testing {model_path}...")
    
    try:
        # Load model using the correct Qwen2.5-VL class
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            trust_remote_code=True
        )
        
        print(f"✅ Model loaded successfully!")
        print(f"📊 Device: {next(model.parameters()).device}")
        print(f"💾 Memory usage: {torch.cuda.memory_allocated() / 1e9:.1f}GB" if torch.cuda.is_available() else "CPU mode")
        
        # Test with a simple question (using the working pattern from transformer_inference.py)
        test_question = "What is spatial reasoning?"
        
        # Format input like in your working transformer_inference.py
        chat = [
            {"role": "user", "content": [{"type": "text", "text": test_question}]}
        ]
        
        text_input = processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        
        # Tokenize (no image for this test)
        inputs = processor(text=[text_input], return_tensors="pt").to(model.device)
        
        # Generate response
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=50)
        
        response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        print(f"🤔 Test response: {response[:100]}...")
        return True
        
    except Exception as e:
        print(f"❌ Failed: {str(e)}")
        return False

def main():
    print("🎯 Testing Local Spatial Reasoning Models")
    print("=" * 50)
    
    models_to_test = [
        "./SpaceOm",
        "./SpaceThinker-Qwen2.5VL-3B"
    ]
    
    working_models = []
    
    for model_path in models_to_test:
        if os.path.exists(model_path):
            if test_model(model_path):
                working_models.append(model_path)
        else:
            print(f"⚠️  Model directory not found: {model_path}")
    
    print(f"\n🎉 Summary:")
    print(f"Working models: {len(working_models)}")
    for model in working_models:
        print(f"  ✅ {model}")
    
    if not working_models:
        print("❌ No working spatial reasoning models found!")
        print("💡 Make sure SpaceOm and SpaceThinker-Qwen2.5VL-3B are in the current directory")
    else:
        print(f"🚀 Ready to use spatial reasoning with {len(working_models)} model(s)!")

if __name__ == "__main__":
    main()