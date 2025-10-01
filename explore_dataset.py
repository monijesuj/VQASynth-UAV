#!/usr/bin/env python3
"""
Script to explore the generated VQASynth dataset
"""
import json
from datasets import load_from_disk
import os

def explore_dataset():
    dataset_path = "/home/isr-lab3/James/vqasynth_output/vqasynth_sample"
    
    print("🔍 Loading VQASynth Dataset...")
    print(f"📂 Path: {dataset_path}")
    
    # Load the dataset
    try:
        dataset = load_from_disk(dataset_path)
        print(f"✅ Dataset loaded successfully!")
        print(f"📊 Dataset info: {dataset}")
        print(f"🎯 Number of examples: {len(dataset['train'])}")
        
        # Show dataset features
        print("\n📋 Dataset Features:")
        for feature, info in dataset['train'].features.items():
            print(f"  - {feature}: {type(info).__name__}")
        
        # Show first example
        print("\n🔬 First Example:")
        example = dataset['train'][0]
        
        for key, value in example.items():
            if key == 'image':
                print(f"  - {key}: PIL Image ({value.size})")
            elif key == 'messages':
                print(f"  - {key}: {len(value)} messages")
                if value:
                    for i, msg in enumerate(value):
                        print(f"    Message {i+1} ({msg['role']}):")
                        if isinstance(msg['content'], list):
                            for content in msg['content']:
                                if content['type'] == 'text':
                                    print(f"      📝 Text: {content['text'][:100]}...")
                                elif content['type'] == 'image':
                                    print(f"      🖼️  Image: {content['index']}")
                        else:
                            print(f"      📝 {msg['content'][:100]}...")
            elif key == 'captions':
                print(f"  - {key}: {len(value)} captions")
                for i, caption in enumerate(value[:3]):  # Show first 3
                    print(f"    Caption {i+1}: {caption[:80]}...")
            elif key == 'prompts':
                print(f"  - {key}: {len(value)} prompts")
                for i, prompt in enumerate(value[:3]):  # Show first 3
                    print(f"    Prompt {i+1}: {prompt[:80]}...")
            elif key == 'pointclouds':
                print(f"  - {key}: {len(value)} point clouds")
                for i, pc in enumerate(value[:3]):  # Show first 3
                    print(f"    Point cloud {i+1}: {pc}")
            elif key in ['embedding', 'masks', 'bboxes_or_points', 'depth_map']:
                if isinstance(value, list):
                    print(f"  - {key}: {len(value)} items")
                else:
                    print(f"  - {key}: {type(value).__name__}")
            else:
                print(f"  - {key}: {value}")
                
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")

if __name__ == "__main__":
    explore_dataset()