#!/usr/bin/env python3
"""
Detailed exploration of the VQASynth dataset
"""
from datasets import load_from_disk
import json

def show_full_conversations():
    dataset_path = "/home/isr-lab3/James/vqasynth_output/vqasynth_sample"
    dataset = load_from_disk(dataset_path)
    
    print("🎯 VQASynth Generated Dataset Analysis")
    print("="*50)
    
    for idx in range(min(2, len(dataset['train']))):  # Show first 2 examples
        print(f"\n📸 EXAMPLE {idx + 1}")
        print("-" * 30)
        
        example = dataset['train'][idx]
        
        print(f"🏷️  Tag: {example['tag']}")
        print(f"🖼️  Image size: {example['image'].size}")
        print(f"📊 Objects detected: {len(example['captions'])}")
        print(f"💬 Conversations: {len(example['messages']) // 2}")  # user/assistant pairs
        
        # Show object captions
        print(f"\n🔍 Object Captions:")
        for i, caption in enumerate(example['captions'][:5]):  # First 5 objects
            print(f"  {i+1}. {caption}")
        
        # Show the conversation in chat format
        print(f"\n💬 Spatial VQA Conversations:")
        messages = example['messages']
        
        for i in range(0, min(10, len(messages)), 2):  # Show first 5 Q&A pairs
            if i + 1 < len(messages):
                user_msg = messages[i]
                assistant_msg = messages[i + 1]
                
                # Extract text from user message
                user_text = ""
                if isinstance(user_msg['content'], list):
                    for content in user_msg['content']:
                        if content['type'] == 'text':
                            user_text = content['text']
                            break
                else:
                    user_text = user_msg['content']
                
                assistant_text = assistant_msg['content']
                
                print(f"\n  Q{i//2 + 1}: {user_text}")
                print(f"  A{i//2 + 1}: {assistant_text}")
        
        print("\n" + "="*50)

def show_dataset_statistics():
    dataset_path = "/home/isr-lab3/James/vqasynth_output/vqasynth_sample"
    dataset = load_from_disk(dataset_path)
    
    print("\n📊 Dataset Statistics:")
    print(f"Total examples: {len(dataset['train'])}")
    
    total_objects = 0
    total_conversations = 0
    tags = []
    
    for example in dataset['train']:
        total_objects += len(example['captions'])
        total_conversations += len(example['messages']) // 2
        tags.append(example['tag'])
    
    print(f"Total objects detected: {total_objects}")
    print(f"Average objects per image: {total_objects / len(dataset['train']):.1f}")
    print(f"Total Q&A pairs: {total_conversations}")
    print(f"Average Q&A per image: {total_conversations / len(dataset['train']):.1f}")
    print(f"Image tags: {set(tags)}")

if __name__ == "__main__":
    show_full_conversations()
    show_dataset_statistics()