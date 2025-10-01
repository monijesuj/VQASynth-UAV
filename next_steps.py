#!/usr/bin/env python3
"""
Next steps guide for using your VQASynth dataset
"""

def print_next_steps():
    print("🎯 VQASynth Dataset - Next Steps")
    print("=" * 50)
    
    print("\n1. 🔬 ANALYZE YOUR DATASET")
    print("   • Use Jupyter notebooks to visualize point clouds")
    print("   • Examine depth maps and object masks")
    print("   • Analyze spatial reasoning patterns")
    
    print("\n2. 📚 EXPAND YOUR DATASET")
    print("   • Modify config/config.yaml to process more images")
    print("   • Change dataset_name to target different HF datasets")
    print("   • Adjust include_tags/exclude_tags for specific domains")
    
    print("\n3. 🧠 TRAIN A SPATIAL VLM")
    print("   • Use your dataset to fine-tune models like:")
    print("     - Qwen2.5-VL")
    print("     - LLaVA")
    print("     - InstructBLIP")
    print("   • Follow SpatialVLM training methodology")
    
    print("\n4. 🎮 INTERACTIVE DEMO")
    print("   • Create Gradio app for spatial reasoning")
    print("   • Test on new images")
    print("   • Compare with existing spatial VLMs")
    
    print("\n5. 📤 SHARE YOUR WORK")
    print("   • Push dataset to Hugging Face Hub")
    print("   • Create model cards and documentation")
    print("   • Contribute to open spatial reasoning research")
    
    print("\n" + "=" * 50)
    
    print("\n🔧 Configuration Tips:")
    print("• Edit config/config.yaml for different datasets:")
    print("  - dataset_name: 'your-dataset-name'")
    print("  - include_tags: 'indoor,outdoor,robot'")
    print("  - target_repo_name: 'your-username/spatial-dataset'")
    
    print("\n📊 Dataset Quality:")
    print("• Your dataset includes:")
    print("  ✅ Accurate distance measurements")
    print("  ✅ Size and spatial comparisons") 
    print("  ✅ Multi-modal data (images + point clouds)")
    print("  ✅ Conversation format for training")
    
    print("\n🎉 Congratulations!")
    print("You've successfully created a spatial reasoning dataset!")
    print("This can be used to enhance VLM spatial understanding capabilities.")

if __name__ == "__main__":
    print_next_steps()