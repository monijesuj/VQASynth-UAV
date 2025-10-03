#!/usr/bin/env python3
"""
Enhanced Semantic Segmentation Launcher
Simple launcher script with options for different modes
"""

import sys
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Enhanced Semantic Segmentation Launcher")
    parser.add_argument("--mode", "-m", choices=["gui", "batch", "test"], default="gui",
                       help="Mode to run: gui (interactive), batch (command line), test (test installation)")
    parser.add_argument("--input", "-i", help="Input directory for batch mode")
    parser.add_argument("--output", "-o", help="Output directory for batch mode")
    parser.add_argument("--prompt", "-p", help="Segmentation prompt for batch mode")
    parser.add_argument("--device", "-d", default="cuda", help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    if args.mode == "gui":
        print("🚀 Launching Enhanced Semantic Segmentation GUI...")
        try:
            from enhanced_semantic_segmentation import create_enhanced_gradio_interface
            demo = create_enhanced_gradio_interface()
            demo.launch(server_name="0.0.0.0", server_port=7861)
        except ImportError as e:
            print(f"❌ Error importing enhanced segmentation: {e}")
            print("Please run the setup script first: ./setup_enhanced_segmentation.sh")
            sys.exit(1)
    
    elif args.mode == "batch":
        if not args.input or not args.output or not args.prompt:
            print("❌ Batch mode requires --input, --output, and --prompt arguments")
            print("Example: python launch_enhanced_segmentation.py --mode batch -i input/ -o output/ -p 'floor'")
            sys.exit(1)
        
        print(f"🚀 Running batch segmentation...")
        print(f"   Input: {args.input}")
        print(f"   Output: {args.output}")
        print(f"   Prompt: {args.prompt}")
        
        try:
            from batch_segmentation import BatchSegmentationProcessor
            processor = BatchSegmentationProcessor(device=args.device)
            results = processor.process_batch(args.input, args.output, args.prompt)
            print("✅ Batch processing complete!")
        except ImportError as e:
            print(f"❌ Error importing batch processor: {e}")
            print("Please run the setup script first: ./setup_enhanced_segmentation.sh")
            sys.exit(1)
    
    elif args.mode == "test":
        print("🧪 Testing Enhanced Segmentation Installation...")
        try:
            from test_enhanced_segmentation import main as test_main
            test_main()
        except ImportError:
            print("❌ Test script not found. Running basic import test...")
            
            # Basic import test
            try:
                import torch
                print(f"✅ PyTorch: {torch.__version__}")
                print(f"   CUDA available: {torch.cuda.is_available()}")
            except ImportError:
                print("❌ PyTorch not installed")
            
            try:
                import clip
                print("✅ CLIP available")
            except ImportError:
                print("❌ CLIP not installed")
            
            try:
                from sam2.build_sam import build_sam2
                print("✅ SAM2 available")
            except ImportError:
                print("❌ SAM2 not installed")
            
            try:
                from transformers import CLIPSegProcessor
                print("✅ CLIPSeg available")
            except ImportError:
                print("❌ CLIPSeg not installed")

if __name__ == "__main__":
    main()
