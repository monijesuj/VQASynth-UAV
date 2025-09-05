#!/usr/bin/env python3
"""
Quick test script for VQASynth Gradio interface
"""

import time
import os
import sys
from pathlib import Path

def test_gradio_app():
    """Test the Gradio app without actually launching the interface"""
    print("🧪 Testing Gradio App Components...")
    
    try:
        # Add the project root to Python path
        project_root = Path(__file__).parent.absolute()
        sys.path.insert(0, str(project_root))
        
        # Import the app
        from examples.app_gradio import (
            depth, localizer, spatial_scene_constructor, 
            prompt_generator, run_vqasynth_pipeline
        )
        
        print("✅ Successfully imported Gradio app components")
        
        # Test with example image
        example_image = "examples/assets/warehouse_rgb.jpg"
        if not os.path.exists(example_image):
            print(f"❌ Example image not found: {example_image}")
            return False
        
        from PIL import Image
        import tempfile
        
        print(f"📷 Loading test image: {example_image}")
        image = Image.open(example_image).convert("RGB")
        
        # Time the full pipeline
        with tempfile.TemporaryDirectory() as temp_dir:
            start_time = time.time()
            
            print("🚀 Running VQASynth pipeline...")
            obj_file, selected_prompt = run_vqasynth_pipeline(image, temp_dir)
            
            end_time = time.time()
            duration = end_time - start_time
            
            print(f"✅ Pipeline completed in {duration:.3f} seconds")
            print(f"📁 Generated 3D model: {obj_file}")
            print(f"💭 Selected prompt: {selected_prompt[:100]}...")
            
            # Check if output file exists
            if os.path.exists(obj_file):
                file_size = os.path.getsize(obj_file)
                print(f"📊 Output file size: {file_size} bytes")
            else:
                print("⚠️  Output file was not created")
            
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all dependencies are installed:")
        print("pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"❌ Error running pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False


def launch_gradio_interface():
    """Launch the Gradio interface for interactive testing"""
    print("🌐 Launching Gradio Interface...")
    
    try:
        from examples.app_gradio import build_demo
        
        demo = build_demo()
        print("✅ Gradio interface built successfully")
        print("🚀 Launching on http://localhost:7860")
        
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            debug=True
        )
        
    except Exception as e:
        print(f"❌ Failed to launch Gradio interface: {e}")
        import traceback
        traceback.print_exc()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test VQASynth Gradio Interface")
    parser.add_argument("--launch", action="store_true",
                       help="Launch the Gradio interface")
    parser.add_argument("--test-only", action="store_true", 
                       help="Run test without launching interface")
    
    args = parser.parse_args()
    
    print("🎮 VQASynth Gradio Test")
    print("=" * 40)
    
    if args.test_only:
        success = test_gradio_app()
        if success:
            print("\n✅ All tests passed!")
        else:
            print("\n❌ Tests failed!")
            sys.exit(1)
    elif args.launch:
        launch_gradio_interface()
    else:
        # Default: run test first, then ask about launching
        success = test_gradio_app()
        if success:
            print("\n✅ Test completed successfully!")
            response = input("Would you like to launch the Gradio interface? (y/N): ")
            if response.lower() in ['y', 'yes']:
                launch_gradio_interface()
        else:
            print("\n❌ Test failed! Fix the issues before launching.")


if __name__ == "__main__":
    main()
