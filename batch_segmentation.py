#!/usr/bin/env python3
"""
Batch Semantic Segmentation
Process multiple images with the same prompt using the enhanced segmentation system
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple
import argparse
import cv2
import numpy as np
from datetime import datetime

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from enhanced_semantic_segmentation import EnhancedSemanticSegmentation

class BatchSegmentationProcessor:
    """Batch processor for semantic segmentation"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.segmenter = EnhancedSemanticSegmentation(device=device)
        self.results = []
    
    def process_batch(self, input_dir: str, output_dir: str, text_prompt: str, 
                     file_extensions: List[str] = None) -> Dict:
        """Process a batch of images"""
        
        if file_extensions is None:
            file_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all image files
        image_files = []
        for ext in file_extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        print(f"🔍 Found {len(image_files)} images to process")
        print(f"📝 Prompt: '{text_prompt}'")
        print(f"📁 Input: {input_dir}")
        print(f"📁 Output: {output_dir}")
        
        # Process each image
        successful = 0
        failed = 0
        
        for i, image_file in enumerate(image_files):
            print(f"\n📸 Processing {i+1}/{len(image_files)}: {image_file.name}")
            
            try:
                # Run segmentation
                result, info = self.segmenter.segment_image(str(image_file), text_prompt)
                
                if result is not None:
                    # Save result
                    output_file = output_path / f"segmented_{image_file.stem}.jpg"
                    cv2.imwrite(str(output_file), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
                    
                    # Save mask
                    mask_file = output_path / f"mask_{image_file.stem}.png"
                    # Extract mask from result (simplified)
                    mask = self._extract_mask_from_result(result)
                    cv2.imwrite(str(mask_file), mask)
                    
                    # Save info
                    info_file = output_path / f"info_{image_file.stem}.json"
                    with open(info_file, 'w') as f:
                        json.dump(info, f, indent=2)
                    
                    # Record result
                    self.results.append({
                        'input_file': str(image_file),
                        'output_file': str(output_file),
                        'mask_file': str(mask_file),
                        'info_file': str(info_file),
                        'success': True,
                        'info': info
                    })
                    
                    successful += 1
                    print(f"   ✅ Success - Score: {info['score']:.3f}")
                else:
                    self.results.append({
                        'input_file': str(image_file),
                        'success': False,
                        'error': 'Segmentation failed'
                    })
                    failed += 1
                    print(f"   ❌ Failed")
                
            except Exception as e:
                self.results.append({
                    'input_file': str(image_file),
                    'success': False,
                    'error': str(e)
                })
                failed += 1
                print(f"   ❌ Error: {e}")
        
        # Save batch summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'prompt': text_prompt,
            'input_dir': str(input_dir),
            'output_dir': str(output_dir),
            'total_images': len(image_files),
            'successful': successful,
            'failed': failed,
            'results': self.results
        }
        
        summary_file = output_path / "batch_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n📊 Batch Processing Complete!")
        print(f"   ✅ Successful: {successful}")
        print(f"   ❌ Failed: {failed}")
        print(f"   📁 Results saved to: {output_dir}")
        print(f"   📄 Summary: {summary_file}")
        
        return summary
    
    def _extract_mask_from_result(self, result_image: np.ndarray) -> np.ndarray:
        """Extract mask from segmented result image"""
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(result_image, cv2.COLOR_RGB2HSV)
        
        # Define range for green mask
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([80, 255, 255])
        
        # Create mask
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        return mask
    
    def process_with_multiple_prompts(self, input_dir: str, output_dir: str, 
                                    prompts: List[str]) -> Dict:
        """Process images with multiple prompts"""
        
        all_results = {}
        
        for prompt in prompts:
            print(f"\n🎯 Processing with prompt: '{prompt}'")
            
            # Create subdirectory for this prompt
            prompt_output_dir = Path(output_dir) / prompt.replace(" ", "_").replace("/", "_")
            
            # Process batch
            results = self.process_batch(input_dir, str(prompt_output_dir), prompt)
            all_results[prompt] = results
        
        return all_results


def main():
    parser = argparse.ArgumentParser(description="Batch Semantic Segmentation")
    parser.add_argument("--input", "-i", required=True, help="Input directory containing images")
    parser.add_argument("--output", "-o", required=True, help="Output directory for results")
    parser.add_argument("--prompt", "-p", required=True, help="Segmentation prompt")
    parser.add_argument("--device", "-d", default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("--extensions", "-e", nargs="+", 
                       default=['.jpg', '.jpeg', '.png', '.bmp', '.tiff'],
                       help="File extensions to process")
    parser.add_argument("--multi-prompt", "-m", nargs="+", 
                       help="Process with multiple prompts")
    
    args = parser.parse_args()
    
    # Validate input directory
    if not Path(args.input).exists():
        print(f"❌ Input directory does not exist: {args.input}")
        return
    
    # Create processor
    processor = BatchSegmentationProcessor(device=args.device)
    
    if args.multi_prompt:
        # Process with multiple prompts
        results = processor.process_with_multiple_prompts(
            args.input, args.output, args.multi_prompt
        )
    else:
        # Process with single prompt
        results = processor.process_batch(
            args.input, args.output, args.prompt, args.extensions
        )
    
    print("\n🎉 Batch processing complete!")


if __name__ == "__main__":
    main()
