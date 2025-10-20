#!/usr/bin/env python3
"""
Online Spatial Reasoning Inference System
Real-time VQA with SpaceOm and SpaceThinker from camera streams
"""
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image
import os
from typing import Optional, List, Dict, Any
from collections import deque
import time

class SpatialModelLoader:
    """Load and manage spatial reasoning models"""
    
    def __init__(self, model_dir: str = "."):
        """
        Initialize model loader
        
        Args:
            model_dir: Base directory containing model folders
        """
        self.model_dir = model_dir
        self.models = {}
        self.processors = {}
        self.current_model_name = None
        
        # Model configurations
        self.model_configs = {
            "SpaceOm": {
                "path": os.path.join(model_dir, "SpaceOm"),
                "description": "Best overall spatial reasoning capabilities",
                "strengths": ["general spatial understanding", "object relationships"]
            },
            "SpaceThinker": {
                "path": os.path.join(model_dir, "SpaceThinker-Qwen2.5VL-3B"),
                "description": "Most accurate distance measurements with reasoning",
                "strengths": ["distance estimation", "quantitative reasoning", "step-by-step thinking"]
            }
        }
        
    def list_available_models(self) -> List[str]:
        """List all available models in the model directory"""
        available = []
        for name, config in self.model_configs.items():
            if os.path.exists(config["path"]):
                available.append(name)
        return available
    
    def load_model(self, model_name: str, device: str = "auto") -> bool:
        """
        Load a specific model into memory
        
        Args:
            model_name: Name of model to load (SpaceOm or SpaceThinker)
            device: Device placement ('auto', 'cuda', 'cpu')
            
        Returns:
            True if successful, False otherwise
        """
        if model_name in self.models:
            print(f"✅ {model_name} already loaded")
            self.current_model_name = model_name
            return True
        
        if model_name not in self.model_configs:
            print(f"❌ Unknown model: {model_name}")
            return False
        
        config = self.model_configs[model_name]
        model_path = config["path"]
        
        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            return False
        
        try:
            print(f"🔄 Loading {model_name} from {model_path}...")
            
            # Load processor
            processor = AutoProcessor.from_pretrained(
                model_path, 
                trust_remote_code=True
            )
            
            # Load model
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map=device if device != "auto" else ("auto" if torch.cuda.is_available() else "cpu"),
                trust_remote_code=True
            )
            
            self.models[model_name] = model
            self.processors[model_name] = processor
            self.current_model_name = model_name
            
            device_info = next(model.parameters()).device
            print(f"✅ {model_name} loaded successfully on {device_info}")
            
            if torch.cuda.is_available():
                memory_gb = torch.cuda.memory_allocated() / 1e9
                print(f"💾 GPU Memory: {memory_gb:.2f} GB")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load {model_name}: {e}")
            return False
    
    def switch_model(self, model_name: str) -> bool:
        """
        Switch to a different model (loads if not already loaded)
        
        Args:
            model_name: Name of model to switch to
            
        Returns:
            True if successful, False otherwise
        """
        if model_name not in self.models:
            return self.load_model(model_name)
        
        self.current_model_name = model_name
        print(f"🔄 Switched to {model_name}")
        return True
    
    def get_current_model(self) -> tuple:
        """
        Get current model and processor
        
        Returns:
            (model, processor) tuple or (None, None) if no model loaded
        """
        if self.current_model_name is None or self.current_model_name not in self.models:
            return None, None
        
        return (
            self.models[self.current_model_name],
            self.processors[self.current_model_name]
        )
    
    def unload_model(self, model_name: str):
        """Unload a model from memory"""
        if model_name in self.models:
            del self.models[model_name]
            del self.processors[model_name]
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print(f"🗑️  {model_name} unloaded")
            
            if self.current_model_name == model_name:
                self.current_model_name = None
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a specific model"""
        if model_name not in self.model_configs:
            return {}
        
        config = self.model_configs[model_name]
        info = {
            "name": model_name,
            "description": config["description"],
            "strengths": config["strengths"],
            "loaded": model_name in self.models,
            "current": model_name == self.current_model_name
        }
        
        if info["loaded"]:
            model = self.models[model_name]
            info["device"] = str(next(model.parameters()).device)
            
            if torch.cuda.is_available():
                info["memory_gb"] = torch.cuda.memory_allocated() / 1e9
        
        return info


class OnlineInferenceEngine:
    """Online inference engine for spatial VQA with streaming"""
    
    def __init__(self, model_loader: SpatialModelLoader, max_history: int = 5):
        """
        Initialize inference engine
        
        Args:
            model_loader: SpatialModelLoader instance
            max_history: Maximum number of frames to keep in history
        """
        self.model_loader = model_loader
        self.max_history = max_history
        
        # Frame history for context
        self.frame_history = deque(maxlen=max_history)
        self.latest_frame = None
        
        # Inference statistics
        self.inference_count = 0
        self.total_inference_time = 0.0
    
    def update_frame(self, frame: Image.Image):
        """
        Update with new frame from camera stream
        
        Args:
            frame: PIL Image from camera
        """
        self.latest_frame = frame
        self.frame_history.append({
            'frame': frame,
            'timestamp': time.time()
        })
    
    def ask_question(
        self,
        question: str,
        use_latest: bool = True,
        frame: Optional[Image.Image] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Ask a spatial reasoning question about the current or provided frame
        
        Args:
            question: Question to ask
            use_latest: Use latest frame from stream (if True and frame is None)
            frame: Specific frame to use (overrides use_latest)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Dictionary with answer, timing, and metadata
        """
        # Get model and processor
        model, processor = self.model_loader.get_current_model()
        
        if model is None or processor is None:
            return {
                'answer': "❌ No model loaded",
                'success': False,
                'error': "No model loaded"
            }
        
        # Determine which frame to use
        if frame is None:
            if use_latest and self.latest_frame is not None:
                frame = self.latest_frame
            else:
                return {
                    'answer': "❌ No frame available",
                    'success': False,
                    'error': "No frame available"
                }
        
        try:
            start_time = time.time()
            
            # Format input
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": frame},
                        {"type": "text", "text": question}
                    ]
                }
            ]
            
            # Apply chat template
            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Process inputs
            inputs = processor(
                text=[text],
                images=[frame],
                return_tensors="pt",
                padding=True
            ).to(model.device)
            
            # Generate response
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0
                )
            
            # Decode response
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            answer = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            inference_time = time.time() - start_time
            
            # Update statistics
            self.inference_count += 1
            self.total_inference_time += inference_time
            
            return {
                'answer': answer,
                'success': True,
                'inference_time': inference_time,
                'model': self.model_loader.current_model_name,
                'timestamp': time.time()
            }
            
        except Exception as e:
            return {
                'answer': f"❌ Inference failed: {str(e)}",
                'success': False,
                'error': str(e)
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get inference statistics"""
        avg_time = (
            self.total_inference_time / self.inference_count
            if self.inference_count > 0
            else 0.0
        )
        
        return {
            'total_inferences': self.inference_count,
            'total_time': self.total_inference_time,
            'average_time': avg_time,
            'frames_in_history': len(self.frame_history)
        }
    
    def clear_history(self):
        """Clear frame history"""
        self.frame_history.clear()
        self.latest_frame = None


def test_online_inference():
    """Test the online inference system"""
    print("🧪 Testing Online Inference System")
    print("=" * 50)
    
    # Initialize model loader
    loader = SpatialModelLoader()
    
    # List available models
    available = loader.list_available_models()
    print(f"📦 Available models: {available}")
    
    if not available:
        print("❌ No models found!")
        return
    
    # Load first available model
    model_name = available[0]
    print(f"\n🔄 Loading {model_name}...")
    
    if not loader.load_model(model_name):
        print("❌ Failed to load model")
        return
    
    # Show model info
    info = loader.get_model_info(model_name)
    print(f"\n📊 Model Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Create inference engine
    engine = OnlineInferenceEngine(loader)
    
    # Test with a dummy frame
    try:
        from PIL import Image
        import numpy as np
        
        # Create test image
        test_image = Image.fromarray(
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        )
        
        engine.update_frame(test_image)
        
        # Ask a test question
        result = engine.ask_question("What can you see in this image?")
        
        print(f"\n🤔 Test Question: What can you see in this image?")
        print(f"✅ Answer: {result['answer'][:200]}...")
        print(f"⏱️  Inference time: {result.get('inference_time', 0):.2f}s")
        
        # Show stats
        stats = engine.get_stats()
        print(f"\n📊 Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        print("\n✅ Online inference system ready!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    test_online_inference()
