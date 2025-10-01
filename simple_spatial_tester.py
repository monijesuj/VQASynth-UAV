#!/usr/bin/env python3
"""
Simple Spatial Reasoning Tester
Quick test interface for spatial VLM capabilities
"""
import gradio as gr
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image
import os

class SimpleSpatialTester:
    def __init__(self):
        self.model = None
        self.processor = None
        
    def load_model(self):
        """Load local spatial reasoning models (SpaceOm or SpaceThinker only)"""
        # Only try the local spatial reasoning models
        local_models = [
            "./SpaceOm",  # SpaceOm - best overall spatial reasoning
            "./SpaceThinker-Qwen2.5VL-3B"  # SpaceThinker - most accurate distances
        ]
        
        for model_path in local_models:
            try:
                print(f"🔄 Loading {model_path}...")
                
                # Load processor first
                self.processor = AutoProcessor.from_pretrained(
                    model_path, 
                    trust_remote_code=True
                )
                
                # Load model with correct Qwen2.5-VL class
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    device_map="auto" if torch.cuda.is_available() else "cpu",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                
                print(f"✅ Successfully loaded {model_path}!")
                print(f"📊 Model device: {next(self.model.parameters()).device}")
                return True
                
            except Exception as e:
                print(f"⚠️  Failed to load {model_path}: {str(e)}")
                continue
        
        print("❌ Could not load SpaceOm or SpaceThinker models")
        print("💡 Make sure the models are in ./SpaceOm and ./SpaceThinker-Qwen2.5VL-3B directories")
        return False
    
    def ask_spatial_question(self, image, question):
        """Ask spatial reasoning questions about an image"""
        if not self.model or not self.processor:
            if not self.load_model():
                return "❌ Failed to load model. Please check your setup."
        
        if image is None:
            return "Please upload an image first."
        
        if not question.strip():
            return "Please enter a spatial question."
        
        try:
            # Format input following the working transformer_inference.py pattern
            chat = [
                {"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question}
                ]}
            ]
            
            # Apply chat template
            text_input = self.processor.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True
            )
            
            # Tokenize with image
            inputs = self.processor(
                text=[text_input], 
                images=[image],
                return_tensors="pt"
            ).to(self.model.device)
            
            # Generate response
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7
                )
            
            # Decode full response and extract the answer part
            response_text = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )[0]
            
            # Extract just the assistant's response (after the input prompt)
            if "assistant\n" in response_text:
                response_text = response_text.split("assistant\n")[-1]
            
            return response_text
            
        except Exception as e:
            return f"❌ Error during inference: {str(e)}"

# Initialize the spatial tester
spatial_tester = SimpleSpatialTester()

def create_demo():
    """Create Gradio interface"""
    
    # Sample spatial questions
    sample_questions = [
        "How tall is the tallest object in this image?",
        "What is the distance between the two main objects?", 
        "Which object is closer to the camera?",
        "How high is the ceiling in this room?",
        "What is the width of the table/desk?",
        "How far apart are the people in this image?",
        "Which object is larger - the chair or the table?",
        "What is the approximate height of the person?",
        "How much space is between the wall and the furniture?",
        "What is the relative position of objects (left/right/above/below)?"
    ]
    
    with gr.Blocks(title="Simple Spatial Reasoning Tester") as demo:
        gr.Markdown("# 🎯 Simple Spatial Reasoning Tester")
        gr.Markdown("Upload an image and ask spatial questions about heights, distances, and relationships.")
        
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="Upload Image")
                question_input = gr.Textbox(
                    placeholder="Ask a spatial question about the image...",
                    label="Spatial Question",
                    lines=2
                )
                
                ask_btn = gr.Button("🤔 Ask Question", variant="primary", size="lg")
                
                gr.Markdown("### 💡 Sample Questions:")
                for i, q in enumerate(sample_questions[:5]):
                    btn = gr.Button(q, size="sm")
                    btn.click(lambda q=q: q, outputs=question_input)
                    
            with gr.Column(scale=1):
                output = gr.Textbox(
                    label="Spatial Analysis",
                    lines=10,
                    placeholder="Upload an image and ask a question to get spatial reasoning..."
                )
        
        ask_btn.click(
            fn=spatial_tester.ask_spatial_question,
            inputs=[image_input, question_input],
            outputs=output
        )
        
        # Examples section
        with gr.Row():
            gr.Examples(
                examples=[
                    [None, "How tall is the main object in meters?"],
                    [None, "What is the distance between the two objects?"],
                    [None, "Which object is closer to the camera?"],
                    [None, "How high is the ceiling?"],
                    [None, "What is the width of the room?"]
                ],
                inputs=[image_input, question_input],
                label="Example Questions"
            )
    
    return demo

if __name__ == "__main__":
    print("🚀 Starting Simple Spatial Reasoning Tester...")
    
    # Load model on startup
    print("Loading spatial reasoning model...")
    spatial_tester.load_model()
    
    # Create and launch demo
    demo = create_demo()
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7861,
        debug=True
    )