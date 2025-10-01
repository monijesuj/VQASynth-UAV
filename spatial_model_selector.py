#!/usr/bin/env python3
"""
Spatial Reasoning Model Selector
Interactive interface to choose between SpaceOm and SpaceThinker models
"""
import gradio as gr
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image
import os

class SpatialModelSelector:
    def __init__(self):
        self.models = {}
        self.processors = {}
        self.current_model = None
        self.available_models = []
        
    def load_available_models(self):
        """Load all available spatial reasoning models"""
        model_configs = {
            "SpaceOm": {
                "path": "./SpaceOm",
                "description": "🌟 SpaceOm - Best overall spatial reasoning capabilities"
            },
            "SpaceThinker": {
                "path": "./SpaceThinker-Qwen2.5VL-3B", 
                "description": "🎯 SpaceThinker - Most accurate distance measurements with thinking process"
            }
        }
        
        print("🔍 Checking available models...")
        for name, config in model_configs.items():
            if os.path.exists(config["path"]):
                try:
                    print(f"Loading {name}...")
                    processor = AutoProcessor.from_pretrained(config["path"], trust_remote_code=True)
                    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        config["path"],
                        torch_dtype=torch.bfloat16,
                        device_map="auto" if torch.cuda.is_available() else "cpu",
                        trust_remote_code=True
                    )
                    
                    self.models[name] = model
                    self.processors[name] = processor
                    self.available_models.append((name, config["description"]))
                    print(f"✅ {name} loaded successfully!")
                    
                except Exception as e:
                    print(f"❌ Failed to load {name}: {e}")
            else:
                print(f"⚠️  {name} not found at {config['path']}")
        
        if self.available_models:
            self.current_model = self.available_models[0][0]  # Default to first available
            print(f"🚀 Ready with {len(self.available_models)} model(s)")
        else:
            print("❌ No spatial reasoning models available!")
    
    def switch_model(self, model_name):
        """Switch to a different model"""
        if model_name in self.models:
            self.current_model = model_name
            return f"✅ Switched to {model_name}"
        else:
            return f"❌ Model {model_name} not available"
    
    def ask_spatial_question(self, images, question, model_choice):
        """Ask spatial reasoning questions with selected model - supports multiple images"""
        if not self.models:
            return "❌ No models loaded. Please check your setup."
        
        # Handle both single image and multiple images
        if isinstance(images, list):
            image_list = [img for img in images if img is not None]
        else:
            image_list = [images] if images is not None else []
        
        if not image_list:
            return "📸 Please upload at least one image first."
        
        if not question.strip():
            return "❓ Please enter a spatial question."
        
        # Switch model if different from current
        if model_choice != self.current_model:
            self.switch_model(model_choice)
        
        try:
            model = self.models[self.current_model]
            processor = self.processors[self.current_model]
            
            # Format input for inference - support multiple images
            content = []
            for i, img in enumerate(image_list):
                content.append({"type": "image", "image": img})
                if len(image_list) > 1:
                    content.append({"type": "text", "text": f"Image {i+1}: "})
            
            content.append({"type": "text", "text": question})
            
            chat = [{"role": "user", "content": content}]
            
            # Add system message for SpaceThinker (thinking model)
            if "thinker" in self.current_model.lower():
                system_msg = (
                    "You are VL-Thinking 🤔, a helpful assistant with excellent reasoning ability. "
                    "You should first think about the reasoning process and then provide the answer. "
                    "Use <think>...</think> and <answer>...</answer> tags."
                )
                chat.insert(0, {"role": "system", "content": [{"type": "text", "text": system_msg}]})
            
            # Apply chat template
            text_input = processor.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True
            )
            
            # Tokenize with images (supports multiple images)
            inputs = processor(
                text=[text_input], 
                images=image_list,
                return_tensors="pt"
            ).to(model.device)
            
            # Generate response
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.1
                )
            
            # Decode full response
            response_text = processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )[0]
            
            # Extract just the assistant's response
            if "assistant\n" in response_text:
                response_text = response_text.split("assistant\n")[-1]
            
            # Add model info to response
            model_info = f"🤖 **{self.current_model}** Response:\n\n"
            return model_info + response_text
            
        except Exception as e:
            return f"❌ Error with {self.current_model}: {str(e)}"

# Initialize the model selector
selector = SpatialModelSelector()

def create_interface():
    """Create Gradio interface with model selection"""
    
    # Sample spatial questions
    sample_questions = [
        "How tall is the main object in meters?",
        "What is the distance between the two largest objects?", 
        "Which object is closer to the camera?",
        "How high is the ceiling in this room?",
        "What is the width of the table or desk?",
        "How far apart are the people in this image?",
        "Which object is larger - compare the sizes?",
        "What is the approximate height of the person?",
        "How much space is between objects?",
        "Describe the relative positions of objects."
    ]
    
    with gr.Blocks(title="Spatial Reasoning Model Selector", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎯 Spatial Reasoning Model Selector")
        gr.Markdown("Choose between different spatial reasoning models and ask questions about images.")
        
        # Model selection
        if selector.available_models:
            model_choices = [name for name, _ in selector.available_models]
            model_descriptions = "\n".join([f"• **{name}**: {desc}" for name, desc in selector.available_models])
        else:
            model_choices = ["No models available"]
            model_descriptions = "❌ No spatial reasoning models found!"
        
        with gr.Row():
            gr.Markdown(f"### Available Models:\n{model_descriptions}")
        
        with gr.Row():
            with gr.Column(scale=1):
                model_selector = gr.Dropdown(
                    choices=model_choices,
                    value=model_choices[0] if model_choices[0] != "No models available" else None,
                    label="🤖 Select Model",
                    info="Choose your spatial reasoning model"
                )
                
                # Support multiple images for multi-view spatial reasoning
                with gr.Tabs():
                    with gr.TabItem("Single Image"):
                        single_image = gr.Image(type="pil", label="📸 Upload Image")
                    
                    with gr.TabItem("Multi-View Images"):
                        multi_images = gr.File(
                            file_count="multiple",
                            file_types=["image"],
                            label="📸 Upload Multiple Images (for multi-view spatial analysis)"
                        )
                
                question_input = gr.Textbox(
                    placeholder="Ask a spatial question about the image(s)...",
                    label="❓ Spatial Question",
                    lines=3,
                    info="Examples: 'Compare objects across views', 'What's the distance between objects in different images?'"
                )
                
                ask_btn = gr.Button("🔍 Analyze", variant="primary", size="lg")
                
            with gr.Column(scale=1):
                output = gr.Textbox(
                    label="🧠 Spatial Analysis",
                    lines=15,
                    placeholder="Select a model, upload an image, and ask a spatial question to get analysis..."
                )
        
        # Quick question buttons
        with gr.Row():
            gr.Markdown("### 💡 Quick Questions:")
        
        with gr.Row():
            for i in range(0, len(sample_questions), 2):
                with gr.Column():
                    for j in range(2):
                        if i + j < len(sample_questions):
                            btn = gr.Button(sample_questions[i + j], size="sm")
                            btn.click(
                                lambda q=sample_questions[i + j]: q, 
                                outputs=question_input
                            )
        
        # Helper function to handle image input
        def process_images_and_question(single_img, multi_imgs, question, model):
            # Combine single and multiple images
            images = []
            if single_img is not None:
                images.append(single_img)
            if multi_imgs is not None:
                for img_file in multi_imgs:
                    try:
                        img = Image.open(img_file.name)
                        images.append(img)
                    except Exception as e:
                        continue
            
            return selector.ask_spatial_question(images, question, model)
        
        # Connect the main function
        ask_btn.click(
            fn=process_images_and_question,
            inputs=[single_image, multi_images, question_input, model_selector],
            outputs=output
        )
        
        # Examples section
        with gr.Row():
            gr.Examples(
                examples=[
                    [None, None, "How tall is the main object in meters?", model_choices[0] if model_choices[0] != "No models available" else None],
                    [None, None, "What is the distance between objects?", model_choices[0] if model_choices[0] != "No models available" else None],
                    [None, None, "Compare objects across different viewpoints", model_choices[0] if model_choices[0] != "No models available" else None]
                ],
                inputs=[single_image, multi_images, question_input, model_selector],
                label="Example Queries"
            )
    
    return demo

if __name__ == "__main__":
    print("🚀 Starting Spatial Reasoning Model Selector...")
    
    # Load available models
    selector.load_available_models()
    
    # Create and launch interface
    demo = create_interface()
    demo.launch(
        share=True,
        server_name="0.0.0.0", 
        server_port=7862,
        debug=True
    )