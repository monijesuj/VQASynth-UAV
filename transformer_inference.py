import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import requests
from io import BytesIO

# Configuration
# model_id = "remyxai/SpaceOm"
model_id = "remyxai/SpaceThinker-Qwen2.5VL-3B"
image_path = "5386811797723543727.jpg"  # or local path
prompt = "Where is the tallest shelf located and what is the height in meters?"
system_message = (
  "You are VL-Thinking 🤔, a helpful assistant with excellent reasoning ability. "
  "You should first think about the reasoning process and then provide the answer. "
  "Use <think>...</think> and <answer>...</answer> tags."
)

# Load model and processor
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id, device_map="auto", torch_dtype=torch.bfloat16
)
processor = AutoProcessor.from_pretrained(model_id)

# Load and preprocess image
if image_path.startswith("http"):
    image = Image.open(BytesIO(requests.get(image_path).content)).convert("RGB")
else:
    image = Image.open(image_path).convert("RGB")
if image.width > 512:
    ratio = image.height / image.width
    image = image.resize((512, int(512 * ratio)), Image.Resampling.LANCZOS)

# Format input
chat = [
    {"role": "system", "content": [{"type": "text", "text": system_message}]},
    {"role": "user", "content": [{"type": "image", "image": image},
                                {"type": "text", "text": prompt}]}
]
text_input = processor.apply_chat_template(chat, tokenize=False,
                                                  add_generation_prompt=True)

# Tokenize
inputs = processor(text=[text_input], images=[image],
                                      return_tensors="pt").to("cuda")

# Generate response
generated_ids = model.generate(**inputs, max_new_tokens=1024)
output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("Response:", output)
