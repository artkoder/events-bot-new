from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import os

# Path to the image
img_path = "/home/codespace/.gemini/antigravity/brain/5787ba91-955f-49ed-be99-0e6994f8f7a0/layout_check_v17.png"

def caption_image():
    if not os.path.exists(img_path):
        print(f"Error: Image not found at {img_path}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading model...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

    print(f"Processing image: {img_path}")
    image = Image.open(img_path).convert("RGB")
    inputs = processor(image, return_tensors="pt").to(device)

    print("Generating caption...")
    out = model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(out[0], skip_special_tokens=True)

    print("\n--------------------------------------------------")
    print("CAPTION:", caption)
    print("--------------------------------------------------\n")

if __name__ == "__main__":
    caption_image()
