from PIL import Image, ImageDraw, ImageFont
import os

def create_composite():
    paths = [
        ('/home/codespace/.gemini/antigravity/brain/5787ba91-955f-49ed-be99-0e6994f8f7a0/debug_frame_0.png', "Frame 0 (Debug)"),
        ('/home/codespace/.gemini/antigravity/brain/5787ba91-955f-49ed-be99-0e6994f8f7a0/final_scene_preview.png', "Frame 40 (Final)"),
        ('/home/codespace/.gemini/antigravity/brain/5787ba91-955f-49ed-be99-0e6994f8f7a0/static_test.png', "Static Test")
    ]
    
    images = []
    for p, label in paths:
        if os.path.exists(p):
            try:
                img = Image.open(p).convert('RGB')
                # Resize to small thumbnail height
                target_h = 400
                aspect = img.width / img.height
                target_w = int(target_h * aspect)
                img = img.resize((target_w, target_h))
                
                # Draw label
                draw = ImageDraw.Draw(img)
                draw.rectangle([(0,0), (target_w, 30)], fill='black')
                draw.text((10, 5), label, fill='white')
                
                images.append(img)
            except Exception as e:
                print(f"Error loading {p}: {e}")
    
    if not images:
        print("No images found!")
        return

    # Combine side by side
    total_w = sum(i.width for i in images)
    max_h = max(i.height for i in images)
    
    composite = Image.new('RGB', (total_w, max_h), (50, 50, 50))
    x = 0
    for img in images:
        composite.paste(img, (x, 0))
        x += img.width
        
    out_path = '/home/codespace/.gemini/antigravity/brain/5787ba91-955f-49ed-be99-0e6994f8f7a0/debug_composite.png'
    composite.save(out_path)
    print(f"Composite saved to {out_path}")

if __name__ == "__main__":
    create_composite()
