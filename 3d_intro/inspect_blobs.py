from PIL import Image
import numpy as np
import cv2

path = '/home/codespace/.gemini/antigravity/brain/d43d08f1-b44b-444b-8452-4f1dbed5e7b1/debug_ref_mask.png'
img = Image.open(path).convert('L')
arr = np.array(img) # This is already a binary mask (0 or 255)

# Find connected components
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(arr, connectivity=8)

print(f"Components found: {num_labels}")
for i in range(1, num_labels): # Skip background 0
    x, y, w, h, area = stats[i]
    print(f"Blob {i}: x={x}, y={y}, w={w}, h={h}, area={area}")
