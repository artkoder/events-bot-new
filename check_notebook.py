import json
import base64
from pathlib import Path

notebook_path = Path("/workspaces/events-bot-new/kaggle/CrumpleVideo/crumple_video.ipynb")
script_path = Path("/workspaces/events-bot-new/kaggle/CrumpleVideo/blender_xpbd_paper.py")

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Find the cell
found = False
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = cell["source"]
        for i, line in enumerate(source):
            if line.strip().startswith("BLENDER_SCRIPT_B64 ="):
                print(f"Found line at index {i}")
                print(f"Current content (start): {line[:100]}")
                found = True
                
if not found:
    print("Could not find BLENDER_SCRIPT_B64 assignment in notebook")
