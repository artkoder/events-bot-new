import json
import base64
from pathlib import Path

notebook_path = Path("/workspaces/events-bot-new/kaggle/CrumpleVideo/crumple_video.ipynb")
script_path = Path("/workspaces/events-bot-new/kaggle/CrumpleVideo/blender_xpbd_paper.py")

# Read and encode the script
with open(script_path, "rb") as f:
    script_content = f.read()
    b64_content = base64.b64encode(script_content).decode('utf-8')

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Update the cell
updated = False
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = cell["source"]
        for i, line in enumerate(source):
            if line.strip().startswith("BLENDER_SCRIPT_B64 ="):
                # Preserve indentation
                indent = line[:line.find("BLENDER_SCRIPT_B64")]
                new_line = f'{indent}BLENDER_SCRIPT_B64 = """{b64_content}"""\n'
                source[i] = new_line
                updated = True
                print("Updated BLENDER_SCRIPT_B64")
                break
        if updated:
            break

if updated:
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Notebook saved successfully")
else:
    print("Failed to find BLENDER_SCRIPT_B64 assignment")
