import json
from pathlib import Path

notebook_path = Path("/workspaces/events-bot-new/kaggle/CrumpleVideo/crumple_video.ipynb")

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# The blocking code to remove
block_start = '    if not payload_path.exists():\n'
block_mid = '        log(f"ERROR: payload.json not found at {payload_path}")\n'
block_end = '        return False\n'

# Find and remove
removed = False
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = cell["source"]
        
        # We need to find the specific block. It's inside main_pipeline.
        # Let's iterate and look for the sequence.
        
        i = 0
        while i < len(source) - 2:
            if (source[i] == block_start and 
                block_mid in source[i+1] and 
                source[i+2] == block_end):
                
                print(f"Removing lines {i}-{i+2}")
                del source[i:i+3]
                removed = True
                break
            i += 1
        
        if removed:
            print("Successfully removed blocking check.")
            break

if removed:
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Notebook saved.")
else:
    print("Could not find the blocking code block to remove.")
