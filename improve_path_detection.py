import json
from pathlib import Path

notebook_path = Path("/workspaces/events-bot-new/kaggle/CrumpleVideo/crumple_video.ipynb")

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# The code to inject
injection = [
    "\n",
    "# Robust payload detection\n",
    "print(\"DEBUG: Searching for payload.json in\", KAGGLE_INPUT_ROOT)\n",
    "payloads = list(Path(KAGGLE_INPUT_ROOT).rglob(\"payload.json\"))\n",
    "if payloads:\n",
    "    SOURCE_FOLDER = payloads[0].parent\n",
    "    log(f\"✅ Found payload.json at {payloads[0]}, setting SOURCE_FOLDER to {SOURCE_FOLDER}\")\n",
    "else:\n",
    "    log(\"⚠️ payload.json not found via rglob, listing input dir:\")\n",
    "    for root, dirs, files in os.walk(KAGGLE_INPUT_ROOT):\n",
    "        for f in files:\n",
    "            print(os.path.join(root, f))\n"
]

# Find where to inject (after SOURCE_FOLDER definition)
# We look for the cell containing "SOURCE_FOLDER ="
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = cell["source"]
        # Find the line "    log(f\"⚠️ Fallback: {SOURCE_FOLDER}\")\n" which is the end of the existing block
        insert_idx = -1
        for i, line in enumerate(source):
            if "Fallback: {SOURCE_FOLDER}" in line:
                insert_idx = i + 1
                break
        
        if insert_idx != -1:
            # Inject payload detection
            source[insert_idx:insert_idx] = injection
            print("Injected path detection logic.")
            break

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("Notebook saved.")
