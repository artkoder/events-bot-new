import json
from pathlib import Path

notebook_path = Path("/workspaces/events-bot-new/kaggle/CrumpleVideo/crumple_video.ipynb")

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# The code to inject
injection = [
    "\n",
    "    # Fallback for missing payload.json\n",
    "    if not payload_path.exists():\n",
    "        log(f\"⚠️ payload.json not found at {payload_path}. Generatng fallback.\")\n",
    "        payload_path = WORKING_DIR / \"payload.json\"\n",
    "        fallback_scenes = []\n",
    "        # Find images in source folder\n",
    "        for ext in [\"*.jpg\", \"*.png\"]:\n",
    "            for img in SOURCE_FOLDER.glob(ext):\n",
    "                fallback_scenes.append({\"image\": img.name, \"text\": f\"Fallback {img.stem}\"})\n",
    "        # Sort and take top 4\n",
    "        fallback_scenes.sort(key=lambda x: x[\"image\"])\n",
    "        fallback_data = {\"scenes\": fallback_scenes[:4]}\n",
    "        with open(payload_path, \"w\") as f:\n",
    "            json.dump(fallback_data, f)\n",
    "        log(f\"✅ Generated fallback payload at {payload_path} with {len(fallback_scenes)} scenes\")\n"
]

# Find where to inject (at the start of main_pipeline or just before payload loading)
# Look for: payload = json.loads(payload_path.read_text())
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = cell["source"]
        insert_idx = -1
        for i, line in enumerate(source):
            if "payload = json.loads(payload_path.read_text())" in line:
                insert_idx = i
                break
        
        if insert_idx != -1:
            # Inject before the load
            source[insert_idx:insert_idx] = injection
            print("Injected fallback payload logic.")
            break

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("Notebook saved.")
