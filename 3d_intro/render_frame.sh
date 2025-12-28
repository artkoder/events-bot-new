#!/bin/bash
# Helper script to render a specific frame from the 3D intro scene

# Usage: ./render_frame.sh <time_in_seconds> [output_path]
# Example: ./render_frame.sh 1.5 /tmp/frame_1_5s.png

SCENE_PATH="${SCENE_PATH:-/tmp/bento_intro_v1_animated.blend}"
TIME="$1"
OUTPUT="${2:-/tmp/frame_${TIME}s.png}"

if [ -z "$TIME" ]; then
    echo "Usage: $0 <time_in_seconds> [output_path]"
    echo "Example: $0 1.5 /tmp/frame_1_5s.png"
    exit 1
fi

echo "Rendering frame at time ${TIME}s from scene: ${SCENE_PATH}"
echo "Output: ${OUTPUT}"

blender --background --python /workspaces/events-bot-new/3d_intro/debug_scene.py -- \
    --scene "$SCENE_PATH" \
    render \
    --time "$TIME" \
    --output "$OUTPUT"

if [ -f "$OUTPUT" ]; then
    echo "✓ Render complete: $OUTPUT"
    # Try to display with imgcat if available
    if command -v imgcat &> /dev/null; then
        imgcat "$OUTPUT"
    fi
else
    echo "✗ Render failed"
    exit 1
fi
