"""
Debug Scene - Interactive frame rendering and scene modification for 3D intro.

Provides CLI tools for iterative scene debugging:
- Render specific frames at given timestamps
- Update camera position/rotation
- Update object parameters
- Save/restore scene snapshots
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Optional

# Check if Blender is available
try:
    import bpy
    BLENDER_AVAILABLE = True
except ImportError:
    BLENDER_AVAILABLE = False
    print("Warning: Blender (bpy) not available. Run this script with Blender Python.", file=sys.stderr)

# Scene configuration
DEFAULT_SCENE_PATH = Path("/tmp/bento_intro_v1_animated.blend")
SNAPSHOT_DIR = Path("/tmp/3d_intro_snapshots")
DEFAULT_FPS = 30
DEFAULT_RESOLUTION = (1080, 1920)  # Vertical format for mobile


class SceneDebugger:
    """Manages scene debugging operations."""
    
    def __init__(self, scene_path: Path):
        """
        Initialize scene debugger.
        
        Args:
            scene_path: Path to Blender scene file
        """
        self.scene_path = scene_path
        if not BLENDER_AVAILABLE:
            raise RuntimeError("Blender not available. Run with: blender --background --python debug_scene.py -- <args>")
        
        # Ensure snapshot directory exists
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    
    def load_scene(self) -> None:
        """Load Blender scene from file."""
        if not self.scene_path.exists():
            raise FileNotFoundError(f"Scene file not found: {self.scene_path}")
        
        # Clear existing scene
        bpy.ops.wm.read_factory_settings(use_empty=True)
        
        # Load scene
        bpy.ops.wm.open_mainfile(filepath=str(self.scene_path))
        print(f"Loaded scene: {self.scene_path}")
    
    def save_scene(self, output_path: Optional[Path] = None) -> None:
        """
        Save current scene to file.
        
        Args:
            output_path: Path to save scene. If None, saves to original path.
        """
        save_path = output_path or self.scene_path
        bpy.ops.wm.save_as_mainfile(filepath=str(save_path))
        print(f"Saved scene: {save_path}")
    
    def render_frame_at_time(self, time_seconds: float, output_path: Path) -> None:
        """
        Render a single frame at specified time.
        
        Args:
            time_seconds: Time in seconds (0.0 - 2.5 for intro)
            output_path: Path for output PNG file
        """
        self.load_scene()
        
        # Calculate frame number from time
        scene = bpy.context.scene
        fps = scene.render.fps
        frame = int(time_seconds * fps)
        
        print(f"Rendering frame {frame} at time {time_seconds}s (FPS: {fps})")
        
        # Set current frame
        scene.frame_set(frame)
        
        # Configure render output
        scene.render.filepath = str(output_path)
        scene.render.image_settings.file_format = 'PNG'
        
        # Ensure resolution is set
        scene.render.resolution_x = DEFAULT_RESOLUTION[0]
        scene.render.resolution_y = DEFAULT_RESOLUTION[1]
        scene.render.resolution_percentage = 100
        
        # Render single frame
        bpy.ops.render.render(write_still=True)
        
        print(f"Rendered frame saved to: {output_path}")
        print(f"Resolution: {scene.render.resolution_x}x{scene.render.resolution_y}")
    
    def update_camera_position(self, position: tuple[float, float, float]) -> None:
        """
        Update camera position.
        
        Args:
            position: (x, y, z) coordinates
        """
        self.load_scene()
        
        camera = bpy.data.objects.get("Camera")
        if not camera:
            raise ValueError("Camera object not found in scene")
        
        camera.location = position
        print(f"Updated camera position to: {position}")
        
        self.save_scene()
    
    def update_camera_rotation(self, rotation: tuple[float, float, float]) -> None:
        """
        Update camera rotation (Euler angles in degrees).
        
        Args:
            rotation: (x, y, z) rotation in degrees
        """
        self.load_scene()
        
        camera = bpy.data.objects.get("Camera")
        if not camera:
            raise ValueError("Camera object not found in scene")
        
        # Convert degrees to radians
        import math
        rotation_rad = tuple(math.radians(deg) for deg in rotation)
        camera.rotation_euler = rotation_rad
        print(f"Updated camera rotation to: {rotation} degrees")
        
        self.save_scene()
    
    def update_object_param(
        self,
        object_name: str,
        param_type: str,
        value: Any
    ) -> None:
        """
        Update object parameter.
        
        Args:
            object_name: Name of object in scene
            param_type: Type of parameter (position, rotation, scale)
            value: New value for parameter
        """
        self.load_scene()
        
        obj = bpy.data.objects.get(object_name)
        if not obj:
            raise ValueError(f"Object '{object_name}' not found in scene")
        
        if param_type == "position":
            obj.location = value
        elif param_type == "rotation":
            import math
            rotation_rad = tuple(math.radians(deg) for deg in value)
            obj.rotation_euler = rotation_rad
        elif param_type == "scale":
            obj.scale = value
        else:
            raise ValueError(f"Unknown parameter type: {param_type}")
        
        print(f"Updated {object_name}.{param_type} to: {value}")
        
        self.save_scene()
    
    def save_snapshot(self, snapshot_name: str) -> None:
        """
        Save scene snapshot with metadata.
        
        Args:
            snapshot_name: Name for snapshot
        """
        snapshot_path = SNAPSHOT_DIR / f"{snapshot_name}.blend"
        metadata_path = SNAPSHOT_DIR / f"{snapshot_name}.json"
        
        # Copy scene file
        shutil.copy2(self.scene_path, snapshot_path)
        
        # Save metadata
        from datetime import datetime
        metadata = {
            "name": snapshot_name,
            "timestamp": datetime.now().isoformat(),
            "original_scene": str(self.scene_path),
        }
        
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Snapshot saved: {snapshot_path}")
        print(f"Metadata: {metadata_path}")
    
    def restore_snapshot(self, snapshot_name: str) -> None:
        """
        Restore scene from snapshot.
        
        Args:
            snapshot_name: Name of snapshot to restore
        """
        snapshot_path = SNAPSHOT_DIR / f"{snapshot_name}.blend"
        
        if not snapshot_path.exists():
            raise FileNotFoundError(f"Snapshot not found: {snapshot_path}")
        
        # Copy snapshot to current scene
        shutil.copy2(snapshot_path, self.scene_path)
        
        print(f"Restored snapshot: {snapshot_name}")
        print(f"Scene updated at: {self.scene_path}")
    
    def list_snapshots(self) -> list[dict]:
        """List all available snapshots."""
        snapshots = []
        
        for metadata_file in SNAPSHOT_DIR.glob("*.json"):
            with open(metadata_file) as f:
                metadata = json.load(f)
                snapshots.append(metadata)
        
        return sorted(snapshots, key=lambda x: x["timestamp"], reverse=True)
    
    def get_scene_info(self) -> dict:
        """Get information about current scene."""
        self.load_scene()
        
        scene = bpy.context.scene
        camera = bpy.data.objects.get("Camera")
        
        info = {
            "scene_name": scene.name,
            "frame_start": scene.frame_start,
            "frame_end": scene.frame_end,
            "frame_current": scene.frame_current,
            "fps": scene.render.fps,
            "resolution": (scene.render.resolution_x, scene.render.resolution_y),
            "objects_count": len(bpy.data.objects),
            "camera": {
                "position": list(camera.location) if camera else None,
                "rotation": list(camera.rotation_euler) if camera else None,
            } if camera else None,
        }
        
        return info


def parse_tuple(value: str) -> tuple:
    """Parse comma-separated values as tuple."""
    return tuple(float(x.strip()) for x in value.split(","))


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Debug 3D Intro Scene")
    parser.add_argument(
        "--scene",
        type=Path,
        default=DEFAULT_SCENE_PATH,
        help="Path to scene file"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Render command
    render_parser = subparsers.add_parser("render", help="Render frame at time")
    render_parser.add_argument("--time", type=float, required=True, help="Time in seconds")
    render_parser.add_argument("--output", type=Path, required=True, help="Output PNG path")
    
    # Camera command
    camera_parser = subparsers.add_parser("camera", help="Update camera")
    camera_parser.add_argument("--position", type=str, help="Position as 'x,y,z'")
    camera_parser.add_argument("--rotation", type=str, help="Rotation as 'x,y,z' (degrees)")
    
    # Object command
    object_parser = subparsers.add_parser("object", help="Update object")
    object_parser.add_argument("--name", type=str, required=True, help="Object name")
    object_parser.add_argument("--position", type=str, help="Position as 'x,y,z'")
    object_parser.add_argument("--rotation", type=str, help="Rotation as 'x,y,z' (degrees)")
    object_parser.add_argument("--scale", type=str, help="Scale as 'x,y,z'")
    
    # Snapshot command
    snapshot_parser = subparsers.add_parser("snapshot", help="Manage snapshots")
    snapshot_parser.add_argument("action", choices=["save", "restore", "list"], help="Snapshot action")
    snapshot_parser.add_argument("--name", type=str, help="Snapshot name (for save/restore)")
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Get scene info")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    debugger = SceneDebugger(args.scene)
    
    try:
        if args.command == "render":
            debugger.render_frame_at_time(args.time, args.output)
        
        elif args.command == "camera":
            if args.position:
                debugger.update_camera_position(parse_tuple(args.position))
            if args.rotation:
                debugger.update_camera_rotation(parse_tuple(args.rotation))
        
        elif args.command == "object":
            if args.position:
                debugger.update_object_param(args.name, "position", parse_tuple(args.position))
            if args.rotation:
                debugger.update_object_param(args.name, "rotation", parse_tuple(args.rotation))
            if args.scale:
                debugger.update_object_param(args.name, "scale", parse_tuple(args.scale))
        
        elif args.command == "snapshot":
            if args.action == "save":
                if not args.name:
                    print("Error: --name required for save action")
                    return
                debugger.save_snapshot(args.name)
            elif args.action == "restore":
                if not args.name:
                    print("Error: --name required for restore action")
                    return
                debugger.restore_snapshot(args.name)
            elif args.action == "list":
                snapshots = debugger.list_snapshots()
                if not snapshots:
                    print("No snapshots found")
                else:
                    print("\nAvailable snapshots:")
                    for snap in snapshots:
                        print(f"  - {snap['name']} ({snap['timestamp']})")
        
        elif args.command == "info":
            info = debugger.get_scene_info()
            print(json.dumps(info, indent=2))
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
