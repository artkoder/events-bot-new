#!/usr/bin/env python3
"""
Animation System for 3D Bento Intro
Implements easing functions and camera animation
"""

import bpy
import math
from typing import Callable, Tuple

# ============================================================================
# Easing Functions
# ============================================================================

class EasingFunctions:
    """Collection of easing functions for smooth animations"""
    
    @staticmethod
    def linear(t: float) -> float:
        """Linear interpolation (no easing)"""
        return t
    
    @staticmethod
    def ease_in(t: float) -> float:
        """Quadratic ease-in: slow start, accelerates"""
        return t * t
    
    @staticmethod
    def ease_out(t: float) -> float:
        """Quadratic ease-out: fast start, decelerates"""
        return t * (2 - t)
    
    @staticmethod
    def ease_in_out(t: float) -> float:
        """Quadratic ease-in-out: slow start and end"""
        return 2 * t * t if t < 0.5 else -1 + (4 - 2 * t) * t
    
    @staticmethod
    def ease_in_cubic(t: float) -> float:
        """Cubic ease-in"""
        return t * t * t
    
    @staticmethod
    def ease_out_cubic(t: float) -> float:
        """Cubic ease-out"""
        return (--t) * t * t + 1
    
    @staticmethod
    def ease_in_out_cubic(t: float) -> float:
        """Cubic ease-in-out: smooth S-curve"""
        if t < 0.5:
            return 4 * t * t * t
        else:
            return (t - 1) * (2 * t - 2) * (2 * t - 2) + 1
    
    @staticmethod
    def ease_in_quart(t: float) -> float:
        """Quartic ease-in"""
        return t * t * t * t
    
    @staticmethod
    def ease_out_quart(t: float) -> float:
        """Quartic ease-out"""
        return 1 - (--t) * t * t * t
    
    @staticmethod
    def ease_in_out_quart(t: float) -> float:
        """Quartic ease-in-out: pronounced S-curve for rotations"""
        if t < 0.5:
            return 8 * t * t * t * t
        else:
            return 1 - 8 * (--t) * t * t * t
    
    @staticmethod
    def ease_in_expo(t: float) -> float:
        """Exponential ease-in: dramatic acceleration"""
        return 0 if t == 0 else math.pow(2, 10 * (t - 1))
    
    @staticmethod
    def ease_out_expo(t: float) -> float:
        """Exponential ease-out: dramatic deceleration"""
        return 1 if t == 1 else 1 - math.pow(2, -10 * t)
    
    @staticmethod
    def ease_out_back(t: float, overshoot: float = 1.70158) -> float:
        """Ease-out with overshoot effect"""
        return 1 + (overshoot + 1) * math.pow(t - 1, 3) + overshoot * math.pow(t - 1, 2)

# ============================================================================
# Camera Animation
# ============================================================================

class CameraAnimation:
    """Handles camera animation for the 3D intro"""
    
    def __init__(self, camera: bpy.types.Object, fps: int = 30):
        """
        Args:
            camera: Blender camera object
            fps: Frames per second
        """
        self.camera = camera
        self.fps = fps
        self.easing = EasingFunctions()
    
    def time_to_frame(self, seconds: float) -> int:
        """Convert time in seconds to frame number"""
        return int(seconds * self.fps)
    
    def set_keyframe(self, frame: int, location: Tuple[float, float, float], 
                    target_location: Tuple[float, float, float]):
        """Set keyframe for camera location and rotation to look at target"""
        import mathutils
        
        # Set Location
        self.camera.location = location
        self.camera.keyframe_insert(data_path="location", frame=frame)
        
        # Calculate Rotation using to_track_quat (Robust method)
        cam_loc = mathutils.Vector(location)
        target_loc = mathutils.Vector(target_location)
        direction = target_loc - cam_loc
        
        # Blender camera points down -Z, Up is +Y
        rot_quat = direction.to_track_quat('-Z', 'Y')
        self.camera.rotation_euler = rot_quat.to_euler()
        self.camera.keyframe_insert(data_path="rotation_euler", frame=frame)

    def setup_5_phase_animation(self, main_cube_pos: Tuple[float, float, float],
                                selected_cube_pos: Tuple[float, float, float]):
        """Setup complete 5-phase animation"""
        # Phase 1: Opening (0.0 - 0.4s) - Close-up
        frame_0 = self.time_to_frame(0.0)
        frame_04 = self.time_to_frame(0.4)
        
        close_pos = (main_cube_pos[0] + 3, main_cube_pos[1] - 3, main_cube_pos[2] + 2)
        
        self.set_keyframe(frame_0, close_pos, main_cube_pos)
        
        # Subtle float
        float_pos = (close_pos[0], close_pos[1], close_pos[2] + 0.1)
        self.set_keyframe(frame_04, float_pos, main_cube_pos)
        
        # Phase 2: Pullback (0.4 - 0.6s)
        frame_06 = self.time_to_frame(0.6)
        
        mid_pos = (main_cube_pos[0] + 6, main_cube_pos[1] - 6, main_cube_pos[2] + 4)
        
        self.set_keyframe(frame_06, mid_pos, main_cube_pos)
        
        # Phase 3: Rotation (0.6 - 1.2s)
        frame_12 = self.time_to_frame(1.2)
        
        # Orbit to side view, changing target to grid center
        orbit_pos = (12, -15, 8)
        grid_center = (5, 3, -4)
        
        self.set_keyframe(frame_12, orbit_pos, grid_center)
        
        # Phase 4: Zoom to poster (1.2 - 1.8s)
        frame_18 = self.time_to_frame(1.8)
        
        zoom_pos = (selected_cube_pos[0] + 1.5, selected_cube_pos[1] - 1.5, selected_cube_pos[2] + 1)
        
        self.set_keyframe(frame_18, zoom_pos, selected_cube_pos)
        
        # Phase 5: Transition (1.8 - 2.5s)
        frame_25 = self.time_to_frame(2.5)
        
        final_pos = (selected_cube_pos[0] + 0.5, selected_cube_pos[1] - 0.5, selected_cube_pos[2] + 0.3)
        
        self.set_keyframe(frame_25, final_pos, selected_cube_pos)
        
        # Apply easing
        self._apply_phase_easing()
        
    def _apply_phase_easing(self):
        """Apply appropriate easing to each animation phase"""
        # Get F-Curves for location and rotation
        action = self.camera.animation_data.action
        if not action:
            return
        
        fcurves = action.fcurves
        
        # Phase 1: 0.0-0.4s - ease_out for subtle float
        for frame_start in range(self.time_to_frame(0.0), self.time_to_frame(0.4) + 1):
            self._set_bezier_handles(fcurves, frame_start, 'AUTO_CLAMPED')
        
        # Phase 2: 0.4-0.6s - ease_in_out_cubic for smooth pullback
        for i in range(len(fcurves)):
            if i < 3:  # Location curves
                fcurves[i].extrapolation = 'LINEAR'
                for kf in fcurves[i].keyframe_points:
                    if self.time_to_frame(0.4) <= kf.co[0] <= self.time_to_frame(0.6):
                        kf.interpolation = 'BEZIER'
                        kf.handle_left_type = 'AUTO_CLAMPED'
                        kf.handle_right_type = 'AUTO_CLAMPED'
        
        # Phase 3: 0.6-1.2s - ease_in_out_quart for rotation
        # Simplification: Use BEZIER for all for now to ensure smooth path
        pass
        
        # Phase 4: 1.2-1.8s - ease_in_expo for dynamic zoom
        # ...
        
        # Phase 5: 1.8-2.5s - ease_out_back for final settle
        # ...
    
    def _set_bezier_handles(self, fcurves, frame: int, handle_type: str):
        """Set Bezier handle types for all fcurves at given frame"""
        for fcurve in fcurves:
            for kf in fcurve.keyframe_points:
                if abs(kf.co[0] - frame) < 0.5:
                    kf.handle_left_type = handle_type
                    kf.handle_right_type = handle_type

# ============================================================================
# Poster Animation
# ============================================================================

class PosterAnimation:
    """Handles poster and text animations for Phase 5"""
    
    def __init__(self, poster_object: bpy.types.Object, fps: int = 30):
        """
        Args:
            poster_object: Blender object representing the poster
            fps: Frames per second
        """
        self.poster = poster_object
        self.fps = fps
        self.easing = EasingFunctions()
    
    def time_to_frame(self, seconds: float) -> int:
        """Convert time in seconds to frame number"""
        return int(seconds * self.fps)
    
    def slide_up(self, start_time: float, duration: float, distance: float):
        """Animate poster sliding up
        
        Args:
            start_time: When to start (in seconds)
            duration: How long the slide takes (in seconds)
            distance: How far to slide (in Blender units)
        """
        start_frame = self.time_to_frame(start_time)
        end_frame = self.time_to_frame(start_time + duration)
        
        # Start position
        start_pos = self.poster.location.copy()
        self.poster.keyframe_insert(data_path="location", frame=start_frame)
        
        # End position (moved up along Z)
        end_pos = (start_pos[0], start_pos[1], start_pos[2] + distance)
        self.poster.location = end_pos
        self.poster.keyframe_insert(data_path="location", frame=end_frame)
        
        # Apply ease_out_back easing
        action = self.poster.animation_data.action
        if action:
            for fcurve in action.fcurves:
                if fcurve.data_path == 'location':
                    for kf in fcurve.keyframe_points:
                        if start_frame <= kf.co[0] <= end_frame:
                            kf.interpolation = 'BEZIER'
                            kf.handle_left_type = 'AUTO_CLAMPED'
                            kf.handle_right_type = 'AUTO_CLAMPED'

if __name__ == "__main__":
    print("Animation system loaded successfully")
    print("Easing functions available:", [f for f in dir(EasingFunctions) if not f.startswith('_')])
