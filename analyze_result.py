import cv2
import sys
import hashlib
from pathlib import Path
import numpy as np

def analyze_video(video_path: str, threshold: float = 0.999):
    print(f"Analyzing {video_path}...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        sys.exit(1)

    frame_count = 0
    prev_frame = None
    static_frames = 0
    identical_sequences = []
    current_sequence = 0
    
    hashes = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_hash = hashlib.md5(frame.tobytes()).hexdigest()
        hashes.append(frame_hash)
        
        if prev_frame is not None:
            # Check for exact identity first (fast)
            if frame_hash == hashes[-2]:
                current_sequence += 1
            else:
                # Check for visual similarity if not identical hash (e.g. compression artifacts)
                # But for static frame bug, they were likely identical.
                # Let's stick to hash for "truly static".
                if current_sequence > 0:
                    identical_sequences.append((frame_count - current_sequence - 1, current_sequence))
                current_sequence = 0
        
        prev_frame = frame
        frame_count += 1

    if current_sequence > 0:
        identical_sequences.append((frame_count - current_sequence - 1, current_sequence))

    cap.release()
    print(f"Total frames: {frame_count}")
    
    print("\nIdentical Frame Sequences (Start Frame, Length):")
    if not identical_sequences:
        print("None detected.")
    else:
        for start, length in identical_sequences:
            print(f"  Start: {start}, Length: {length} (Frames {start}-{start+length})")
            
    # Check specifically for the initial static block
    if identical_sequences and identical_sequences[0][0] == 0:
        initial_static = identical_sequences[0][1]
        print(f"\nInitial static block length: {initial_static} frames")
        if initial_static > 5:
            print("FAIL: Initial static block is too long (expected <= 5 for hold_flat=3)")
        else:
            print("PASS: Initial static block len is acceptable.")
    else:
         print("PASS: No initial static block detected.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_result.py <video_path>")
        sys.exit(1)
    analyze_video(sys.argv[1])
