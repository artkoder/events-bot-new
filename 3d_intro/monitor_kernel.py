#!/usr/bin/env python
"""Monitor Blender test kernel on Kaggle"""
import asyncio
import sys
from pathlib import Path
from video_announce.kaggle_client import KaggleClient

async def poll_kernel(kernel_ref: str, max_polls: int = 20):
    client = KaggleClient()
    
    print(f"Monitoring kernel: {kernel_ref}")
    print(f"Will poll up to {max_polls} times (every 30s)")
    print("=" * 60)
    
    unknown_count = 0
    MAX_UNKNOWN = 10
    
    for i in range(max_polls):
        try:
            status = await asyncio.to_thread(client.get_kernel_status, kernel_ref)
            state = str(status.get("status") or "").lower()
            
            print(f"[{i+1}/{max_polls}] Status: {status.get('status')} (raw: {status})")
            
            if not state or state in {"none", "unknown"}:
                unknown_count += 1
                print(f"  ⚠ Unknown status (count: {unknown_count}/{MAX_UNKNOWN})")
                if unknown_count >= MAX_UNKNOWN:
                    print("  ❌ Too many unknown statuses, giving up")
                    return
            else:
                unknown_count = 0  # Reset on valid status
                
            if state == "complete":
                print("  ✓ Kernel completed!")
                # Download output
                output_dir = Path("/tmp/blender_test_output")
                output_dir.mkdir(exist_ok=True)
                print(f"  Downloading output to {output_dir}...")
                files = await asyncio.to_thread(
                    client.download_kernel_output,
                    kernel_ref,
                    path=output_dir,
                    force=True,
                    quiet=False
                )
                print(f"  ✓ Downloaded {len(files)} files:")
                for f in files:
                    print(f"    - {f}")
                return
                
            if state in {"error", "failed"}:
                failure = status.get("failureMessage") or status.get("failure_message") or "unknown"
                print(f"  ❌ Kernel failed: {failure}")
                return
                
        except Exception as e:
            print(f"  ⚠ Error polling: {e}")
            
        if i < max_polls - 1:
            await asyncio.sleep(30)
    
    print("⏱ Timeout reached")

if __name__ == "__main__":
    kernel_ref = sys.argv[1] if len(sys.argv) > 1 else "artkoder/blender-quick-test"
    asyncio.run(poll_kernel(kernel_ref))
