
import asyncio
import logging
import json
import time
from pathlib import Path
from video_announce.kaggle_client import KaggleClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("e2e_runner")

async def run_e2e_tests():
    """
    Push E2E tests to Kaggle, run them, and verify results.
    """
    client = KaggleClient()
    
    # Identify local kernel path
    kernel_dir = Path("kaggle/E2ETests")
    if not kernel_dir.exists():
        logger.error(f"Kernel directory not found: {kernel_dir}")
        return

    logger.info(f"Pushing kernel from {kernel_dir}...")
    
    # Push kernel
    # Note: We are not using datasets here, just the code. 
    # Secrets must be already configured in the user's account for this kernel.
    try:
        # push_kernel expects secrets/config to be handled by Kaggle backend 
        # based on kernel-metadata.json.
        client.push_kernel(kernel_path=kernel_dir)
    except Exception as e:
        logger.error(f"Failed to push kernel: {e}")
        return

    # Get kernel ref from metadata
    meta_path = kernel_dir / "kernel-metadata.json"
    meta = json.loads(meta_path.read_text())
    # Standardize ref: username/slug
    # If the user env has different username, it might mismatch what's in file if not careful.
    # But usually push_kernel uses what is in metadata or API defaults.
    kernel_ref = f"{meta['id']}"  # e.g. zigomaro/events-bot-e2e
    
    logger.info(f"Kernel pushed: {kernel_ref}. Waiting for execution...")
    
    # Poll for status
    start_time = time.time()
    max_wait = 1200 # 20 minutes
    has_started = False
    
    while time.time() - start_time < max_wait:
        status_info = client.get_kernel_status(kernel_ref)
        status = status_info.get("status", "UNKNOWN").upper()
        
        logger.info(f"Status: {status}")
        
        if status in ["QUEUED", "RUNNING"]:
            has_started = True
            
        if status == "COMPLETE":
             if has_started or (time.time() - start_time > 60):
                 logger.info("Kernel completed successfully!")
                 break
             else:
                 logger.info("Status is COMPLETE but we just started. Waiting for new run to pick up...")
                 
        elif status in ["ERROR", "FAILED", "CANCELLED"]:
             # If it failed immediately, it might be the new run failed or old run.
             if has_started or (time.time() - start_time > 60):
                 logger.error(f"Kernel failed: {status_info.get('failureMessage')}")
                 break
             else:
                  logger.info(f"Status is {status} (possibly old run). Waiting...")

        await asyncio.sleep(15)
        
    # Download output
    output_dir = Path("e2e_results")
    output_dir.mkdir(exist_ok=True)
    
    logger.info("Downloading outputs...")
    files = client.download_kernel_output(kernel_ref, path=output_dir, force=True)
    
    report_file = output_dir / "test_report.json"
    if report_file.exists():
        logger.info("Test report found!")
        try:
            data = json.loads(report_file.read_text())
            # Basic summary
            # behave json output is usually a list of features
            passed = 0
            failed = 0
            for feature in data:
                for scenario in feature.get('elements', []):
                    # Check steps
                    scenario_passed = True
                    for step in scenario.get('steps', []):
                        if step.get('result', {}).get('status') != 'passed':
                            scenario_passed = False
                            break
                    if scenario_passed:
                        passed += 1
                    else:
                        failed += 1
            
            logger.info(f"TEST RESULTS: Passed={passed}, Failed={failed}")
            
            if failed > 0:
                logger.error("Some tests failed!")
            else:
                logger.info("All tests passed.")
                
        except Exception as e:
            logger.error(f"Failed to parse report: {e}")
            print(report_file.read_text())
    else:
        logger.warning("No test_report.json found in output.")
        logger.info(f"Files downloaded: {files}")

if __name__ == "__main__":
    asyncio.run(run_e2e_tests())
