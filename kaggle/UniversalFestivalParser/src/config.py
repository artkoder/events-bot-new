"""Configuration module for Universal Festival Parser.

Reads configuration from environment variables and Kaggle input.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ParserConfig:
    """Configuration for the festival parser."""
    
    # Input
    festival_url: str
    run_id: str
    
    # Parser settings
    parser_version: str = "1.0.0"
    debug: bool = False
    
    # Playwright settings
    headless: bool = True
    timeout_ms: int = 30000
    wait_until: str = "networkidle"
    
    # LLM settings
    llm_model: str = "gemma-3-27b"
    max_retries: int = 3
    
    # Output paths
    output_dir: Path = Path("/kaggle/working")
    
    @classmethod
    def from_environment(cls) -> "ParserConfig":
        """Create config from environment variables."""
        festival_url = os.getenv("FESTIVAL_URL", "")
        run_id = os.getenv("RUN_ID", "")
        
        if not festival_url:
            # Try to read from config file
            config_path = Path("/kaggle/input/run-config/config.json")
            if config_path.exists():
                try:
                    config_data = json.loads(config_path.read_text())
                    festival_url = config_data.get("festival_url", "")
                    run_id = config_data.get("run_id", "")
                except Exception as e:
                    logger.error("Failed to read config file: %s", e)
        
        if not festival_url:
            raise ValueError("FESTIVAL_URL environment variable is required")
        
        if not run_id:
            from datetime import datetime, timezone
            import hashlib
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            url_hash = hashlib.sha256(festival_url.encode()).hexdigest()[:8]
            run_id = f"{timestamp}_{url_hash}"
        
        return cls(
            festival_url=festival_url,
            run_id=run_id,
            parser_version=os.getenv("PARSER_VERSION", "1.0.0"),
            debug=os.getenv("DEBUG", "").lower() in ("1", "true", "yes"),
            headless=os.getenv("HEADLESS", "true").lower() != "false",
            timeout_ms=int(os.getenv("TIMEOUT_MS", "30000")),
            llm_model=os.getenv("LLM_MODEL", "gemma-3-27b"),
        )
    
    def get_output_path(self, filename: str) -> Path:
        """Get full output path for a file."""
        return self.output_dir / filename
