"""LLM interaction logging for Universal Festival Parser.

Logs all LLM requests and responses to JSON for analysis and debugging.
This is critical for iterating on parsing strategy.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class LLMInteraction:
    """Single LLM request/response pair."""
    
    request_id: str
    timestamp: str
    model: str
    phase: str  # "distill" | "reason" | "validate"
    prompt: str
    prompt_tokens: int
    response: str
    response_tokens: int
    duration_ms: float
    success: bool
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)


class LLMLogger:
    """Logger for LLM interactions during parsing.
    
    Usage:
        logger = LLMLogger(run_id="20260104T120000Z_abc123")
        
        with logger.track("reason", model="gemma-3-27b", prompt="...") as tracker:
            response = call_llm(prompt)
            tracker.set_response(response)
        
        # At the end, save all interactions
        logger.save("/kaggle/working/llm_log.json")
    """
    
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.interactions: list[LLMInteraction] = []
        self._request_counter = 0
    
    def _next_request_id(self) -> str:
        self._request_counter += 1
        return f"{self.run_id}_{self._request_counter:03d}"
    
    def track(self, phase: str, model: str, prompt: str, **metadata) -> "LLMTracker":
        """Start tracking an LLM call."""
        request_id = self._next_request_id()
        return LLMTracker(
            logger=self,
            request_id=request_id,
            phase=phase,
            model=model,
            prompt=prompt,
            metadata=metadata,
        )
    
    def log_interaction(self, interaction: LLMInteraction) -> None:
        """Record a completed interaction."""
        self.interactions.append(interaction)
        logger.info(
            "LLM %s: phase=%s model=%s tokens=%d/%d duration=%.0fms success=%s",
            interaction.request_id,
            interaction.phase,
            interaction.model,
            interaction.prompt_tokens,
            interaction.response_tokens,
            interaction.duration_ms,
            interaction.success,
        )
    
    def save(self, path: str | Path) -> None:
        """Save all interactions to JSON."""
        path = Path(path)
        data = {
            "run_id": self.run_id,
            "total_interactions": len(self.interactions),
            "total_prompt_tokens": sum(i.prompt_tokens for i in self.interactions),
            "total_response_tokens": sum(i.response_tokens for i in self.interactions),
            "total_duration_ms": sum(i.duration_ms for i in self.interactions),
            "success_rate": (
                sum(1 for i in self.interactions if i.success) / len(self.interactions)
                if self.interactions else 0.0
            ),
            "interactions": [asdict(i) for i in self.interactions],
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Saved %d LLM interactions to %s", len(self.interactions), path)
    
    def get_summary(self) -> dict:
        """Get summary statistics."""
        return {
            "total_calls": len(self.interactions),
            "by_phase": self._group_by("phase"),
            "by_model": self._group_by("model"),
            "errors": [i.error for i in self.interactions if i.error],
        }
    
    def _group_by(self, key: str) -> dict[str, int]:
        result: dict[str, int] = {}
        for i in self.interactions:
            k = getattr(i, key, "unknown")
            result[k] = result.get(k, 0) + 1
        return result


class LLMTracker:
    """Context manager for tracking a single LLM call."""
    
    def __init__(
        self,
        logger: LLMLogger,
        request_id: str,
        phase: str,
        model: str,
        prompt: str,
        metadata: dict,
    ):
        self._logger = logger
        self._request_id = request_id
        self._phase = phase
        self._model = model
        self._prompt = prompt
        self._metadata = metadata
        self._start_time: float = 0
        self._response: str = ""
        self._error: Optional[str] = None
        self._response_tokens: int = 0
    
    def __enter__(self) -> "LLMTracker":
        self._start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        duration_ms = (time.perf_counter() - self._start_time) * 1000
        
        if exc_val:
            self._error = str(exc_val)
        
        # Estimate token counts (rough approximation: ~4 chars per token)
        prompt_tokens = len(self._prompt) // 4
        response_tokens = len(self._response) // 4 if self._response else 0
        
        interaction = LLMInteraction(
            request_id=self._request_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            model=self._model,
            phase=self._phase,
            prompt=self._prompt,
            prompt_tokens=prompt_tokens,
            response=self._response,
            response_tokens=response_tokens,
            duration_ms=duration_ms,
            success=self._error is None,
            error=self._error,
            metadata=self._metadata,
        )
        
        self._logger.log_interaction(interaction)
    
    def set_response(self, response: str, tokens: int | None = None) -> None:
        """Set the LLM response."""
        self._response = response
        if tokens is not None:
            self._response_tokens = tokens
    
    def set_error(self, error: str) -> None:
        """Set an error message."""
        self._error = error
