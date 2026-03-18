"""
Base Probe Module - Abstract base class for all verification probes.
"""

from abc import ABC, abstractmethod
from typing import Any

from utils.api_client import ModelClient
from utils.types import ProbeResult, ModelConfig, Verdict


class BaseProbe(ABC):
    """Abstract base class for all verification probes."""

    def __init__(self, client: ModelClient, model_config: ModelConfig):
        self.client = client
        self.model_config = model_config

    @abstractmethod
    def run(self) -> ProbeResult:
        """Execute the probe and return results."""
        pass

    def _compute_verdict(self, score: float, confidence: float) -> Verdict:
        """Determine verdict based on score and confidence."""
        if score >= 0.8 and confidence >= 0.5:
            return Verdict.PASS
        elif score >= 0.5:
            return Verdict.WARN
        else:
            return Verdict.FAIL

    def _error_result(self, error_msg: str) -> ProbeResult:
        """Create an error result when probe fails."""
        from utils.types import ProbeType

        return ProbeResult(
            probe_type=self._get_probe_type(),
            score=0.0,
            confidence=0.0,
            verdict=Verdict.FAIL,
            evidence=[f"Probe error: {error_msg}"],
            details={"error": error_msg},
        )

    @abstractmethod
    def _get_probe_type(self) -> Any:
        """Return the probe type enum value."""
        pass
