"""
Model Verification System - Shared Types

This module defines all dataclasses and type definitions used across the project.
Having all types in one place enables maximum parallelism in development.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class Verdict(str, Enum):
    """Verification verdict levels."""

    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


class ProbeType(str, Enum):
    """Types of verification probes."""

    IDENTITY = "identity"
    FINGERPRINT = "fingerprint"
    BENCHMARK = "benchmark"
    LOGPROB = "logprob"
    LATENCY = "latency"
    TIER_SIGNATURE = "tier_signature"
    COMPARISON = "comparison"


class ProviderType(str, Enum):
    OPENAI = "openai"
    OPENAI_COMPATIBLE = "openai_compatible"
    ANTHROPIC = "anthropic"


# =============================================================================
# Configuration Types
# =============================================================================


@dataclass
class ProviderConfig:
    """Configuration for an API provider (official or reseller)."""

    name: str
    type: ProviderType
    base_url: str
    api_key_env: str
    models: list[str]
    claims: dict[str, str] = field(default_factory=dict)
    # Optional settings
    timeout: float = 60.0
    max_retries: int = 3
    rate_limit_rpm: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type.value,
            "base_url": self.base_url,
            "api_key_env": self.api_key_env,
            "models": self.models,
            "claims": self.claims,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "rate_limit_rpm": self.rate_limit_rpm,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProviderConfig":
        return cls(
            name=data["name"],
            type=ProviderType(data["type"]),
            base_url=data["base_url"],
            api_key_env=data["api_key_env"],
            models=data["models"],
            claims=data.get("claims", {}),
            timeout=data.get("timeout", 60.0),
            max_retries=data.get("max_retries", 3),
            rate_limit_rpm=data.get("rate_limit_rpm"),
        )


@dataclass
class ModelCharacteristics:
    """Expected characteristics of a model."""

    # Identity markers
    family: str  # e.g., "claude", "gpt", "gemini"
    developer: str  # e.g., "anthropic", "openai"
    version: Optional[str] = None  # e.g., "4", "opus-4"

    # Capability baselines (optional)
    mmlu_baseline: Optional[float] = None
    math_baseline: Optional[float] = None
    humaneval_baseline: Optional[float] = None

    # Hard benchmark baselines for tier differentiation
    hard_math_baseline: Optional[float] = None
    hard_code_baseline: Optional[float] = None
    hard_reasoning_baseline: Optional[float] = None
    expected_response_length: Optional[str] = None  # "short", "medium", "long", "very_long"
    tier: Optional[str] = None  # "opus", "sonnet", "haiku", etc.

    # Latency expectations (in ms)
    expected_ttft_min: Optional[float] = None
    expected_ttft_max: Optional[float] = None
    expected_tps_min: Optional[float] = None
    expected_tps_max: Optional[float] = None


@dataclass
class ModelConfig:
    """Configuration for a specific model."""

    name: str
    display_name: str
    characteristics: ModelCharacteristics
    aliases: list[str] = field(default_factory=list)
    context_window: int = 8192
    supports_logprobs: bool = False
    supports_streaming: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "characteristics": {
                "family": self.characteristics.family,
                "developer": self.characteristics.developer,
                "version": self.characteristics.version,
                "mmlu_baseline": self.characteristics.mmlu_baseline,
                "math_baseline": self.characteristics.math_baseline,
                "humaneval_baseline": self.characteristics.humaneval_baseline,
                "hard_math_baseline": self.characteristics.hard_math_baseline,
                "hard_code_baseline": self.characteristics.hard_code_baseline,
                "hard_reasoning_baseline": self.characteristics.hard_reasoning_baseline,
                "expected_response_length": self.characteristics.expected_response_length,
                "tier": self.characteristics.tier,
                "expected_ttft_min": self.characteristics.expected_ttft_min,
                "expected_ttft_max": self.characteristics.expected_ttft_max,
                "expected_tps_min": self.characteristics.expected_tps_min,
                "expected_tps_max": self.characteristics.expected_tps_max,
            },
            "aliases": self.aliases,
            "context_window": self.context_window,
            "supports_logprobs": self.supports_logprobs,
            "supports_streaming": self.supports_streaming,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelConfig":
        chars_data = data["characteristics"]
        characteristics = ModelCharacteristics(
            family=chars_data["family"],
            developer=chars_data["developer"],
            version=chars_data.get("version"),
            mmlu_baseline=chars_data.get("mmlu_baseline"),
            math_baseline=chars_data.get("math_baseline"),
            humaneval_baseline=chars_data.get("humaneval_baseline"),
            hard_math_baseline=chars_data.get("hard_math_baseline"),
            hard_code_baseline=chars_data.get("hard_code_baseline"),
            hard_reasoning_baseline=chars_data.get("hard_reasoning_baseline"),
            expected_response_length=chars_data.get("expected_response_length"),
            tier=chars_data.get("tier"),
            expected_ttft_min=chars_data.get("expected_ttft_min"),
            expected_ttft_max=chars_data.get("expected_ttft_max"),
            expected_tps_min=chars_data.get("expected_tps_min"),
            expected_tps_max=chars_data.get("expected_tps_max"),
        )
        return cls(
            name=data["name"],
            display_name=data["display_name"],
            characteristics=characteristics,
            aliases=data.get("aliases", []),
            context_window=data.get("context_window", 8192),
            supports_logprobs=data.get("supports_logprobs", False),
            supports_streaming=data.get("supports_streaming", True),
        )


# =============================================================================
# API Response Types
# =============================================================================


@dataclass
class TokenInfo:
    """Token probability information."""

    token: str
    logprob: float
    top_logprobs: dict[str, float] = field(default_factory=dict)


@dataclass
class CompletionResult:
    """Result from a model completion API call."""

    # Core response
    text: str
    model: str

    # Latency metrics
    latency_ms: float
    ttft_ms: Optional[float] = None  # Time to first token (streaming)
    tps: Optional[float] = None  # Tokens per second

    # Token counts
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None

    # Logprobs (if available)
    logprobs: list[TokenInfo] = field(default_factory=list)

    # Metadata
    finish_reason: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    raw_response: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "model": self.model,
            "latency_ms": self.latency_ms,
            "ttft_ms": self.ttft_ms,
            "tps": self.tps,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "logprobs": [
                {"token": t.token, "logprob": t.logprob, "top_logprobs": t.top_logprobs}
                for t in self.logprobs
            ],
            "finish_reason": self.finish_reason,
            "timestamp": self.timestamp.isoformat(),
        }


# =============================================================================
# Probe Result Types
# =============================================================================


@dataclass
class ProbeResult:
    """Base class for all probe results."""

    probe_type: ProbeType
    score: float  # 0.0 to 1.0, higher = more likely genuine
    confidence: float  # 0.0 to 1.0, confidence in the score
    verdict: Verdict
    evidence: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "probe_type": self.probe_type.value,
            "score": self.score,
            "confidence": self.confidence,
            "verdict": self.verdict.value,
            "evidence": self.evidence,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class IdentityResult(ProbeResult):
    """Result from identity probing."""

    probe_type: ProbeType = field(default=ProbeType.IDENTITY, init=False)

    # Identity-specific fields
    claimed_identity: Optional[str] = None
    matched_patterns: list[str] = field(default_factory=list)
    mismatched_patterns: list[str] = field(default_factory=list)
    identity_responses: dict[str, str] = field(default_factory=dict)


@dataclass
class FingerprintResult(ProbeResult):
    """Result from behavioral fingerprinting."""

    probe_type: ProbeType = field(default=ProbeType.FINGERPRINT, init=False)

    # Fingerprint-specific fields
    predicted_model: Optional[str] = None
    predicted_family: Optional[str] = None
    similarity_scores: dict[str, float] = field(default_factory=dict)
    category_scores: dict[str, float] = field(default_factory=dict)


@dataclass
class BenchmarkResult(ProbeResult):
    """Result from capability benchmarking."""

    probe_type: ProbeType = field(default=ProbeType.BENCHMARK, init=False)

    # Benchmark-specific fields
    overall_accuracy: float = 0.0
    category_accuracies: dict[str, float] = field(default_factory=dict)
    baseline_deltas: dict[str, float] = field(default_factory=dict)
    question_results: list[dict[str, Any]] = field(default_factory=list)
    total_questions: int = 0
    correct_answers: int = 0


@dataclass
class LogprobResult(ProbeResult):
    """Result from logprob analysis."""

    probe_type: ProbeType = field(default=ProbeType.LOGPROB, init=False)

    # Logprob-specific fields
    kl_divergence: Optional[float] = None
    js_divergence: Optional[float] = None
    top_k_overlap: float = 0.0
    distribution_available: bool = False
    per_prompt_divergence: dict[str, float] = field(default_factory=dict)


@dataclass
class LatencyResult(ProbeResult):
    """Result from latency fingerprinting."""

    probe_type: ProbeType = field(default=ProbeType.LATENCY, init=False)

    # Latency-specific fields
    ttft_stats: dict[str, float] = field(default_factory=dict)  # mean, median, p95, std
    tps_stats: dict[str, float] = field(default_factory=dict)
    baseline_comparison: dict[str, float] = field(default_factory=dict)
    anomaly_detected: bool = False


@dataclass
class TierSignatureResult(ProbeResult):
    probe_type: ProbeType = field(default=ProbeType.TIER_SIGNATURE, init=False)

    predicted_tier: Optional[str] = None
    tier_scores: dict[str, float] = field(default_factory=dict)
    dimension_scores: dict[str, float] = field(default_factory=dict)
    response_length_stats: dict[str, float] = field(default_factory=dict)
    reasoning_depth_stats: dict[str, float] = field(default_factory=dict)
    instruction_following_score: float = 0.0
    hard_benchmark_accuracy: float = 0.0
    category_accuracies: dict[str, float] = field(default_factory=dict)


@dataclass
class ComparisonResult(ProbeResult):
    probe_type: ProbeType = field(default=ProbeType.COMPARISON, init=False)

    reference_provider: str = ""
    reference_model: str = ""
    accuracy_delta: float = 0.0
    response_similarity: float = 0.0
    latency_ratio: float = 0.0
    mmd_result: Optional["MMDResult"] = None
    per_prompt_comparison: list[dict[str, Any]] = field(default_factory=list)
    hard_accuracy_test: float = 0.0
    hard_accuracy_reference: float = 0.0


# =============================================================================
# Aggregation Types
# =============================================================================


@dataclass
class LayerScore:
    """Score for a single verification layer."""

    probe_type: ProbeType
    score: float
    confidence: float
    verdict: Verdict
    weight: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoreCard:
    """Aggregated multi-dimensional verification score."""

    provider: str
    model: str
    claimed_model: Optional[str]

    # Per-layer scores
    layer_scores: dict[str, LayerScore] = field(default_factory=dict)

    # Aggregate
    aggregate_score: float = 0.0
    overall_verdict: Verdict = Verdict.WARN
    confidence_level: float = 0.0

    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    duration_seconds: float = 0.0

    def get_weighted_score(self) -> float:
        """Calculate weighted aggregate score."""
        if not self.layer_scores:
            return 0.0
        total_weight = sum(ls.weight for ls in self.layer_scores.values())
        if total_weight == 0:
            return 0.0
        weighted_sum = sum(ls.score * ls.weight for ls in self.layer_scores.values())
        return weighted_sum / total_weight

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "claimed_model": self.claimed_model,
            "layer_scores": {
                k: {
                    "probe_type": v.probe_type.value,
                    "score": v.score,
                    "confidence": v.confidence,
                    "verdict": v.verdict.value,
                    "weight": v.weight,
                    "details": v.details,
                }
                for k, v in self.layer_scores.items()
            },
            "aggregate_score": self.aggregate_score,
            "overall_verdict": self.overall_verdict.value,
            "confidence_level": self.confidence_level,
            "timestamp": self.timestamp.isoformat(),
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class VerificationReport:
    scorecard: ScoreCard

    identity_result: Optional[IdentityResult] = None
    fingerprint_result: Optional[FingerprintResult] = None
    benchmark_result: Optional[BenchmarkResult] = None
    logprob_result: Optional[LogprobResult] = None
    latency_result: Optional[LatencyResult] = None
    tier_signature_result: Optional[TierSignatureResult] = None
    comparison_result: Optional[ComparisonResult] = None

    executive_summary: str = ""
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scorecard": self.scorecard.to_dict(),
            "identity_result": self.identity_result.to_dict() if self.identity_result else None,
            "fingerprint_result": self.fingerprint_result.to_dict()
            if self.fingerprint_result
            else None,
            "benchmark_result": self.benchmark_result.to_dict() if self.benchmark_result else None,
            "logprob_result": self.logprob_result.to_dict() if self.logprob_result else None,
            "latency_result": self.latency_result.to_dict() if self.latency_result else None,
            "tier_signature_result": self.tier_signature_result.to_dict()
            if self.tier_signature_result
            else None,
            "comparison_result": self.comparison_result.to_dict()
            if self.comparison_result
            else None,
            "executive_summary": self.executive_summary,
            "recommendations": self.recommendations,
        }


# =============================================================================
# MMD Test Types
# =============================================================================


@dataclass
class MMDResult:
    """Result from MMD two-sample test."""

    mmd_statistic: float
    p_value: float
    reject_null: bool  # True if distributions are significantly different
    n_samples_a: int
    n_samples_b: int
    n_permutations: int
    effect_size: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "mmd_statistic": self.mmd_statistic,
            "p_value": self.p_value,
            "reject_null": self.reject_null,
            "n_samples_a": self.n_samples_a,
            "n_samples_b": self.n_samples_b,
            "n_permutations": self.n_permutations,
            "effect_size": self.effect_size,
        }
