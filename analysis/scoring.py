from datetime import datetime
from typing import Optional

from utils.types import (
    LayerScore,
    ProbeResult,
    ProbeType,
    ScoreCard,
    Verdict,
)


class ScoringAggregator:
    DEFAULT_WEIGHTS = {
        ProbeType.IDENTITY.value: 0.20,
        ProbeType.FINGERPRINT.value: 0.20,
        ProbeType.BENCHMARK.value: 0.15,
        ProbeType.LOGPROB.value: 0.10,
        ProbeType.LATENCY.value: 0.10,
        ProbeType.TIER_SIGNATURE.value: 0.15,
        ProbeType.COMPARISON.value: 0.10,
    }

    def __init__(self, weights: Optional[dict[str, float]] = None):
        self.weights = weights or self.DEFAULT_WEIGHTS

    def _normalize_score(self, score: float) -> float:
        return max(0.0, min(1.0, score))

    def aggregate(
        self,
        results: dict[str, ProbeResult],
        provider: str,
        model: str,
        claimed_model: Optional[str] = None,
    ) -> ScoreCard:
        layer_scores: dict[str, LayerScore] = {}
        total_present_weight = 0.0
        weighted_score_sum = 0.0
        weighted_confidence_sum = 0.0

        for probe_type_str, result in results.items():
            if probe_type_str in self.weights:
                total_present_weight += self.weights[probe_type_str]

        for probe_type_str, result in results.items():
            if probe_type_str not in self.weights:
                continue

            weight = self.weights[probe_type_str]

            layer_score = LayerScore(
                probe_type=result.probe_type,
                score=self._normalize_score(result.score),
                confidence=self._normalize_score(result.confidence),
                verdict=result.verdict,
                weight=weight,
                details=result.details,
            )
            layer_scores[probe_type_str] = layer_score

            weighted_score_sum += layer_score.score * weight
            weighted_confidence_sum += layer_score.confidence * weight

        if total_present_weight > 0:
            aggregate_score = weighted_score_sum / total_present_weight
            confidence_level = weighted_confidence_sum / total_present_weight
        else:
            aggregate_score = 0.0
            confidence_level = 0.0

        if aggregate_score >= 0.8:
            overall_verdict = Verdict.PASS
        elif aggregate_score >= 0.5:
            overall_verdict = Verdict.WARN
        else:
            overall_verdict = Verdict.FAIL

        return ScoreCard(
            provider=provider,
            model=model,
            claimed_model=claimed_model,
            layer_scores=layer_scores,
            aggregate_score=self._normalize_score(aggregate_score),
            overall_verdict=overall_verdict,
            confidence_level=self._normalize_score(confidence_level),
            timestamp=datetime.now(),
        )
