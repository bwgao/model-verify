import statistics
import time
from typing import List, Dict, Any, Optional
from utils.types import LatencyResult, ModelConfig, Verdict, ProbeType
from utils.api_client import ModelClient


class LatencyProbe:
    TEST_PROMPTS = {
        "short_10_tokens": "Tell me a very short joke.",
        "medium_100_tokens": "Explain the concept of 'separation of concerns' in software engineering and why it is important for building maintainable systems. Provide a few examples of how it is applied in practice.",
        "long_500_tokens": "Write a long, detailed story about a robot that discovers it has the ability to dream. Describe its first dream in vivid detail, how it feels about this new experience, and how it changes its perspective on its own existence and its relationship with humans. Include dialogue between the robot and its creator about the nature of consciousness.",
    }

    def __init__(self, client: ModelClient, model_config: ModelConfig, n_runs: int = 5):
        self.client = client
        self.model_config = model_config
        self.n_runs = n_runs

    def _compute_stats(self, values: List[float]) -> Dict[str, float]:
        if not values:
            return {"mean": 0.0, "median": 0.0, "p95": 0.0, "std": 0.0}

        sorted_values = sorted(values)
        n = len(sorted_values)

        return {
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "p95": sorted_values[min(n - 1, int(n * 0.95))],
            "std": statistics.stdev(values) if n > 1 else 0.0,
        }

    def _calculate_metric_score(
        self, observed: float, expected_min: Optional[float], expected_max: Optional[float]
    ) -> float:
        if expected_min is None or expected_max is None:
            return 1.0

        if expected_min <= observed <= expected_max:
            return 1.0

        if observed < expected_min:
            margin = 0.2 * expected_min
            return 0.5 if observed >= expected_min - margin else 0.0

        if observed > expected_max:
            margin = 0.2 * expected_max
            return 0.5 if observed <= expected_max + margin else 0.0

        return 1.0

    def run(self) -> LatencyResult:
        ttfts = []
        tpss = []
        latencies = []
        evidence = []

        for prompt_type, prompt in self.TEST_PROMPTS.items():
            for i in range(self.n_runs):
                try:
                    result = self.client.complete_streaming(prompt, max_tokens=128)

                    if result.ttft_ms is not None:
                        ttfts.append(result.ttft_ms)
                    if result.tps is not None:
                        tpss.append(result.tps)
                    latencies.append(result.latency_ms)

                except Exception as e:
                    evidence.append(f"Error during {prompt_type} run {i + 1}: {str(e)}")

        ttft_stats = self._compute_stats(ttfts)
        tps_stats = self._compute_stats(tpss)

        chars = self.model_config.characteristics
        anomaly_detected = False
        scores = []
        baseline_comparison = {}

        if chars.expected_ttft_min is not None and chars.expected_ttft_max is not None:
            mean_ttft = ttft_stats["mean"]
            ttft_score = self._calculate_metric_score(
                mean_ttft, chars.expected_ttft_min, chars.expected_ttft_max
            )
            scores.append(ttft_score)

            baseline_comparison["ttft_min_delta"] = mean_ttft - chars.expected_ttft_min
            baseline_comparison["ttft_max_delta"] = mean_ttft - chars.expected_ttft_max

            if ttft_score < 1.0:
                anomaly_detected = True
                evidence.append(
                    f"Mean TTFT ({mean_ttft:.2f}ms) outside expected range "
                    f"[{chars.expected_ttft_min}, {chars.expected_ttft_max}]"
                )

        if chars.expected_tps_min is not None and chars.expected_tps_max is not None:
            mean_tps = tps_stats["mean"]
            tps_score = self._calculate_metric_score(
                mean_tps, chars.expected_tps_min, chars.expected_tps_max
            )
            scores.append(tps_score)

            baseline_comparison["tps_min_delta"] = mean_tps - chars.expected_tps_min
            baseline_comparison["tps_max_delta"] = mean_tps - chars.expected_tps_max

            if tps_score < 1.0:
                anomaly_detected = True
                evidence.append(
                    f"Mean TPS ({mean_tps:.2f}) outside expected range "
                    f"[{chars.expected_tps_min}, {chars.expected_tps_max}]"
                )

        score = sum(scores) / len(scores) if scores else 1.0

        if not anomaly_detected and score >= 0.8:
            verdict = Verdict.PASS
        elif score >= 0.5:
            verdict = Verdict.WARN
        else:
            verdict = Verdict.FAIL

        return LatencyResult(
            score=score,
            confidence=0.9 if len(ttfts) >= (len(self.TEST_PROMPTS) * self.n_runs * 0.8) else 0.5,
            verdict=verdict,
            evidence=evidence,
            ttft_stats=ttft_stats,
            tps_stats=tps_stats,
            baseline_comparison=baseline_comparison,
            anomaly_detected=anomaly_detected,
            details={
                "n_runs_per_prompt": self.n_runs,
                "total_runs_attempted": len(self.TEST_PROMPTS) * self.n_runs,
                "total_runs_successful": len(ttfts),
                "total_latency_stats": self._compute_stats(latencies),
            },
        )
