import math
from typing import List, Dict, Any, Optional
from utils.types import (
    LogprobResult,
    TokenInfo,
    ModelConfig,
    Verdict,
    ProbeType,
)
from utils.api_client import ModelClient
from utils.data_store import DataStore


class LogprobProbe:
    """
    Probe that analyzes token probability distributions to verify model identity.
    Compares current logprobs against baseline distributions using KL and JS divergence.
    """

    STANDARD_PROMPTS = [
        "The capital of France is",
        "The square root of 144 is",
        "To be or not to be, that is the",
        "The chemical symbol for gold is",
        "The tallest mountain in the world is",
        "The author of 'Romeo and Juliet' is",
        "The speed of light in a vacuum is approximately",
        "The largest planet in our solar system is",
        "The first president of the United States was",
        "The process by which plants make their food is called",
    ]

    def __init__(
        self,
        client: ModelClient,
        model_config: ModelConfig,
        baseline_store: Optional[DataStore] = None,
    ):
        self.client = client
        self.model_config = model_config
        self.baseline_store = baseline_store or DataStore()

    def run(self) -> LogprobResult:
        """
        Runs the logprob analysis probe.
        """
        if not self.client.supports_logprobs:
            return LogprobResult(
                score=0.0,
                confidence=1.0,
                verdict=Verdict.WARN,
                evidence=["Model provider does not support logprobs"],
                distribution_available=False,
            )

        # Load baseline from baselines/logprobs/{model}.json
        baseline_data = self.baseline_store.load_baseline("logprobs", self.model_config.name)
        if not baseline_data:
            return LogprobResult(
                score=0.0,
                confidence=0.0,
                verdict=Verdict.WARN,
                evidence=[f"No baseline found for model {self.model_config.name}"],
                distribution_available=False,
            )

        js_divergences = []
        kl_divergences = []
        overlaps = []
        per_prompt_js = {}

        for prompt in self.STANDARD_PROMPTS:
            if prompt not in baseline_data:
                continue

            try:
                result = self.client.complete(prompt, max_tokens=1, logprobs=True, temperature=0.0)
                if not result.logprobs:
                    continue

                current_top_logprobs = result.logprobs[0].top_logprobs
                baseline_top_logprobs = baseline_data[prompt]

                current_tokens = [
                    TokenInfo(token=t, logprob=lp) for t, lp in current_top_logprobs.items()
                ]
                baseline_tokens = [
                    TokenInfo(token=t, logprob=lp) for t, lp in baseline_top_logprobs.items()
                ]

                kl = self._compute_kl_divergence(baseline_tokens, current_tokens)
                js = self._compute_js_divergence(baseline_tokens, current_tokens)

                # Top-k overlap (top 5)
                current_top5 = set(
                    sorted(
                        current_top_logprobs.keys(),
                        key=lambda x: current_top_logprobs[x],
                        reverse=True,
                    )[:5]
                )
                baseline_top5 = set(
                    sorted(
                        baseline_top_logprobs.keys(),
                        key=lambda x: baseline_top_logprobs[x],
                        reverse=True,
                    )[:5]
                )
                overlap = (
                    len(current_top5.intersection(baseline_top5)) / 5.0 if baseline_top5 else 0.0
                )

                js_divergences.append(js)
                kl_divergences.append(kl)
                overlaps.append(overlap)
                per_prompt_js[prompt] = js

            except Exception:
                continue

        if not js_divergences:
            return LogprobResult(
                score=0.0,
                confidence=0.0,
                verdict=Verdict.FAIL,
                evidence=["Failed to collect logprobs for any standard prompt"],
                distribution_available=False,
            )

        avg_js = sum(js_divergences) / len(js_divergences)
        avg_kl = sum(kl_divergences) / len(kl_divergences)
        avg_overlap = sum(overlaps) / len(overlaps)

        # Score: 1.0 - min(1.0, avg_js_divergence / threshold) where threshold=0.5
        threshold = 0.5
        score = 1.0 - min(1.0, avg_js / threshold)

        # Verdict based on JS divergence
        if avg_js < 0.1:
            verdict = Verdict.PASS
        elif avg_js < 0.3:
            verdict = Verdict.WARN
        else:
            verdict = Verdict.FAIL

        return LogprobResult(
            score=score,
            confidence=1.0,
            verdict=verdict,
            kl_divergence=avg_kl,
            js_divergence=avg_js,
            top_k_overlap=avg_overlap,
            distribution_available=True,
            per_prompt_divergence=per_prompt_js,
            evidence=[
                f"Average JS divergence: {avg_js:.4f}",
                f"Average top-5 token overlap: {avg_overlap:.2f}",
            ],
        )

    def _compute_kl_divergence(
        self, logprobs_p: List[TokenInfo], logprobs_q: List[TokenInfo]
    ) -> float:
        """
        Computes KL divergence D_KL(P||Q) = sum P(i) * log(P(i)/Q(i)).
        P is the baseline distribution, Q is the current distribution.
        """
        p_probs = {t.token: math.exp(t.logprob) for t in logprobs_p}
        q_probs = {t.token: math.exp(t.logprob) for t in logprobs_q}

        sum_p = sum(p_probs.values())
        sum_q = sum(q_probs.values())

        if sum_p > 0:
            p_probs = {k: v / sum_p for k, v in p_probs.items()}
        if sum_q > 0:
            q_probs = {k: v / sum_q for k, v in q_probs.items()}

        all_tokens = set(p_probs.keys()) | set(q_probs.keys())
        kl = 0.0
        epsilon = 1e-10

        for token in all_tokens:
            p = p_probs.get(token, 0.0)
            q = q_probs.get(token, 0.0)

            if p > 0:
                kl += p * math.log(p / (q + epsilon))

        return kl

    def _compute_js_divergence(
        self, logprobs_p: List[TokenInfo], logprobs_q: List[TokenInfo]
    ) -> float:
        """
        Computes JS divergence D_JS(P||Q) = 0.5 * D_KL(P||M) + 0.5 * D_KL(Q||M)
        where M = 0.5 * (P + Q).
        """
        p_probs = {t.token: math.exp(t.logprob) for t in logprobs_p}
        q_probs = {t.token: math.exp(t.logprob) for t in logprobs_q}

        sum_p = sum(p_probs.values())
        sum_q = sum(q_probs.values())

        if sum_p > 0:
            p_probs = {k: v / sum_p for k, v in p_probs.items()}
        if sum_q > 0:
            q_probs = {k: v / sum_q for k, v in q_probs.items()}

        all_tokens = set(p_probs.keys()) | set(q_probs.keys())

        m_probs = {}
        for token in all_tokens:
            m_probs[token] = 0.5 * (p_probs.get(token, 0.0) + q_probs.get(token, 0.0))

        def kl_div(dist_probs: Dict[str, float], ref_probs: Dict[str, float]) -> float:
            div = 0.0
            epsilon = 1e-10
            for token, p in dist_probs.items():
                if p > 0:
                    q = ref_probs.get(token, 0.0)
                    div += p * math.log(p / (q + epsilon))
            return div

        return 0.5 * kl_div(p_probs, m_probs) + 0.5 * kl_div(q_probs, m_probs)
