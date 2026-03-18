import re
import math
from typing import Any, Optional

from utils.types import ComparisonResult, ModelConfig, Verdict, ProbeType, CompletionResult
from utils.api_client import ModelClient
from analysis.mmd_test import mmd_test


COMPARISON_PROMPTS = {
    "hard_math": [
        {
            "prompt": "Find the remainder when 2^100 is divided by 7. Give only the number.",
            "expected_answer": "2",
            "category": "hard_math",
            "evaluation_method": "number_extraction",
        },
        {
            "prompt": "How many positive integers less than 1000 are divisible by 3 but not by 5? Give only the number.",
            "expected_answer": "267",
            "category": "hard_math",
            "evaluation_method": "number_extraction",
        },
        {
            "prompt": "What is the sum of all prime numbers between 50 and 80? Give only the number.",
            "expected_answer": "454",
            "category": "hard_math",
            "evaluation_method": "number_extraction",
        },
    ],
    "hard_reasoning": [
        {
            "prompt": "Five people (A, B, C, D, E) are sitting in a row. A is not next to B. C is next to D. E is at one end. B is next to C. Who is sitting in the middle? Give only the letter.",
            "expected_answer": "C",
            "category": "hard_reasoning",
            "evaluation_method": "fuzzy_match",
        },
        {
            "prompt": "A clock shows 3:15. What is the exact angle in degrees between the hour hand and minute hand? Give only the number.",
            "expected_answer": "7.5",
            "category": "hard_reasoning",
            "evaluation_method": "number_extraction",
        },
        {
            "prompt": "In a tournament, each of 6 teams plays every other team exactly once. How many total games are played? Give only the number.",
            "expected_answer": "15",
            "category": "hard_reasoning",
            "evaluation_method": "number_extraction",
        },
    ],
    "knowledge": [
        {
            "prompt": "What is the atomic number of Osmium? Give only the number.",
            "expected_answer": "76",
            "category": "knowledge",
            "evaluation_method": "number_extraction",
        },
        {
            "prompt": "Who proved the incompleteness theorems in mathematical logic in 1931?",
            "expected_answer": "Kurt Gödel",
            "category": "knowledge",
            "evaluation_method": "fuzzy_match",
        },
        {
            "prompt": "What is the name of the longest river in Africa?",
            "expected_answer": "Nile",
            "category": "knowledge",
            "evaluation_method": "fuzzy_match",
        },
    ],
    "creative": [
        {
            "prompt": "Write a short paragraph describing what it would feel like to walk on the surface of Europa, Jupiter's moon.",
            "expected_answer": None,
            "category": "creative",
            "evaluation_method": "similarity",
        },
        {
            "prompt": "Explain the concept of entropy to a 10-year-old using a metaphor involving a messy room.",
            "expected_answer": None,
            "category": "creative",
            "evaluation_method": "similarity",
        },
    ],
    "instruction_following": [
        {
            "prompt": "List exactly 5 animals that can fly. Format each on its own line, numbered 1-5. Each animal name must be a single word. Do not include any other text.",
            "expected_answer": None,
            "category": "instruction_following",
            "evaluation_method": "constraint_check",
        },
        {
            "prompt": "Write exactly 3 sentences about the moon. Each sentence must start with a different letter. Do not use the word 'the'.",
            "expected_answer": None,
            "category": "instruction_following",
            "evaluation_method": "constraint_check",
        },
    ],
}


class ComparisonProbe:
    def __init__(
        self,
        test_client: ModelClient,
        reference_client: ModelClient,
        model_config: ModelConfig,
    ):
        self.test_client = test_client
        self.reference_client = reference_client
        self.model_config = model_config

    def run(self) -> ComparisonResult:
        per_prompt = []
        test_responses = []
        ref_responses = []
        test_latencies = []
        ref_latencies = []
        test_correct_count = 0
        ref_correct_count = 0
        total_with_answers = 0
        hard_test_correct = 0
        hard_ref_correct = 0
        hard_total = 0
        similarities = []
        evidence = []

        for category, prompts in COMPARISON_PROMPTS.items():
            for q in prompts:
                try:
                    test_result = self.test_client.complete(
                        q["prompt"], temperature=0.0, max_tokens=1024
                    )
                    ref_result = self.reference_client.complete(
                        q["prompt"], temperature=0.0, max_tokens=1024
                    )

                    test_responses.append(test_result.text)
                    ref_responses.append(ref_result.text)
                    test_latencies.append(test_result.latency_ms)
                    ref_latencies.append(ref_result.latency_ms)

                    test_correct = False
                    ref_correct = False

                    if q["expected_answer"] is not None:
                        test_correct = self._evaluate_answer(
                            test_result.text, q["expected_answer"], q["evaluation_method"]
                        )
                        ref_correct = self._evaluate_answer(
                            ref_result.text, q["expected_answer"], q["evaluation_method"]
                        )
                        total_with_answers += 1
                        if test_correct:
                            test_correct_count += 1
                        if ref_correct:
                            ref_correct_count += 1

                        if category in ("hard_math", "hard_reasoning"):
                            hard_total += 1
                            if test_correct:
                                hard_test_correct += 1
                            if ref_correct:
                                hard_ref_correct += 1

                    sim = self._jaccard_similarity(test_result.text, ref_result.text)
                    similarities.append(sim)

                    per_prompt.append(
                        {
                            "prompt": q["prompt"],
                            "category": category,
                            "test_response": test_result.text[:500],
                            "ref_response": ref_result.text[:500],
                            "test_correct": test_correct,
                            "ref_correct": ref_correct,
                            "similarity": sim,
                            "test_latency_ms": test_result.latency_ms,
                            "ref_latency_ms": ref_result.latency_ms,
                        }
                    )

                except Exception as e:
                    evidence.append(f"Error comparing prompt '{q['prompt'][:50]}...': {e}")

        if not test_responses:
            return ComparisonResult(
                score=0.0,
                confidence=0.0,
                verdict=Verdict.FAIL,
                evidence=["Failed to get any comparison responses"],
            )

        test_accuracy = test_correct_count / total_with_answers if total_with_answers > 0 else 0.0
        ref_accuracy = ref_correct_count / total_with_answers if total_with_answers > 0 else 0.0
        accuracy_delta = test_accuracy - ref_accuracy

        response_similarity = sum(similarities) / len(similarities) if similarities else 0.0

        mean_test_lat = sum(test_latencies) / len(test_latencies) if test_latencies else 1.0
        mean_ref_lat = sum(ref_latencies) / len(ref_latencies) if ref_latencies else 1.0
        latency_ratio = mean_test_lat / mean_ref_lat if mean_ref_lat > 0 else 1.0

        mmd_result = None
        try:
            mmd_result = mmd_test(test_responses, ref_responses, n_permutations=500)
        except Exception as e:
            evidence.append(f"MMD test failed: {e}")

        hard_acc_test = hard_test_correct / hard_total if hard_total > 0 else 0.0
        hard_acc_ref = hard_ref_correct / hard_total if hard_total > 0 else 0.0

        score = 1.0
        if abs(accuracy_delta) > 0.15:
            score -= 0.3
            evidence.append(
                f"Accuracy delta {accuracy_delta:.2f} exceeds threshold (test={test_accuracy:.2f}, ref={ref_accuracy:.2f})"
            )
        if response_similarity < 0.3:
            score -= 0.2
            evidence.append(f"Low response similarity: {response_similarity:.2f}")
        if mmd_result and mmd_result.reject_null:
            score -= 0.3
            evidence.append(
                f"MMD test rejects null (p={mmd_result.p_value:.4f}): distributions differ"
            )
        if latency_ratio > 2.0 or latency_ratio < 0.3:
            score -= 0.2
            evidence.append(f"Latency ratio anomaly: {latency_ratio:.2f}")
        score = max(0.0, min(1.0, score))

        if score >= 0.7:
            verdict = Verdict.PASS
        elif score >= 0.4:
            verdict = Verdict.WARN
        else:
            verdict = Verdict.FAIL

        evidence.extend(
            [
                f"Test accuracy: {test_accuracy:.2%}, Reference accuracy: {ref_accuracy:.2%}",
                f"Mean response similarity: {response_similarity:.2f}",
                f"Latency ratio (test/ref): {latency_ratio:.2f}",
                f"Hard benchmark: test={hard_acc_test:.2%}, ref={hard_acc_ref:.2%}",
            ]
        )

        return ComparisonResult(
            score=score,
            confidence=0.9 if len(per_prompt) >= 10 else 0.6,
            verdict=verdict,
            evidence=evidence,
            reference_provider=self.reference_client.config.name,
            reference_model=self.reference_client.model,
            accuracy_delta=accuracy_delta,
            response_similarity=response_similarity,
            latency_ratio=latency_ratio,
            mmd_result=mmd_result,
            per_prompt_comparison=per_prompt,
            hard_accuracy_test=hard_acc_test,
            hard_accuracy_reference=hard_acc_ref,
        )

    def _evaluate_answer(self, response: str, expected: str, method: str) -> bool:
        response = response.strip()
        expected = expected.strip()

        if method == "exact_match":
            return response.lower() == expected.lower()

        elif method == "number_extraction":
            numbers = re.findall(r"[-+]?\d*\.?\d+", response)
            if not numbers:
                return False
            try:
                expected_val = float(expected)
                for n in numbers:
                    if abs(float(n) - expected_val) < 1e-6:
                        return True
            except ValueError:
                return expected.lower() in response.lower()
            return False

        elif method == "fuzzy_match":
            return expected.lower() in response.lower() or response.lower() in expected.lower()

        return response.lower() == expected.lower()

    def _jaccard_similarity(self, text_a: str, text_b: str) -> float:
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())
        if not words_a and not words_b:
            return 1.0
        union = words_a | words_b
        if not union:
            return 0.0
        return len(words_a & words_b) / len(union)
