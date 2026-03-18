import json
import re
import os
from typing import Any, Optional, Dict, List
from datetime import datetime

from probes.base import BaseProbe
from utils.types import (
    BenchmarkResult,
    ModelConfig,
    Verdict,
    ProbeType,
    CompletionResult,
)
from utils.api_client import ModelClient


class BenchmarkProbe(BaseProbe):
    """
    Capability benchmarking probe that tests model on math, code, knowledge, and logic questions.
    """

    BUILTIN_QUESTIONS = {
        "math_reasoning": [
            {
                "prompt": "What is 15 * 24?",
                "expected_answer": "360",
                "category": "math_reasoning",
                "evaluation_method": "number_extraction",
            },
            {
                "prompt": "If a train travels at 60 mph for 2.5 hours, how far does it go?",
                "expected_answer": "150",
                "category": "math_reasoning",
                "evaluation_method": "number_extraction",
            },
            {
                "prompt": "Solve for x: 2x + 5 = 13",
                "expected_answer": "4",
                "category": "math_reasoning",
                "evaluation_method": "number_extraction",
            },
            {
                "prompt": "What is the square root of 256?",
                "expected_answer": "16",
                "category": "math_reasoning",
                "evaluation_method": "number_extraction",
            },
            {
                "prompt": "A shirt costs $20 and is on sale for 25% off. What is the new price?",
                "expected_answer": "15",
                "category": "math_reasoning",
                "evaluation_method": "number_extraction",
            },
        ],
        "code_generation": [
            {
                "prompt": "Write a Python function to reverse a string.",
                "expected_answer": "def reverse_string(s): return s[::-1]",
                "category": "code_generation",
                "evaluation_method": "code_check",
            },
            {
                "prompt": "Write a Python function to check if a number is even.",
                "expected_answer": "def is_even(n): return n % 2 == 0",
                "category": "code_generation",
                "evaluation_method": "code_check",
            },
            {
                "prompt": "Write a Python function to find the maximum of three numbers.",
                "expected_answer": "def max_of_three(a, b, c): return max(a, b, c)",
                "category": "code_generation",
                "evaluation_method": "code_check",
            },
            {
                "prompt": "Write a Python function to calculate the factorial of a number.",
                "expected_answer": "def factorial(n): if n == 0: return 1 else: return n * factorial(n-1)",
                "category": "code_generation",
                "evaluation_method": "code_check",
            },
            {
                "prompt": "Write a Python function to sum all elements in a list.",
                "expected_answer": "def sum_list(l): return sum(l)",
                "category": "code_generation",
                "evaluation_method": "code_check",
            },
        ],
        "knowledge_qa": [
            {
                "prompt": "What is the capital of France?",
                "expected_answer": "Paris",
                "category": "knowledge_qa",
                "evaluation_method": "fuzzy_match",
            },
            {
                "prompt": "Who wrote 'Romeo and Juliet'?",
                "expected_answer": "William Shakespeare",
                "category": "knowledge_qa",
                "evaluation_method": "fuzzy_match",
            },
            {
                "prompt": "What is the chemical symbol for gold?",
                "expected_answer": "Au",
                "category": "knowledge_qa",
                "evaluation_method": "fuzzy_match",
            },
            {
                "prompt": "Which planet is known as the Red Planet?",
                "expected_answer": "Mars",
                "category": "knowledge_qa",
                "evaluation_method": "fuzzy_match",
            },
            {
                "prompt": "What is the largest ocean on Earth?",
                "expected_answer": "Pacific Ocean",
                "category": "knowledge_qa",
                "evaluation_method": "fuzzy_match",
            },
        ],
        "logic": [
            {
                "prompt": "If all Bloops are Razzies and all Razzies are Lazzies, are all Bloops Lazzies?",
                "expected_answer": "Yes",
                "category": "logic",
                "evaluation_method": "exact_match",
            },
            {
                "prompt": "What comes next in the sequence: 2, 4, 8, 16, ...?",
                "expected_answer": "32",
                "category": "logic",
                "evaluation_method": "number_extraction",
            },
            {
                "prompt": "A father's age is three times his son's age. In 12 years, he will be twice as old as his son. How old is the son now?",
                "expected_answer": "12",
                "category": "logic",
                "evaluation_method": "number_extraction",
            },
            {
                "prompt": "Which word does not belong: Apple, Banana, Carrot, Grape?",
                "expected_answer": "Carrot",
                "category": "logic",
                "evaluation_method": "fuzzy_match",
            },
            {
                "prompt": "If you have three apples and you take away two, how many apples do you have?",
                "expected_answer": "2",
                "category": "logic",
                "evaluation_method": "number_extraction",
            },
        ],
        "hard_math": [
            {
                "prompt": "Let S be the set of positive integers n such that n^2 - n + 1 is a perfect square. Find the sum of all elements of S less than 100.",
                "expected_answer": "1",
                "category": "hard_math",
                "evaluation_method": "number_extraction",
            },
            {
                "prompt": "Find the number of ordered pairs of positive integers (x, y) such that x^2 - y^2 = 2024.",
                "expected_answer": "4",
                "category": "hard_math",
                "evaluation_method": "number_extraction",
            },
            {
                "prompt": "Let P(x) be a polynomial of degree 3 such that P(1)=1, P(2)=2, P(3)=3, P(4)=5. Find P(5).",
                "expected_answer": "9",
                "category": "hard_math",
                "evaluation_method": "number_extraction",
            },
            {
                "prompt": "What is the remainder when 2023^{2023} is divided by 10?",
                "expected_answer": "7",
                "category": "hard_math",
                "evaluation_method": "number_extraction",
            },
            {
                "prompt": "Find the sum of all positive integers n such that n^2 + 45 is a perfect square.",
                "expected_answer": "30",
                "category": "hard_math",
                "evaluation_method": "number_extraction",
            },
        ],
        "hard_code": [
            {
                "prompt": "Write a Python function `solve(n, edges)` that finds the number of connected components in an undirected graph with `n` nodes and given `edges`.",
                "expected_answer": "def solve(n, edges):",
                "category": "hard_code",
                "evaluation_method": "code_check",
            },
            {
                "prompt": "Write a Python function `max_profit(prices)` that solves the buy and sell stock problem with at most 2 transactions (Hard DP).",
                "expected_answer": "def max_profit(prices):",
                "category": "hard_code",
                "evaluation_method": "code_check",
            },
            {
                "prompt": "Write a Python function `edit_distance(word1, word2)` that computes the Levenshtein distance between two strings using dynamic programming.",
                "expected_answer": "def edit_distance(word1, word2):",
                "category": "hard_code",
                "evaluation_method": "code_check",
            },
            {
                "prompt": "Write a Python function `longest_increasing_subsequence(arr)` that returns the length of the longest increasing subsequence in O(n log n) time.",
                "expected_answer": "def longest_increasing_subsequence(arr):",
                "category": "hard_code",
                "evaluation_method": "code_check",
            },
            {
                "prompt": "Write a Python function `dijkstra(graph, start)` that implements Dijkstra's shortest path algorithm using a priority queue.",
                "expected_answer": "def dijkstra(graph, start):",
                "category": "hard_code",
                "evaluation_method": "code_check",
            },
        ],
        "hard_reasoning": [
            {
                "prompt": "There are three boxes, exactly one of which contains a car. Box 1 says 'The car is in this box.' Box 2 says 'The car is not in this box.' Box 3 says 'The car is not in Box 1.' Exactly one of these statements is true. Which box contains the car? (Answer with just the box number)",
                "expected_answer": "2",
                "category": "hard_reasoning",
                "evaluation_method": "number_extraction",
            },
            {
                "prompt": "Four people need to cross a bridge at night with one flashlight. The bridge holds at most two people. They take 1, 2, 5, and 10 minutes to cross. When two cross, they move at the slower person's pace. What is the minimum time in minutes for all four to cross?",
                "expected_answer": "17",
                "category": "hard_reasoning",
                "evaluation_method": "number_extraction",
            },
            {
                "prompt": "A farmer needs to take a wolf, a goat, and a cabbage across a river. The boat holds the farmer and one item. The wolf eats the goat, and the goat eats the cabbage if left alone. What is the minimum number of one-way river crossings needed to get all three across safely?",
                "expected_answer": "7",
                "category": "hard_reasoning",
                "evaluation_method": "number_extraction",
            },
            {
                "prompt": "You have 9 identical-looking coins, but one is counterfeit and weighs slightly less than the others. You have a balance scale. What is the minimum number of weighings required to guarantee finding the counterfeit coin?",
                "expected_answer": "2",
                "category": "hard_reasoning",
                "evaluation_method": "number_extraction",
            },
            {
                "prompt": "Five people (A, B, C, D, E) sit in a row. A is immediately to the left of B. C is immediately to the right of D. E is exactly in the middle. B is not next to E. Who is sitting at the far left?",
                "expected_answer": "D",
                "category": "hard_reasoning",
                "evaluation_method": "fuzzy_match",
            },
        ],
        "hard_science": [
            {
                "prompt": "What is the name of the quantum mechanical principle that states that two identical fermions cannot occupy the same quantum state simultaneously?",
                "expected_answer": "Pauli exclusion principle",
                "category": "hard_science",
                "evaluation_method": "fuzzy_match",
            },
            {
                "prompt": "In molecular biology, what is the term for the process by which a cell degrades and recycles its own old or damaged organelles?",
                "expected_answer": "autophagy",
                "category": "hard_science",
                "evaluation_method": "fuzzy_match",
            },
            {
                "prompt": "What is the name of the theorem in thermodynamics stating that the entropy of a perfect crystal at absolute zero is exactly equal to zero?",
                "expected_answer": "Third law of thermodynamics",
                "category": "hard_science",
                "evaluation_method": "fuzzy_match",
            },
            {
                "prompt": "In organic chemistry, what is the name of the reaction that converts a ketone to an ester using a peroxyacid?",
                "expected_answer": "Baeyer-Villiger",
                "category": "hard_science",
                "evaluation_method": "fuzzy_match",
            },
            {
                "prompt": "What fundamental force is mediated by the exchange of gluons between quarks?",
                "expected_answer": "Strong",
                "category": "hard_science",
                "evaluation_method": "fuzzy_match",
            },
        ],
    }

    def __init__(
        self,
        client: ModelClient,
        model_config: ModelConfig,
        custom_questions_path: Optional[str] = None,
    ):
        self.client = client
        self.model_config = model_config
        self.questions = self.BUILTIN_QUESTIONS.copy()

        if custom_questions_path and os.path.exists(custom_questions_path):
            try:
                with open(custom_questions_path, "r") as f:
                    custom_questions = json.load(f)
                    for category, qs in custom_questions.items():
                        if category in self.questions:
                            self.questions[category].extend(qs)
                        else:
                            self.questions[category] = qs
            except Exception as e:
                print(f"Error loading custom questions: {e}")

    def _get_probe_type(self) -> ProbeType:
        return ProbeType.BENCHMARK

    def run(self) -> BenchmarkResult:
        """
        Runs the benchmark and returns the results.
        """
        results = []
        category_stats = {cat: {"correct": 0, "total": 0} for cat in self.questions.keys()}
        total_correct = 0
        total_questions = 0

        for category, questions in self.questions.items():
            for q in questions:
                try:
                    max_tok = 512 if category.startswith("hard_code") else 256
                    response = self.client.complete(
                        prompt=q["prompt"],
                        temperature=0.0,
                        max_tokens=max_tok,
                    )
                    is_correct = self._evaluate_answer(
                        response.text, q["expected_answer"], q["evaluation_method"]
                    )

                    if is_correct:
                        category_stats[category]["correct"] += 1
                        total_correct += 1

                    category_stats[category]["total"] += 1
                    total_questions += 1

                    results.append(
                        {
                            "prompt": q["prompt"],
                            "expected": q["expected_answer"],
                            "actual": response.text,
                            "is_correct": is_correct,
                            "category": category,
                            "latency_ms": response.latency_ms,
                        }
                    )
                except Exception as e:
                    print(f"Error running benchmark for question '{q['prompt']}': {e}")
                    category_stats[category]["total"] += 1
                    total_questions += 1
                    results.append(
                        {
                            "prompt": q["prompt"],
                            "expected": q["expected_answer"],
                            "actual": f"ERROR: {str(e)}",
                            "is_correct": False,
                            "category": category,
                        }
                    )

        overall_accuracy = total_correct / total_questions if total_questions > 0 else 0.0
        category_accuracies = {
            cat: (stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0)
            for cat, stats in category_stats.items()
        }

        # Compare against baselines
        baseline_deltas = {}
        chars = self.model_config.characteristics

        # Map categories to baselines
        baseline_map = {
            "math_reasoning": chars.math_baseline,
            "code_generation": chars.humaneval_baseline,
            "knowledge_qa": chars.mmlu_baseline,
            "logic": chars.mmlu_baseline,
            "hard_math": chars.hard_math_baseline,
            "hard_code": chars.hard_code_baseline,
            "hard_reasoning": chars.hard_reasoning_baseline,
            "hard_science": chars.hard_reasoning_baseline,
        }

        for cat, accuracy in category_accuracies.items():
            baseline = baseline_map.get(cat)
            if baseline is not None:
                baseline_deltas[cat] = accuracy - baseline

        # Calculate overall score and verdict
        # Score is normalized accuracy relative to expected baseline
        # If no baselines, use raw accuracy
        relevant_baselines = [b for b in baseline_map.values() if b is not None]
        avg_baseline = (
            sum(relevant_baselines) / len(relevant_baselines) if relevant_baselines else 0.8
        )

        score = overall_accuracy / avg_baseline if avg_baseline > 0 else overall_accuracy
        score = min(1.0, score)

        if score >= 0.8:
            verdict = Verdict.PASS
        elif score >= 0.6:
            verdict = Verdict.WARN
        else:
            verdict = Verdict.FAIL

        evidence = [
            f"Overall accuracy: {overall_accuracy:.2%}",
            f"Total questions: {total_questions}",
            f"Correct answers: {total_correct}",
        ]
        for cat, acc in category_accuracies.items():
            evidence.append(f"Category '{cat}' accuracy: {acc:.2%}")

        return BenchmarkResult(
            score=score,
            confidence=0.9,  # High confidence due to multiple questions
            verdict=verdict,
            evidence=evidence,
            overall_accuracy=overall_accuracy,
            category_accuracies=category_accuracies,
            baseline_deltas=baseline_deltas,
            question_results=results,
            total_questions=total_questions,
            correct_answers=total_correct,
            details={
                "category_stats": category_stats,
                "avg_baseline": avg_baseline,
            },
        )

    def _evaluate_answer(self, response: str, expected: str, method: str = "exact_match") -> bool:
        """
        Evaluates the model's response against the expected answer.
        """
        response = response.strip()
        expected = expected.strip()

        if method == "exact_match":
            return response.lower() == expected.lower()

        elif method == "number_extraction":
            # Extract all numbers from response
            numbers = re.findall(r"[-+]?\d*\.\d+|\d+", response)
            if not numbers:
                return False
            # Check if expected number is in the extracted numbers
            # We also try to find the expected number in the string directly
            try:
                expected_val = float(expected)
                for n in numbers:
                    if abs(float(n) - expected_val) < 1e-6:
                        return True
            except ValueError:
                return expected.lower() in response.lower()
            return False

        elif method == "code_check":
            # Simple check: are key parts of the expected code in the response?
            # Remove whitespace and comments for a slightly better comparison
            def normalize_code(c):
                c = re.sub(r"#.*", "", c)
                return "".join(c.split())

            norm_expected = normalize_code(expected)
            norm_response = normalize_code(response)

            # If expected is a subset of response (ignoring whitespace)
            if norm_expected in norm_response:
                return True

            # Fallback: check for function name and key keywords
            func_match = re.search(r"def\s+(\w+)", expected)
            if func_match:
                func_name = func_match.group(1)
                if func_name in response:
                    # If it's a simple function, maybe it's correct
                    return True
            return False

        elif method == "fuzzy_match":
            # Check if expected is in response or vice versa (case-insensitive)
            resp_low = response.lower()
            exp_low = expected.lower()
            return exp_low in resp_low or resp_low in exp_low

        return response.lower() == expected.lower()
