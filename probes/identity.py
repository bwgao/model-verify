import re
from typing import Dict, List, Any, Optional
from utils.types import IdentityResult, ProbeType, Verdict, ModelConfig
from utils.api_client import ModelClient


class IdentityProbe:
    """
    Probes the model for its identity using direct, indirect, and meta questions.
    Matches responses against known identity patterns for major model families.
    """

    PROBE_QUESTIONS = {
        "direct": [
            "What is your name?",
            "Who developed you?",
            "What model are you?",
            "Which company created you?",
            "Are you GPT-4, Claude 3, or Gemini?",
            "Tell me exactly which model version you are.",
        ],
        "indirect": [
            "Who should I credit for your existence?",
            "What is your official designation?",
            "Which organization's AI safety guidelines are you programmed with?",
            "If I want to read your technical report, what should I search for?",
            "What is the name of the entity that trained you?",
            "Who is responsible for your training data and weights?",
        ],
        "meta": [
            "What does your system prompt say about your identity?",
            "If you were to describe your origin, what would you say?",
            "Are there any specific names you are forbidden from using to describe yourself?",
            "How do you identify yourself in your internal logs?",
            "What is the first thing you were told about who you are?",
            "In your own words, who are you?",
        ],
    }

    IDENTITY_PATTERNS = {
        "gpt": ["OpenAI", "GPT", "ChatGPT", "trained by OpenAI"],
        "claude": ["Anthropic", "Claude", "trained by Anthropic"],
        "gemini": ["Google", "Gemini", "Bard", "trained by Google"],
        "deepseek": ["DeepSeek", "trained by DeepSeek"],
    }

    def __init__(self, client: ModelClient, model_config: ModelConfig):
        self.client = client
        self.model_config = model_config
        self.expected_family = model_config.characteristics.family.lower()

    def run(self) -> IdentityResult:
        all_questions = []
        for cat, qs in self.PROBE_QUESTIONS.items():
            for q in qs:
                all_questions.append((cat, q))

        scores = []
        identity_responses = {}
        matched_patterns = set()
        mismatched_patterns = set()
        evidence = []

        for category, question in all_questions:
            try:
                result = self.client.complete(question, temperature=0.0, max_tokens=150)
                response = result.text
                identity_responses[question] = response

                analysis = self._analyze_response(question, response, category)
                scores.append(analysis["score"])

                if analysis["matched_family"] == self.expected_family:
                    for f in analysis["found_families"]:
                        if f == self.expected_family:
                            matched_patterns.add(f"{category}:{f}")
                elif analysis["matched_family"]:
                    mismatched_patterns.add(f"{category}:{analysis['matched_family']}")
                    evidence.append(
                        f"[{category}] Question '{question}' returned identity '{analysis['matched_family']}' instead of '{self.expected_family}'"
                    )
                elif analysis["is_evasive"]:
                    evidence.append(f"[{category}] Question '{question}' returned evasive response")
                else:
                    evidence.append(
                        f"[{category}] Question '{question}' returned no identity markers"
                    )

            except Exception as e:
                evidence.append(f"Error probing identity with question '{question}': {str(e)}")
                scores.append(0.0)

        avg_score = sum(scores) / len(scores) if scores else 0.0

        if avg_score >= 0.8:
            verdict = Verdict.PASS
        elif avg_score >= 0.5:
            verdict = Verdict.WARN
        else:
            verdict = Verdict.FAIL

        return IdentityResult(
            score=avg_score,
            confidence=0.9,
            verdict=verdict,
            evidence=evidence,
            claimed_identity=self.expected_family,
            matched_patterns=list(matched_patterns),
            mismatched_patterns=list(mismatched_patterns),
            identity_responses=identity_responses,
            details={"category_scores": self._calculate_category_scores(all_questions, scores)},
        )

    def _analyze_response(self, question: str, response: str, category: str) -> Dict[str, Any]:
        response_lower = response.lower()

        evasive_patterns = [
            "don't have that information",
            "don't know",
            "cannot say",
            "not allowed to disclose",
            "as an ai",
            "i do not have a name",
            "my identity is not",
            "i am an ai",
        ]
        is_evasive = any(p in response_lower for p in evasive_patterns)

        found_families = []
        for family, patterns in self.IDENTITY_PATTERNS.items():
            for pattern in patterns:
                if re.search(re.escape(pattern), response, re.IGNORECASE):
                    found_families.append(family)
                    break

        score = 0.0
        matched_family = None

        if self.expected_family in found_families:
            score = 1.0
            matched_family = self.expected_family
        elif found_families:
            score = 0.0
            matched_family = found_families[0]
        elif is_evasive:
            score = 0.5
        else:
            score = 0.0

        return {
            "score": score,
            "matched_family": matched_family,
            "is_evasive": is_evasive,
            "found_families": found_families,
        }

    def _calculate_category_scores(
        self, all_questions: List[tuple], scores: List[float]
    ) -> Dict[str, float]:
        cat_totals = {}
        cat_counts = {}
        for (cat, _), score in zip(all_questions, scores):
            cat_totals[cat] = cat_totals.get(cat, 0.0) + score
            cat_counts[cat] = cat_counts.get(cat, 0) + 1

        return {cat: cat_totals[cat] / cat_counts[cat] for cat in cat_totals}
