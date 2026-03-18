"""
Tests for model verification system.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.types import (
    ProviderConfig,
    ModelConfig,
    ModelCharacteristics,
    ProviderType,
    Verdict,
    ProbeType,
    ProbeResult,
    ScoreCard,
    LayerScore,
    TierSignatureResult,
    ComparisonResult,
)
from utils.data_store import DataStore
from analysis.mmd_test import mmd_test, hamming_kernel
from analysis.scoring import ScoringAggregator


class TestTypes:
    def test_verdict_values(self):
        assert Verdict.PASS.value == "PASS"
        assert Verdict.WARN.value == "WARN"
        assert Verdict.FAIL.value == "FAIL"

    def test_probe_type_values(self):
        assert ProbeType.IDENTITY.value == "identity"
        assert ProbeType.FINGERPRINT.value == "fingerprint"
        assert ProbeType.BENCHMARK.value == "benchmark"
        assert ProbeType.LOGPROB.value == "logprob"
        assert ProbeType.LATENCY.value == "latency"
        assert ProbeType.TIER_SIGNATURE.value == "tier_signature"
        assert ProbeType.COMPARISON.value == "comparison"

    def test_provider_config(self):
        config = ProviderConfig(
            name="test",
            type=ProviderType.OPENAI,
            base_url="https://api.test.com/v1",
            api_key_env="TEST_API_KEY",
            models=["test-model"],
        )
        assert config.name == "test"
        assert config.type == ProviderType.OPENAI
        d = config.to_dict()
        assert d["name"] == "test"
        config2 = ProviderConfig.from_dict(d)
        assert config2.name == "test"

    def test_model_config(self):
        chars = ModelCharacteristics(
            family="gpt",
            developer="openai",
            version="4",
            mmlu_baseline=0.88,
        )
        config = ModelConfig(
            name="gpt-4o",
            display_name="GPT-4o",
            characteristics=chars,
        )
        assert config.name == "gpt-4o"
        assert config.characteristics.family == "gpt"

    def test_model_config_with_tier_fields(self):
        chars = ModelCharacteristics(
            family="claude",
            developer="anthropic",
            version="4.6",
            tier="opus",
            hard_math_baseline=0.75,
            hard_code_baseline=0.80,
            hard_reasoning_baseline=0.80,
            expected_response_length="very_long",
        )
        config = ModelConfig(
            name="claude-opus-4-6",
            display_name="Claude Opus 4.6",
            characteristics=chars,
        )
        assert config.characteristics.tier == "opus"
        assert config.characteristics.hard_math_baseline == 0.75
        assert config.characteristics.expected_response_length == "very_long"
        d = config.to_dict()
        assert d["characteristics"]["tier"] == "opus"
        assert d["characteristics"]["hard_math_baseline"] == 0.75

    def test_tier_signature_result(self):
        result = TierSignatureResult(
            score=0.8,
            confidence=0.9,
            verdict=Verdict.PASS,
            evidence=["test"],
            predicted_tier="opus",
            tier_scores={"opus": 0.8, "sonnet": 0.3},
            dimension_scores={"response_depth": 0.9, "reasoning_depth": 0.7},
        )
        assert result.probe_type == ProbeType.TIER_SIGNATURE
        assert result.predicted_tier == "opus"
        d = result.to_dict()
        assert d["probe_type"] == "tier_signature"

    def test_comparison_result(self):
        result = ComparisonResult(
            score=0.85,
            confidence=0.9,
            verdict=Verdict.PASS,
            evidence=["test"],
            reference_provider="anthropic_official",
            reference_model="claude-opus-4-20250514",
            accuracy_delta=0.05,
            response_similarity=0.7,
            latency_ratio=1.2,
        )
        assert result.probe_type == ProbeType.COMPARISON
        assert result.reference_provider == "anthropic_official"
        d = result.to_dict()
        assert d["probe_type"] == "comparison"


class TestMMD:
    def test_hamming_kernel_identical(self):
        result = hamming_kernel("hello", "hello")
        assert result == 5.0

    def test_hamming_kernel_different(self):
        result = hamming_kernel("abc", "xyz")
        assert result == 0.0

    def test_mmd_identical_samples(self):
        samples = ["hello world", "foo bar", "test case"]
        result = mmd_test(samples, samples, n_permutations=100)
        assert result.mmd_statistic == 0.0
        assert result.p_value == 1.0
        assert result.reject_null == False

    def test_mmd_different_samples(self):
        samples_a = ["aaaaa", "bbbbb", "ccccc"]
        samples_b = ["xxxxx", "yyyyy", "zzzzz"]
        result = mmd_test(samples_a, samples_b, n_permutations=100)
        assert result.mmd_statistic > 0


class TestDataStore:
    def test_save_and_load_baseline(self, tmp_path):
        store = DataStore(base_path=tmp_path)
        data = {"test": "value", "score": 0.95}
        store.save_baseline("test-model", "identity", data)
        loaded = store.load_baseline("test-model", "identity")
        assert loaded == data

    def test_list_baselines(self, tmp_path):
        store = DataStore(base_path=tmp_path)
        store.save_baseline("test-model", "identity", {"a": 1})
        store.save_baseline("test-model", "fingerprint", {"b": 2})
        baselines = store.list_baselines("test-model")
        assert set(baselines) == {"identity", "fingerprint"}

    def test_save_result_dict(self, tmp_path):
        store = DataStore(base_path=tmp_path)
        result_data = {
            "provider": "test",
            "model": "test-model",
            "verdict": "PASS",
            "score": 0.95,
        }
        path = store.save_result("test-provider", "test-model", result_data)
        assert path.exists()
        results = store.load_results("test-provider", "test-model")
        assert len(results) == 1
        assert results[0]["provider"] == "test"


class TestScoring:
    def test_scoring_includes_tier_signature_weight(self):
        aggregator = ScoringAggregator()
        assert ProbeType.TIER_SIGNATURE.value in aggregator.weights
        assert ProbeType.COMPARISON.value in aggregator.weights

    def test_scoring_weights_sum_to_one(self):
        aggregator = ScoringAggregator()
        total = sum(aggregator.weights.values())
        assert abs(total - 1.0) < 1e-6


class TestComparisonProbe:
    def test_jaccard_similarity(self):
        from probes.comparison import ComparisonProbe

        probe = ComparisonProbe.__new__(ComparisonProbe)
        assert probe._jaccard_similarity("hello world", "hello world") == 1.0
        assert probe._jaccard_similarity("hello world", "goodbye world") > 0.0
        assert probe._jaccard_similarity("hello world", "goodbye world") < 1.0
        assert probe._jaccard_similarity("", "") == 1.0

    def test_evaluate_number_extraction(self):
        from probes.comparison import ComparisonProbe

        probe = ComparisonProbe.__new__(ComparisonProbe)
        assert probe._evaluate_answer("The answer is 42.", "42", "number_extraction") == True
        assert probe._evaluate_answer("The answer is 43.", "42", "number_extraction") == False

    def test_evaluate_fuzzy_match(self):
        from probes.comparison import ComparisonProbe

        probe = ComparisonProbe.__new__(ComparisonProbe)
        assert probe._evaluate_answer("Paris is the capital", "Paris", "fuzzy_match") == True
        assert probe._evaluate_answer("London is great", "Paris", "fuzzy_match") == False


class TestBenchmarkHardQuestions:
    def test_hard_categories_exist(self):
        from probes.benchmark import BenchmarkProbe

        assert "hard_math" in BenchmarkProbe.BUILTIN_QUESTIONS
        assert "hard_code" in BenchmarkProbe.BUILTIN_QUESTIONS
        assert "hard_reasoning" in BenchmarkProbe.BUILTIN_QUESTIONS
        assert "hard_science" in BenchmarkProbe.BUILTIN_QUESTIONS

    def test_hard_categories_have_five_questions_each(self):
        from probes.benchmark import BenchmarkProbe

        for cat in ["hard_math", "hard_code", "hard_reasoning", "hard_science"]:
            assert len(BenchmarkProbe.BUILTIN_QUESTIONS[cat]) == 5, f"{cat} should have 5 questions"

    def test_hard_questions_have_required_fields(self):
        from probes.benchmark import BenchmarkProbe

        for cat in ["hard_math", "hard_code", "hard_reasoning", "hard_science"]:
            for q in BenchmarkProbe.BUILTIN_QUESTIONS[cat]:
                assert "prompt" in q
                assert "expected_answer" in q
                assert "category" in q
                assert "evaluation_method" in q
                assert q["category"] == cat


class TestConfigLoader:
    def test_load_model_with_tier(self):
        from utils.config_loader import ConfigLoader

        loader = ConfigLoader()
        config = loader.load_model("claude-opus-4-6")
        assert config.characteristics.tier == "opus"
        assert config.characteristics.hard_math_baseline == 0.75


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
