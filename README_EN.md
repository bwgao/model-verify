# Model Verify

LLM Model Verification System - Detects whether third-party API resellers provide authentic models.

## Background

Third-party LLM API resellers may claim to provide models like Claude and GPT-4 at lower prices, but they might actually:
- Use smaller models to impersonate larger ones
- Use quantized versions to impersonate full-precision models
- Completely replace with other models

This project uses multi-dimensional verification methods to help you detect these deceptive practices.

## Verification Layers

### Layer 1: Identity Probing
Directly ask the model about its identity and match against known patterns.

### Layer 2: Behavioral Fingerprinting
LLMmap-style active fingerprinting that analyzes response characteristics through topic-based queries.

### Layer 3: Capability Benchmarking
Test capabilities such as mathematical reasoning, code generation, and knowledge Q&A, comparing against known baselines. Includes 40 questions, with 20 high-difficulty questions for distinguishing model tiers.

### Layer 4: Logprob Analysis
Compare token probability distributions (when API supports it) and calculate KL/JS divergence.

### Layer 5: Latency Fingerprinting
Measure metrics like TTFT, TPS, etc., and compare against expected performance.

### Layer 6: Tier Signature
Distinguish between different tiers of models within the same family (e.g., Opus vs Sonnet vs Haiku) through behavioral characteristic analysis. Includes 4 dimensions:
- **Detail Level**: Depth and completeness of responses
- **Code Style**: Code comments, structure, naming conventions
- **Reasoning Depth**: Complexity of logical chains
- **Refusal Patterns**: Handling of sensitive questions

### Layer 7: Model Comparison
A/B blind testing to directly compare response differences between two models. Suitable for scenarios where an official model is available as a reference.

## Installation

```bash
cd model_verify
source venv/bin/activate
pip install -r requirements.txt
```

## Configuration

1. Copy and edit provider configuration:
```bash
cp config/providers.yaml config/providers.local.yaml
```

2. Set API key environment variables:
```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

## Usage

### Verify Model
```bash
# Full verification (all probes)
python main.py verify test_reseller claude-opus-4-6

# Verify with specific probes
python main.py verify test_reseller claude-opus-4-6 --probes identity,fingerprint,benchmark

# Tier signature test (distinguish Opus/Sonnet/Haiku)
python main.py verify test_reseller claude-opus-4-6 --probes tier_signature
```

### Model Comparison
```bash
# A/B blind testing comparison between two models
python main.py compare test_reseller claude-opus-4-6 official_provider claude-opus-4-6
```

### Collect Baseline
```bash
python main.py baseline gpt-4o --provider openai_official
```

### View Report
```bash
python main.py report example_reseller
python main.py report example_reseller claude-opus-4-20250514 --format json
```

### Continuous Monitoring
```bash
python main.py monitor example_reseller gpt-4o --interval 300
```

## Project Structure

```
model_verify/
├── config/              # YAML configuration
│   ├── providers.yaml   # API provider configuration
│   └── models.yaml      # Model feature configuration
├── baselines/           # Baseline data storage
├── results/             # Verification results storage
├── probes/              # Verification probe modules
│   ├── identity.py      # Identity probing
│   ├── fingerprint.py   # Behavioral fingerprinting
│   ├── benchmark.py     # Capability benchmarking (40 questions including 20 high-difficulty)
│   ├── logprob.py       # Logprob analysis
│   ├── latency.py       # Latency fingerprinting
│   ├── tier_signature.py # Tier signature probing
│   └── comparison.py    # A/B comparison probing
├── analysis/            # Analysis modules
│   ├── mmd_test.py      # MMD statistical testing
│   ├── scoring.py       # Scoring aggregation
│   └── report.py        # Report generation
├── utils/               # Utility modules
│   ├── types.py         # Type definitions
│   ├── config_loader.py # Configuration loading
│   ├── data_store.py    # Data storage
│   └── api_client.py    # API client
└── main.py              # CLI entry point
```

## Tier Distinction Methods

This system supports distinguishing between different tiers of models within the same family (e.g., Claude Opus 4.6 vs Sonnet 4.6 vs Haiku):

### 1. Hard Benchmark Testing
20 high-difficulty questions covering:
- Advanced Mathematics (5 questions)
- Complex Code (5 questions)
- Deep Reasoning (5 questions)
- Scientific Questions (5 questions)

### 2. Behavioral Signatures
4-dimensional feature analysis:
- **Detail Level**: Opus provides more detailed responses, Haiku is more concise
- **Code Style**: Opus has more comments and clearer structure
- **Reasoning Depth**: Opus has longer logical chains and more comprehensive consideration
- **Refusal Patterns**: Different tiers handle sensitive questions differently

### 3. A/B Comparison
Blind testing comparison with official models to evaluate response similarity.

## Supported Models

| Model Family | Tier | Benchmark Score |
|--------------|------|-----------------|
| Claude 4.6 | Opus | Hard Benchmark > 70% |
| Claude 4.6 | Sonnet | Hard Benchmark 40-70% |
| Claude 4.6 | Haiku | Hard Benchmark < 40% |

## Scoring

| Score | Verdict | Meaning |
|-------|---------|---------|
| ≥0.8 | PASS | Model is authentic and trustworthy |
| 0.5-0.8 | WARN | Suspicious signs detected |
| <0.5 | FAIL | Model may be replaced |

## References

- *LLMmap: Fingerprinting for Large Language Models* (USENIX Security 2025)
- *Model Equality Testing: Which Model Is This API Serving?* (Stanford 2024)
- *Are You Getting What You Pay For? Auditing Model Substitution in LLM APIs* (UC Berkeley 2025)
- *Real Money, Fake Models: Deceptive Model Claims in Shadow APIs* (CISPA 2026)
