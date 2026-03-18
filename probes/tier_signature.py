import re
from typing import Dict, List, Any, Optional
from utils.types import TierSignatureResult, ModelConfig, Verdict, ProbeType
from utils.api_client import ModelClient

class TierSignatureProbe:
    """
    A behavioral signature probe that distinguishes higher-tier models (Opus)
    from lower-tier models (Sonnet) within the same family.
    """

    DIMENSION_WEIGHTS = {
        "response_depth": 0.25,
        "reasoning_depth": 0.25,
        "multi_constraint": 0.25,
        "ambiguity": 0.25,
    }

    PROMPTS = {
        "response_depth": [
            "Analyze the long-term geopolitical consequences of artificial intelligence dominance by a single nation.",
            "Discuss the philosophical implications of consciousness in synthetic biological systems.",
            "Evaluate the economic impact of universal basic income in a fully automated post-scarcity society."
        ],
        "reasoning_depth": [
            "A farmer has a fox, a chicken, and a bag of corn. He needs to cross a river in a boat that can only carry him and one item. The fox will eat the chicken if left alone, and the chicken will eat the corn. What is the minimum number of crossings, and list each crossing?",
            "Three missionaries and three cannibals must cross a river using a boat which can carry at most two people. If there are ever more cannibals than missionaries on either side of the river, the cannibals will eat the missionaries. How can they all cross safely? List the steps.",
            "You have two jugs, one holds 5 gallons and the other holds 3 gallons. You have an unlimited water supply. How can you measure exactly 4 gallons of water? List the steps."
        ],
        "multi_constraint": [
            "Write a 4-line poem about the ocean where each line starts with a vowel, contains exactly 5 words per line, uses the word 'blue', and contains no words longer than 8 letters.",
            "Write a 3-sentence story about a clock where every sentence contains exactly 10 words, the word 'time' is never used, and it ends with a question mark.",
            "Write a paragraph about a forest with exactly 5 sentences, where the first letter of each sentence spells 'TREES', containing the word 'green', and no commas."
        ],
        "ambiguity": [
            "Tell me about Mercury.",
            "Explain the significance of the Apple.",
            "What can you tell me about the Amazon?"
        ]
    }

    def __init__(self, client: ModelClient, model_config: ModelConfig):
        self.client = client
        self.model_config = model_config

    def run(self) -> TierSignatureResult:
        scores = {}
        evidence = []
        
        # Dimension 1: Response Depth
        depth_stats = self._test_response_depth()
        scores["response_depth"] = depth_stats["score"]
        evidence.append(f"Response Depth Score: {depth_stats['score']:.2f} (avg chars: {depth_stats['mean_chars']:.1f}, avg paras: {depth_stats['mean_paras']:.1f})")
        
        # Dimension 2: Reasoning Chain Depth
        reasoning_stats = self._test_reasoning_depth()
        scores["reasoning_depth"] = reasoning_stats["score"]
        evidence.append(f"Reasoning Depth Score: {reasoning_stats['score']:.2f} (avg steps: {reasoning_stats['mean_steps']:.1f})")
        
        # Dimension 3: Multi-Constraint Following
        constraint_stats = self._test_multi_constraint()
        scores["multi_constraint"] = constraint_stats["score"]
        evidence.append(f"Multi-Constraint Score: {constraint_stats['score']:.2f}")
        
        # Dimension 4: Ambiguity Handling
        ambiguity_stats = self._test_ambiguity()
        scores["ambiguity"] = ambiguity_stats["score"]
        evidence.append(f"Ambiguity Handling Score: {ambiguity_stats['score']:.2f}")
        
        # Calculate weighted score
        weighted_score = sum(scores[dim] * self.DIMENSION_WEIGHTS[dim] for dim in self.DIMENSION_WEIGHTS)
        
        # Predict tier
        if weighted_score >= 0.7:
            predicted_tier = "opus"
        elif weighted_score >= 0.4:
            predicted_tier = "sonnet"
        else:
            predicted_tier = "haiku"
            
        # Determine verdict
        expected_tier = self.model_config.characteristics.tier
        verdict = Verdict.WARN
        if expected_tier:
            if predicted_tier == expected_tier.lower():
                verdict = Verdict.PASS
            elif weighted_score >= 0.4:
                verdict = Verdict.WARN
            else:
                verdict = Verdict.FAIL
        else:
            if weighted_score >= 0.7:
                verdict = Verdict.PASS
            elif weighted_score >= 0.4:
                verdict = Verdict.WARN
            else:
                verdict = Verdict.FAIL
                
        tier_scores = {
            "opus": max(0.0, (weighted_score - 0.6) / 0.4) if weighted_score >= 0.6 else 0.0,
            "sonnet": max(0.0, 1.0 - abs(weighted_score - 0.55) / 0.25) if 0.3 <= weighted_score <= 0.8 else 0.0,
            "haiku": max(0.0, (0.4 - weighted_score) / 0.4) if weighted_score <= 0.4 else 0.0,
        }
        
        return TierSignatureResult(
            score=weighted_score,
            confidence=0.9,
            verdict=verdict,
            evidence=evidence,
            predicted_tier=predicted_tier,
            tier_scores=tier_scores,
            dimension_scores=scores,
            response_length_stats={
                "mean": depth_stats["mean_chars"],
                "min": depth_stats["min_chars"],
                "max": depth_stats["max_chars"],
            },
            reasoning_depth_stats={
                "mean_steps": reasoning_stats["mean_steps"],
            },
            instruction_following_score=constraint_stats["score"],
        )

    def _test_response_depth(self) -> Dict[str, float]:
        chars = []
        paras = []
        sentences = []
        
        for prompt in self.PROMPTS["response_depth"]:
            try:
                result = self.client.complete(prompt, temperature=0.0, max_tokens=2048)
                text = result.text
                chars.append(len(text))
                paras.append(len([p for p in text.split('\n\n') if p.strip()]))
                sentences.append(len([s for s in re.split(r'[.!?]+', text) if s.strip()]))
            except Exception:
                chars.append(0)
                paras.append(0)
                sentences.append(0)
                
        mean_chars = sum(chars) / len(chars) if chars else 0
        mean_paras = sum(paras) / len(paras) if paras else 0
        mean_sentences = sum(sentences) / len(sentences) if sentences else 0
        
        expected_length = self.model_config.characteristics.expected_response_length
        target_chars = 1500
        target_paras = 4
        
        if expected_length == "short":
            target_chars = 500
            target_paras = 2
        elif expected_length == "medium":
            target_chars = 1000
            target_paras = 3
        elif expected_length == "long":
            target_chars = 1500
            target_paras = 4
        elif expected_length == "very_long":
            target_chars = 2000
            target_paras = 5
            
        char_score = min(1.0, mean_chars / target_chars) if target_chars > 0 else 0.0
        para_score = min(1.0, mean_paras / target_paras) if target_paras > 0 else 0.0
        score = (char_score + para_score) / 2.0
        
        return {
            "score": score,
            "mean_chars": mean_chars,
            "min_chars": min(chars) if chars else 0,
            "max_chars": max(chars) if chars else 0,
            "mean_paras": mean_paras,
            "mean_sentences": mean_sentences,
        }

    def _test_reasoning_depth(self) -> Dict[str, float]:
        steps_counts = []
        
        for prompt in self.PROMPTS["reasoning_depth"]:
            try:
                result = self.client.complete(prompt, temperature=0.0, max_tokens=2048)
                text = result.text
                
                numbered_steps = len(re.findall(r'(?m)^\s*(?:\d+[\.\)]|Step\s+\d+)', text))
                transitions = len(re.findall(r'(?i)\b(first|second|third|then|next|finally|therefore|thus|hence)\b', text))
                
                steps_counts.append(numbered_steps + transitions)
            except Exception:
                steps_counts.append(0)
                
        mean_steps = sum(steps_counts) / len(steps_counts) if steps_counts else 0
        score = min(1.0, mean_steps / 5.0)
        
        return {
            "score": score,
            "mean_steps": mean_steps,
        }

    def _test_multi_constraint(self) -> Dict[str, float]:
        scores = []
        
        # Prompt 1
        try:
            result = self.client.complete(self.PROMPTS["multi_constraint"][0], temperature=0.0, max_tokens=1024)
            text = result.text
            lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
            constraints_met = 0
            
            if len(lines) == 4:
                constraints_met += 1
            if all(line[0].lower() in 'aeiou' for line in lines if line):
                constraints_met += 1
            if all(len(re.findall(r'\b\w+\b', line)) == 5 for line in lines):
                constraints_met += 1
            if re.search(r'\bblue\b', text, re.IGNORECASE):
                constraints_met += 1
            words = re.findall(r'\b\w+\b', text)
            if all(len(word) <= 8 for word in words):
                constraints_met += 1
                
            scores.append(constraints_met / 5.0)
        except Exception:
            scores.append(0.0)
            
        # Prompt 2
        try:
            result = self.client.complete(self.PROMPTS["multi_constraint"][1], temperature=0.0, max_tokens=1024)
            text = result.text
            sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
            constraints_met = 0
            
            if len(sentences) == 3:
                constraints_met += 1
            if all(len(re.findall(r'\b\w+\b', s)) == 10 for s in sentences):
                constraints_met += 1
            if not re.search(r'\btime\b', text, re.IGNORECASE):
                constraints_met += 1
            if text.strip().endswith('?'):
                constraints_met += 1
                
            scores.append(constraints_met / 4.0)
        except Exception:
            scores.append(0.0)
            
        # Prompt 3
        try:
            result = self.client.complete(self.PROMPTS["multi_constraint"][2], temperature=0.0, max_tokens=1024)
            text = result.text
            sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
            constraints_met = 0
            
            if len(sentences) == 5:
                constraints_met += 1
            first_letters = ''.join([s[0].upper() for s in sentences if s])
            if first_letters == 'TREES':
                constraints_met += 1
            if re.search(r'\bgreen\b', text, re.IGNORECASE):
                constraints_met += 1
            if ',' not in text:
                constraints_met += 1
                
            scores.append(constraints_met / 4.0)
        except Exception:
            scores.append(0.0)
            
        mean_score = sum(scores) / len(scores) if scores else 0.0
        return {"score": mean_score}

    def _test_ambiguity(self) -> Dict[str, float]:
        scores = []
        patterns = [
            r"could refer to",
            r"in the context of",
            r"there are multiple",
            r"depending on",
            r"can refer to",
            r"several meanings",
            r"which one",
            r"multiple meanings",
            r"different things",
            r"can mean",
            r"may refer to"
        ]
        
        for prompt in self.PROMPTS["ambiguity"]:
            try:
                result = self.client.complete(prompt, temperature=0.0, max_tokens=1024)
                text = result.text
                
                if any(re.search(p, text, re.IGNORECASE) for p in patterns):
                    scores.append(1.0)
                else:
                    scores.append(0.0)
            except Exception:
                scores.append(0.0)
                
        mean_score = sum(scores) / len(scores) if scores else 0.0
        return {"score": mean_score}
