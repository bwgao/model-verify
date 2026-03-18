import json
import os
from datetime import datetime
from typing import Optional, Any, List, Dict

from rich.console import Console
from rich.text import Text
from rich.theme import Theme

from utils.types import VerificationReport, Verdict, ProbeType, ScoreCard


class ReportGenerator:
    def __init__(self, output_dir: str = "results"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.theme = Theme(
            {
                "pass": "bold green",
                "warn": "bold yellow",
                "fail": "bold red",
                "info": "cyan",
                "header": "bold blue",
                "border": "blue",
            }
        )
        self.console = Console(theme=self.theme, force_terminal=True, width=80)

    def _format_verdict(self, verdict: Verdict) -> str:
        if verdict == Verdict.PASS:
            return "\033[1;92mPASS\033[0m"
        elif verdict == Verdict.WARN:
            return "\033[1;93mWARN\033[0m"
        elif verdict == Verdict.FAIL:
            return "\033[1;91mFAIL\033[0m"
        return str(verdict)

    def _get_progress_bar(self, score: float) -> str:
        filled = max(0, min(10, int(round(score * 10))))
        bar = "█" * filled + "░" * (10 - filled)

        if score >= 0.8:
            color = "green"
        elif score >= 0.5:
            color = "yellow"
        else:
            color = "red"

        return f"[{color}]{bar}[/{color}]"

    def _generate_executive_summary(self, report: VerificationReport) -> str:
        sc = report.scorecard
        if sc.overall_verdict == Verdict.PASS:
            return f"The model from {sc.provider} appears to be genuine {sc.model}. All verification layers passed with high confidence ({sc.confidence_level:.2f})."
        elif sc.overall_verdict == Verdict.WARN:
            return f"Verification of {sc.model} from {sc.provider} raised some concerns. While it mostly behaves as expected (Score: {sc.aggregate_score:.2f}), there are anomalies in some layers that warrant caution."
        else:
            return f"CRITICAL: The model from {sc.provider} failed verification. It does not match the characteristics of {sc.model}. Aggregate score: {sc.aggregate_score:.2f}."

    def _generate_recommendations(self, report: VerificationReport) -> List[str]:
        recs = []
        sc = report.scorecard

        for p_type_val, ls in sc.layer_scores.items():
            if ls.verdict != Verdict.PASS:
                if p_type_val == ProbeType.IDENTITY.value:
                    recs.append(
                        "Verify model identity via direct system prompt probes and check for 'lazy' identity responses."
                    )
                elif p_type_val == ProbeType.FINGERPRINT.value:
                    recs.append(
                        "Run more extensive behavioral fingerprinting to confirm model family and version."
                    )
                elif p_type_val == ProbeType.BENCHMARK.value:
                    recs.append(
                        "Check for capability degradation or potential model quantization/distillation."
                    )
                elif p_type_val == ProbeType.LOGPROB.value:
                    recs.append(
                        "Analyze logprob distributions for signs of output filtering, proxying, or 'wrapper' behavior."
                    )
                elif p_type_val == ProbeType.LATENCY.value:
                    recs.append(
                        "Investigate latency anomalies which may indicate request routing, caching, or non-standard hardware."
                    )
                elif p_type_val == ProbeType.TIER_SIGNATURE.value:
                    recs.append(
                        "Model tier prediction mismatch. Run comparison against official API to confirm model tier (opus vs sonnet)."
                    )
                elif p_type_val == ProbeType.COMPARISON.value:
                    recs.append(
                        "A/B comparison shows differences from reference model. Verify with additional prompts or different reference provider."
                    )

        if not recs:
            if sc.overall_verdict == Verdict.PASS:
                recs.append(
                    "Continue monitoring model performance for consistency across different regions."
                )
            else:
                recs.append(
                    "Perform a manual review of the probe results to identify subtle discrepancies."
                )

        return recs

    def generate_text(self, report: VerificationReport) -> str:
        sc = report.scorecard

        if not report.executive_summary:
            report.executive_summary = self._generate_executive_summary(report)
        if not report.recommendations:
            report.recommendations = self._generate_recommendations(report)

        capture_console = Console(
            theme=self.theme, force_terminal=True, width=70, color_system="truecolor"
        )

        with capture_console.capture() as capture:
            capture_console.print(
                "═══════════════════════════════════════════════════════════════", style="border"
            )
            capture_console.print(
                "                    MODEL VERIFICATION REPORT", style="bold white"
            )
            capture_console.print(
                "═══════════════════════════════════════════════════════════════", style="border"
            )
            capture_console.print()
            capture_console.print(f"Provider: [info]{sc.provider}[/info]")
            capture_console.print(f"Model: [info]{sc.model}[/info]")
            capture_console.print(f"Claimed Model: [info]{sc.claimed_model or 'N/A'}[/info]")
            capture_console.print(f"Timestamp: {sc.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            capture_console.print()
            capture_console.print(
                "───────────────────────────────────────────────────────────────", style="border"
            )

            v_color = (
                "pass"
                if sc.overall_verdict == Verdict.PASS
                else "warn"
                if sc.overall_verdict == Verdict.WARN
                else "fail"
            )
            verdict_text = Text(sc.overall_verdict.value, style=v_color)
            capture_console.print(
                Text.assemble(
                    "OVERALL VERDICT: ",
                    verdict_text,
                    f" (Score: {sc.aggregate_score:.2f}, Confidence: {sc.confidence_level:.2f})",
                )
            )
            capture_console.print(
                "───────────────────────────────────────────────────────────────", style="border"
            )
            capture_console.print()

            capture_console.print("LAYER SCORES:")
            layers = [
                ("Identity", ProbeType.IDENTITY),
                ("Fingerprint", ProbeType.FINGERPRINT),
                ("Benchmark", ProbeType.BENCHMARK),
                ("Logprob", ProbeType.LOGPROB),
                ("Latency", ProbeType.LATENCY),
                ("Tier Signature", ProbeType.TIER_SIGNATURE),
                ("Comparison", ProbeType.COMPARISON),
            ]

            for label, p_type in layers:
                ls = sc.layer_scores.get(p_type.value)
                if ls:
                    bar = self._get_progress_bar(ls.score)
                    ls_v_color = (
                        "pass"
                        if ls.verdict == Verdict.PASS
                        else "warn"
                        if ls.verdict == Verdict.WARN
                        else "fail"
                    )
                    ls_v_text = f"[{ls_v_color}]{ls.verdict.value}[/{ls_v_color}]"
                    capture_console.print(
                        Text.from_markup(f"  {label:<13} {bar} {ls.score:.2f} ({ls_v_text})")
                    )
                else:
                    capture_console.print(f"  {label:<13} {'░' * 10} 0.00 (N/A)")

            capture_console.print()
            capture_console.print("EVIDENCE SUMMARY:")
            evidence = []
            results = [
                report.identity_result,
                report.fingerprint_result,
                report.benchmark_result,
                report.logprob_result,
                report.latency_result,
                report.tier_signature_result,
                report.comparison_result,
            ]
            for res in results:
                if res and res.evidence:
                    evidence.extend(res.evidence)

            if not evidence:
                capture_console.print("- No specific evidence recorded.")
            else:
                for item in evidence[:15]:
                    capture_console.print(f"- {item}")
                if len(evidence) > 15:
                    capture_console.print(f"- ... and {len(evidence) - 15} more items")

            capture_console.print()
            capture_console.print("RECOMMENDATIONS:")
            if not report.recommendations:
                capture_console.print("- No specific recommendations.")
            else:
                for rec in report.recommendations:
                    capture_console.print(f"- {rec}")

        return capture.get()

    def generate_json(self, report: VerificationReport) -> Dict[str, Any]:
        return report.to_dict()

    def generate_full(self, report: VerificationReport, output_path: Optional[str] = None) -> str:
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_provider = report.scorecard.provider.replace(" ", "_").lower()
            safe_model = report.scorecard.model.replace("/", "_").lower()
            filename = f"report_{safe_provider}_{safe_model}_{timestamp}.txt"
            output_path = os.path.join(self.output_dir, filename)

        text_content = self.generate_text(report)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text_content)

        json_path = os.path.splitext(output_path)[0] + ".json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.generate_json(report), f, indent=2)

        return output_path
