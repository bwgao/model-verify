#!/usr/bin/env python3
"""
Model Verification CLI

Verify LLM API providers are serving genuine models.
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich import print as rprint

from utils.config_loader import ConfigLoader
from utils.data_store import DataStore
from utils.api_client import ModelClient
from utils.types import Verdict, ProbeType
from probes.identity import IdentityProbe
from probes.fingerprint import FingerprintProbe
from probes.benchmark import BenchmarkProbe
from probes.logprob import LogprobProbe
from probes.latency import LatencyProbe
from probes.tier_signature import TierSignatureProbe
from probes.comparison import ComparisonProbe

console = Console()


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Model Verification System - Detect fake model substitution in API resellers."""
    pass


@cli.command()
@click.argument("provider")
@click.argument("model")
@click.option(
    "--layers",
    "-l",
    multiple=True,
    help="Specific probe layers to run (identity, fingerprint, benchmark, logprob, latency)",
)
@click.option("--config", "-c", type=click.Path(exists=True), help="Path to config directory")
@click.option(
    "--output", "-o", type=click.Path(), default="results", help="Output directory for results"
)
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def verify(
    provider: str, model: str, layers: tuple, config: Optional[str], output: str, verbose: bool
):
    config_path = str(config) if config else None
    loader = ConfigLoader(config_path)
    store = DataStore(base_path=Path(output))

    # Load configs
    try:
        provider_config = loader.load_provider(provider)
        model_config = loader.load_model(model)
    except Exception as e:
        console.print(f"[red]Error loading config: {e}[/red]")
        sys.exit(1)

    # Create client
    try:
        client = ModelClient(provider_config, model)
    except Exception as e:
        console.print(f"[red]Error creating client: {e}[/red]")
        sys.exit(1)

    # Determine which probes to run
    all_layers = {"identity", "fingerprint", "benchmark", "logprob", "latency", "tier_signature"}
    selected_layers = set(layers) if layers else all_layers

    results = {}

    console.print(
        Panel(
            f"[bold]Model Verification[/bold]\nProvider: {provider}\nModel: {model}", expand=False
        )
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Run probes
        if "identity" in selected_layers:
            task_id = progress.add_task("Running identity probe...", total=None)
            try:
                probe = IdentityProbe(client, model_config)
                results["identity"] = probe.run()
                progress.update(
                    task_id,
                    description=f"[green]✓[/green] Identity: {results['identity'].verdict.value}",
                )
            except Exception as e:
                progress.update(task_id, description=f"[red]✗ Identity: {e}[/red]")

        if "fingerprint" in selected_layers:
            task_id = progress.add_task("Running fingerprint probe...", total=None)
            try:
                probe = FingerprintProbe(client, model_config, store)
                results["fingerprint"] = probe.run()
                progress.update(
                    task_id,
                    description=f"[green]✓[/green] Fingerprint: {results['fingerprint'].verdict.value}",
                )
            except Exception as e:
                progress.update(task_id, description=f"[red]✗ Fingerprint: {e}[/red]")

        if "benchmark" in selected_layers:
            task_id = progress.add_task("Running benchmark probe...", total=None)
            try:
                probe = BenchmarkProbe(client, model_config)
                results["benchmark"] = probe.run()
                progress.update(
                    task_id,
                    description=f"[green]✓[/green] Benchmark: {results['benchmark'].verdict.value}",
                )
            except Exception as e:
                progress.update(task_id, description=f"[red]✗ Benchmark: {e}[/red]")

        if "logprob" in selected_layers:
            task_id = progress.add_task("Running logprob probe...", total=None)
            try:
                probe = LogprobProbe(client, model_config, store)
                results["logprob"] = probe.run()
                progress.update(
                    task_id,
                    description=f"[green]✓[/green] Logprob: {results['logprob'].verdict.value}",
                )
            except Exception as e:
                progress.update(task_id, description=f"[red]✗ Logprob: {e}[/red]")

        if "latency" in selected_layers:
            task_id = progress.add_task("Running latency probe...", total=None)
            try:
                probe = LatencyProbe(client, model_config)
                results["latency"] = probe.run()
                progress.update(
                    task_id,
                    description=f"[green]✓[/green] Latency: {results['latency'].verdict.value}",
                )
            except Exception as e:
                progress.update(task_id, description=f"[red]✗ Latency: {e}[/red]")

        if "tier_signature" in selected_layers:
            task_id = progress.add_task("Running tier signature probe...", total=None)
            try:
                probe = TierSignatureProbe(client, model_config)
                results["tier_signature"] = probe.run()
                tier_result = results["tier_signature"]
                progress.update(
                    task_id,
                    description=f"[green]✓[/green] Tier Signature: {tier_result.verdict.value} (predicted: {tier_result.predicted_tier})",
                )
            except Exception as e:
                progress.update(task_id, description=f"[red]✗ Tier Signature: {e}[/red]")

    # Calculate aggregate score
    if results:
        scores = [r.score for r in results.values()]
        confidences = [r.confidence for r in results.values()]
        avg_score = sum(scores) / len(scores)
        avg_confidence = sum(confidences) / len(confidences)

        if avg_score >= 0.8:
            overall_verdict = Verdict.PASS
            verdict_color = "green"
        elif avg_score >= 0.5:
            overall_verdict = Verdict.WARN
            verdict_color = "yellow"
        else:
            overall_verdict = Verdict.FAIL
            verdict_color = "red"
    else:
        avg_score = 0
        avg_confidence = 0
        overall_verdict = Verdict.FAIL
        verdict_color = "red"

    # Display results table
    table = Table(title="Verification Results")
    table.add_column("Layer", style="cyan")
    table.add_column("Score", justify="right")
    table.add_column("Confidence", justify="right")
    table.add_column("Verdict", justify="center")

    for layer_name, result in results.items():
        v_color = (
            "green"
            if result.verdict == Verdict.PASS
            else "yellow"
            if result.verdict == Verdict.WARN
            else "red"
        )
        table.add_row(
            layer_name.capitalize(),
            f"{result.score:.2f}",
            f"{result.confidence:.2f}",
            f"[{v_color}]{result.verdict.value}[/{v_color}]",
        )

    console.print(table)

    # Display overall verdict
    console.print()
    console.print(
        Panel(
            f"[bold]Overall Verdict: [{verdict_color}]{overall_verdict.value}[/{verdict_color}][/bold]\n"
            f"Aggregate Score: {avg_score:.2f}\n"
            f"Confidence: {avg_confidence:.2f}",
            expand=False,
        )
    )

    # Save results
    result_data = {
        "provider": provider,
        "model": model,
        "timestamp": datetime.now().isoformat(),
        "overall_verdict": overall_verdict.value,
        "aggregate_score": avg_score,
        "confidence": avg_confidence,
        "layers": {k: v.to_dict() for k, v in results.items()},
    }
    store.save_result(provider, model, result_data)

    # Exit code based on verdict
    if overall_verdict == Verdict.PASS:
        sys.exit(0)
    elif overall_verdict == Verdict.WARN:
        sys.exit(1)
    else:
        sys.exit(2)


@cli.command()
@click.argument("model")
@click.option(
    "--provider", "-p", default="openai_official", help="Official provider to use for baseline"
)
@click.option("--config", "-c", type=click.Path(exists=True), help="Path to config directory")
def baseline(model: str, provider: str, config: Optional[str]):
    config_path = str(config) if config else None
    loader = ConfigLoader(config_path)
    store = DataStore(base_path=Path("baselines"))

    try:
        provider_config = loader.load_provider(provider)
        model_config = loader.load_model(model)
    except Exception as e:
        console.print(f"[red]Error loading config: {e}[/red]")
        sys.exit(1)

    try:
        client = ModelClient(provider_config, model)
    except Exception as e:
        console.print(f"[red]Error creating client: {e}[/red]")
        sys.exit(1)

    console.print(f"[cyan]Collecting baseline for {model} from {provider}...[/cyan]")

    # Run identity probe for baseline
    console.print("  Running identity probe...")
    identity_probe = IdentityProbe(client, model_config)
    identity_result = identity_probe.run()
    store.save_baseline(model, "identity", identity_result.to_dict())
    console.print(f"    [green]✓[/green] Identity baseline saved")

    # Run benchmark probe for baseline
    console.print("  Running benchmark probe...")
    benchmark_probe = BenchmarkProbe(client, model_config)
    benchmark_result = benchmark_probe.run()
    store.save_baseline(model, "benchmark", benchmark_result.to_dict())
    console.print(f"    [green]✓[/green] Benchmark baseline saved")

    # Run latency probe for baseline
    console.print("  Running latency probe...")
    latency_probe = LatencyProbe(client, model_config)
    latency_result = latency_probe.run()
    store.save_baseline(model, "latency", latency_result.to_dict())
    console.print(f"    [green]✓[/green] Latency baseline saved")

    console.print(f"\n[green]Baseline collection complete for {model}[/green]")


@cli.command("report")
@click.argument("provider")
@click.argument("model", required=False)
@click.option(
    "--format", "-f", type=click.Choice(["text", "json"]), default="text", help="Output format"
)
@click.option("--latest", "-l", is_flag=True, help="Show latest result only")
def report_cmd(provider: str, model: Optional[str], format: str, latest: bool):
    """Show verification report for a provider/model.

    PROVIDER: Provider name
    MODEL: Model name (optional, shows all models if not specified)
    """
    store = DataStore(base_path=Path("results"))

    if model:
        results = store.load_results(provider, model, limit=1 if latest else 10)
        if not results:
            console.print(f"[yellow]No results found for {provider}/{model}[/yellow]")
            return

        if format == "json":
            rprint(json.dumps(results[0] if latest else results, indent=2))
        else:
            for result in results:
                _display_text_report(result)
    else:
        # List all models for provider
        results_dir = Path("results") / provider
        if not results_dir.exists():
            console.print(f"[yellow]No results found for provider: {provider}[/yellow]")
            return

        models = [d.name for d in results_dir.iterdir() if d.is_dir()]
        if not models:
            console.print(f"[yellow]No results found for provider: {provider}[/yellow]")
            return

        table = Table(title=f"Results for {provider}")
        table.add_column("Model")
        table.add_column("Latest Verdict")
        table.add_column("Score")
        table.add_column("Timestamp")

        for model_name in sorted(models):
            results = store.load_results(provider, model_name, limit=1)
            if results:
                r = results[0]
                v_color = (
                    "green"
                    if r.get("overall_verdict") == "PASS"
                    else "yellow"
                    if r.get("overall_verdict") == "WARN"
                    else "red"
                )
                table.add_row(
                    model_name,
                    f"[{v_color}]{r.get('overall_verdict', 'N/A')}[/{v_color}]",
                    f"{r.get('aggregate_score', 0):.2f}",
                    r.get("timestamp", "N/A")[:19],
                )

        console.print(table)


def _display_text_report(result: dict):
    """Display a text report for a single result."""
    verdict = result.get("overall_verdict", "UNKNOWN")
    score = result.get("aggregate_score", 0)
    confidence = result.get("confidence", 0)

    v_color = "green" if verdict == "PASS" else "yellow" if verdict == "WARN" else "red"

    console.print(
        Panel(
            f"[bold]Verification Report[/bold]\n"
            f"Provider: {result.get('provider', 'N/A')}\n"
            f"Model: {result.get('model', 'N/A')}\n"
            f"Timestamp: {result.get('timestamp', 'N/A')[:19]}\n\n"
            f"[bold]Overall: [{v_color}]{verdict}[/{v_color}][/bold]\n"
            f"Score: {score:.2f} | Confidence: {confidence:.2f}",
            expand=False,
        )
    )

    layers = result.get("layers", {})
    if layers:
        table = Table()
        table.add_column("Layer")
        table.add_column("Score")
        table.add_column("Verdict")

        for layer_name, layer_data in layers.items():
            lv = layer_data.get("verdict", "UNKNOWN")
            lv_color = "green" if lv == "PASS" else "yellow" if lv == "WARN" else "red"
            table.add_row(
                layer_name.capitalize(),
                f"{layer_data.get('score', 0):.2f}",
                f"[{lv_color}]{lv}[/{lv_color}]",
            )

        console.print(table)


@cli.command()
@click.argument("provider")
@click.argument("model")
@click.option("--interval", "-i", default=300, help="Check interval in seconds (default: 300)")
@click.option("--config", "-c", type=click.Path(exists=True), help="Path to config directory")
def monitor(provider: str, model: str, interval: int, config: Optional[str]):
    """Continuously monitor a provider/model.

    PROVIDER: Provider name
    MODEL: Model name
    """
    console.print(f"[cyan]Starting monitor for {provider}/{model}[/cyan]")
    console.print(f"[dim]Checking every {interval} seconds. Press Ctrl+C to stop.[/dim]")

    config_path = str(config) if config else None
    loader = ConfigLoader(config_path)

    try:
        provider_config = loader.load_provider(provider)
        model_config = loader.load_model(model)
    except Exception as e:
        console.print(f"[red]Error loading config: {e}[/red]")
        sys.exit(1)

    last_verdict = None

    try:
        while True:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            console.print(f"\n[dim]{timestamp}[/dim] Running verification...")

            try:
                client = ModelClient(provider_config, model)

                # Run quick verification (identity only for speed)
                probe = IdentityProbe(client, model_config)
                result = probe.run()

                verdict = result.verdict
                v_color = (
                    "green"
                    if verdict == Verdict.PASS
                    else "yellow"
                    if verdict == Verdict.WARN
                    else "red"
                )

                console.print(
                    f"  Verdict: [{v_color}]{verdict.value}[/{v_color}] | Score: {result.score:.2f}"
                )

                # Alert on verdict change
                if last_verdict and last_verdict != verdict:
                    console.print(
                        f"  [bold red]⚠ VERDICT CHANGED: {last_verdict.value} → {verdict.value}[/bold red]"
                    )

                last_verdict = verdict

            except Exception as e:
                console.print(f"  [red]Error: {e}[/red]")

            time.sleep(interval)

    except KeyboardInterrupt:
        console.print("\n[yellow]Monitor stopped.[/yellow]")


@cli.command()
@click.argument("test_provider")
@click.argument("ref_provider")
@click.argument("model")
@click.option("--config", "-c", type=click.Path(exists=True), help="Path to config directory")
@click.option(
    "--output", "-o", type=click.Path(), default="results", help="Output directory for results"
)
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def compare(
    test_provider: str,
    ref_provider: str,
    model: str,
    config: Optional[str],
    output: str,
    verbose: bool,
):
    """Compare a test provider's model against an official reference.

    TEST_PROVIDER: Provider to test (e.g. test_reseller)
    REF_PROVIDER: Official reference provider (e.g. anthropic_official)
    MODEL: Model name to compare (e.g. claude-opus-4-20250514)
    """
    config_path = str(config) if config else None
    loader = ConfigLoader(config_path)
    store = DataStore(base_path=Path(output))

    try:
        test_provider_config = loader.load_provider(test_provider)
        ref_provider_config = loader.load_provider(ref_provider)
        model_config = loader.load_model(model)
    except Exception as e:
        console.print(f"[red]Error loading config: {e}[/red]")
        sys.exit(1)

    try:
        test_client = ModelClient(test_provider_config, model)
        ref_client = ModelClient(ref_provider_config, model)
    except Exception as e:
        console.print(f"[red]Error creating clients: {e}[/red]")
        sys.exit(1)

    console.print(
        Panel(
            f"[bold]Model Comparison[/bold]\n"
            f"Test: {test_provider} → {model}\n"
            f"Reference: {ref_provider} → {model}",
            expand=False,
        )
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task_id = progress.add_task("Running A/B comparison...", total=None)
        try:
            probe = ComparisonProbe(test_client, ref_client, model_config)
            result = probe.run()
            progress.update(
                task_id,
                description=f"[green]✓[/green] Comparison: {result.verdict.value}",
            )
        except Exception as e:
            console.print(f"[red]Error running comparison: {e}[/red]")
            sys.exit(1)

    v_color = (
        "green"
        if result.verdict == Verdict.PASS
        else "yellow"
        if result.verdict == Verdict.WARN
        else "red"
    )

    table = Table(title="Comparison Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Overall Score", f"{result.score:.2f}")
    table.add_row("Verdict", f"[{v_color}]{result.verdict.value}[/{v_color}]")
    table.add_row("Accuracy Delta", f"{result.accuracy_delta:+.2f}")
    table.add_row("Response Similarity", f"{result.response_similarity:.2f}")
    table.add_row("Latency Ratio (test/ref)", f"{result.latency_ratio:.2f}")
    table.add_row("Hard Accuracy (test)", f"{result.hard_accuracy_test:.2%}")
    table.add_row("Hard Accuracy (ref)", f"{result.hard_accuracy_reference:.2%}")

    if result.mmd_result:
        mmd_color = "red" if result.mmd_result.reject_null else "green"
        table.add_row(
            "MMD Test",
            f"[{mmd_color}]{'DIFFER' if result.mmd_result.reject_null else 'SAME'}[/{mmd_color}] (p={result.mmd_result.p_value:.4f})",
        )

    console.print(table)

    if result.evidence:
        console.print("\n[bold]Evidence:[/bold]")
        for e in result.evidence:
            console.print(f"  • {e}")

    if verbose and result.per_prompt_comparison:
        console.print("\n[bold]Per-Prompt Details:[/bold]")
        detail_table = Table()
        detail_table.add_column("Prompt", max_width=40)
        detail_table.add_column("Category")
        detail_table.add_column("Test ✓")
        detail_table.add_column("Ref ✓")
        detail_table.add_column("Similarity")

        for item in result.per_prompt_comparison:
            detail_table.add_row(
                item["prompt"][:40] + "...",
                item["category"],
                "[green]✓[/green]" if item.get("test_correct") else "[red]✗[/red]",
                "[green]✓[/green]" if item.get("ref_correct") else "[red]✗[/red]",
                f"{item.get('similarity', 0):.2f}",
            )
        console.print(detail_table)

    result_data = {
        "test_provider": test_provider,
        "ref_provider": ref_provider,
        "model": model,
        "timestamp": datetime.now().isoformat(),
        "verdict": result.verdict.value,
        "score": result.score,
        "accuracy_delta": result.accuracy_delta,
        "response_similarity": result.response_similarity,
        "latency_ratio": result.latency_ratio,
        "hard_accuracy_test": result.hard_accuracy_test,
        "hard_accuracy_reference": result.hard_accuracy_reference,
        "evidence": result.evidence,
    }
    store.save_result(test_provider, model, result_data)

    console.print(
        Panel(
            f"[bold]Verdict: [{v_color}]{result.verdict.value}[/{v_color}][/bold]\n"
            f"Score: {result.score:.2f} | Confidence: {result.confidence:.2f}",
            expand=False,
        )
    )


if __name__ == "__main__":
    cli()
