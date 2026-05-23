"""
Shared logging & pretty-print utilities using Rich.
"""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box
from typing import List, Dict
import pandas as pd

console = Console()

SEVERITY_COLORS = {
    "CRITICAL": "bold red",
    "HIGH":     "bold orange3",
    "MEDIUM":   "bold yellow",
    "LOW":      "cyan",
    "INFO":     "dim white",
}


def print_banner():
    banner = Text()
    banner.append("  AI-Powered IDS + DNS Anomaly Detector\n", style="bold cyan")
    banner.append("  Two-Layer Network Defense System\n",        style="dim cyan")
    console.print(Panel(banner, border_style="cyan", padding=(0, 2)))


def print_threat_table(events_df: pd.DataFrame, max_rows: int = 20):
    if events_df.empty:
        console.print("[green]✓ No threats detected.[/green]")
        return

    table = Table(
        title=f"🚨 Threat Events ({len(events_df)} total)",
        box=box.ROUNDED,
        show_lines=True,
        border_style="red",
    )
    table.add_column("Rank",       style="dim",      width=5)
    table.add_column("Severity",   width=10)
    table.add_column("Score",      width=7)
    table.add_column("Source IP",  style="cyan",     width=18)
    table.add_column("IDS Attack", style="yellow",   width=18)
    table.add_column("DNS Attack", style="magenta",  width=20)
    table.add_column("Correlation",                  width=45)

    for i, (_, row) in enumerate(events_df.head(max_rows).iterrows(), 1):
        sev   = str(row.get("severity", "INFO"))
        color = SEVERITY_COLORS.get(sev, "white")
        table.add_row(
            str(i),
            f"[{color}]{sev}[/{color}]",
            f"[bold]{row.get('threat_score', 0):.1f}[/bold]",
            str(row.get("src_ip", "")),
            str(row.get("ids_attack", "none")),
            str(row.get("dns_attack", "none")),
            str(row.get("correlation", ""))[:45],
        )
    console.print(table)


def print_model_metrics(ids_metrics: Dict, dns_metrics: Dict):
    table = Table(title="📊 Model Training Metrics", box=box.SIMPLE_HEAD)
    table.add_column("Metric",     style="bold")
    table.add_column("IDS Model",  style="cyan")
    table.add_column("DNS Model",  style="magenta")

    rows = [
        ("IF Threshold",     "if_threshold"),
        ("IF Attack Recall", "if_attack_recall"),
        ("RF Weighted F1",   "rf_weighted_f1"),
        ("# Features",       "n_features"),
    ]
    for label, key in rows:
        table.add_row(
            label,
            str(ids_metrics.get(key, "—")),
            str(dns_metrics.get(key, "—")),
        )
    console.print(table)


def print_explanation(src_ip: str, expl: Dict):
    sev_map = {"claude": "green", "fallback": "yellow"}
    src     = expl.get("source", "fallback")
    color   = sev_map.get("claude" if "claude" in src else "fallback", "white")

    console.print(f"\n[bold]📋 Threat Analysis for [cyan]{src_ip}[/cyan]"
                  f"  ([{color}]{src}[/{color}])[/bold]")
    console.print(f"  [bold]Explanation:[/bold] {expl.get('explanation','')}")
    console.print(f"  [bold]Attack Vector:[/bold] {expl.get('attack_vector','')}")
    console.print(f"  [bold]Business Impact:[/bold] {expl.get('business_impact','')}")
    steps = expl.get("remediation", [])
    if steps:
        console.print("  [bold]Remediation:[/bold]")
        for i, s in enumerate(steps, 1):
            console.print(f"    {i}. {s}")
    ttps = expl.get("mitre_ttps", [])
    if ttps:
        console.print(f"  [bold]MITRE ATT&CK:[/bold] {', '.join(ttps)}")
