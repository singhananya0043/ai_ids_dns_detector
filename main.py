"""
AI-Powered IDS + DNS Anomaly Detector — CLI Entry Point

Usage:
    python -X utf8 main.py                     # run simulation, print results
    python -X utf8 main.py --n 500             # larger dataset
    python -X utf8 main.py --seed 123          # reproducible run
    python -X utf8 main.py --explain           # add Claude AI explanations
    python -X utf8 main.py --dashboard         # launch Streamlit dashboard
    python -X utf8 main.py --top 15            # show top N threats
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from data.simulator        import NetworkFlowSimulator, DNSSimulator
from features.ids_features  import IDSFeatureExtractor
from features.dns_features  import DNSFeatureExtractor
from detection.ids_model    import IDSModel
from detection.dns_model    import DNSModel
from detection.correlator   import ThreatCorrelator
from llm.explainer          import ThreatExplainer
from utils.logger           import (
    console, print_banner, print_threat_table,
    print_model_metrics, print_explanation,
)


def run_pipeline(n: int = 400, seed: int = 42, explain: bool = False, top: int = 10):
    print_banner()

    # ── 1. Simulate Data ────────────────────────────
    console.print(f"\n[bold cyan]▶ Step 1/5 — Generating simulated traffic[/bold cyan]  "
                  f"(n={n}, seed={seed})")
    net_sim  = NetworkFlowSimulator(seed=seed)
    dns_sim  = DNSSimulator(seed=seed)
    flow_df  = net_sim.generate(n_normal=n)
    dns_df   = dns_sim.generate(n_normal=n)
    console.print(f"   Network flows : {len(flow_df):,}  "
                  f"({(flow_df['label'] != 'normal').sum()} attacks)")
    console.print(f"   DNS events    : {len(dns_df):,}  "
                  f"({(dns_df['label'] != 'normal').sum()} attacks)")

    # ── 2. Feature Extraction ───────────────────────
    console.print("\n[bold cyan]▶ Step 2/5 — Extracting features[/bold cyan]")
    ids_ext = IDSFeatureExtractor(window_sec=60)
    dns_ext = DNSFeatureExtractor(window_sec=60)
    ids_feat, ids_labels = ids_ext.extract_df(flow_df)
    dns_feat, dns_labels = dns_ext.extract_df(dns_df)
    console.print(f"   IDS features : {ids_feat.shape}  |  "
                  f"DNS features : {dns_feat.shape}")

    # ── 3. Train Models ─────────────────────────────
    console.print("\n[bold cyan]▶ Step 3/5 — Training detection models[/bold cyan]")

    ids_model = IDSModel()
    dns_model = DNSModel()

    ids_normal  = ids_feat[ids_labels == "normal"].reset_index(drop=True)
    dns_normal  = dns_feat[dns_labels == "normal"].reset_index(drop=True)

    ids_metrics = ids_model.train(ids_normal, ids_feat, ids_labels)
    dns_metrics = dns_model.train(dns_normal, dns_feat, dns_labels)

    print_model_metrics(ids_metrics, dns_metrics)

    ids_model.save()
    dns_model.save()
    console.print("   Models saved to [dim]data/models/[/dim]")

    # ── 4. Predict & Correlate ──────────────────────
    console.print("\n[bold cyan]▶ Step 4/5 — Running detection & correlation[/bold cyan]")

    ids_preds = ids_model.predict(ids_feat)
    dns_preds = dns_model.predict(dns_feat)

    flow_sorted = flow_df.sort_values("timestamp").reset_index(drop=True)
    dns_sorted  = dns_df.sort_values("timestamp").reset_index(drop=True)

    correlator = ThreatCorrelator()
    events = correlator.correlate(flow_sorted, ids_preds, dns_sorted, dns_preds)
    events_df = ThreatCorrelator.to_dataframe(events)

    console.print(f"   Total threat events : [bold red]{len(events_df)}[/bold red]")
    if not events_df.empty:
        dist = events_df["severity"].value_counts().to_dict()
        for sev, cnt in dist.items():
            console.print(f"   {sev:<10}: {cnt}")

    print_threat_table(events_df, max_rows=top)

    # ── 5. AI Explanations ──────────────────────────
    if explain and not events_df.empty:
        console.print(f"\n[bold cyan]▶ Step 5/5 — AI Threat Explanations[/bold cyan]  "
                      f"(top {min(top, 3)} threats)")
        explainer = ThreatExplainer()
        api_status = "[green]Claude API[/green]" if explainer._client else "[yellow]rule-based fallback[/yellow]"
        console.print(f"   Engine: {api_status}")

        for _, row in events_df.head(min(top, 3)).iterrows():
            expl = explainer.explain(
                src_ip         = str(row["src_ip"]),
                ids_attack     = str(row["ids_attack"]),
                dns_attack     = str(row["dns_attack"]),
                severity       = str(row["severity"]),
                threat_score   = float(row["threat_score"]),
                correlation    = str(row["correlation"]),
                ids_confidence = float(row["ids_confidence"]),
                dns_confidence = float(row["dns_confidence"]),
            )
            print_explanation(str(row["src_ip"]), expl)
    elif not explain:
        console.print("\n[dim]Tip: run with --explain to add AI threat analysis[/dim]")

    console.print("\n[bold green]✓ Pipeline complete.[/bold green]")
    return events_df


def launch_dashboard():
    import subprocess
    dashboard_path = os.path.join(os.path.dirname(__file__), "app.py")
    console.print("[bold cyan]Launching Streamlit dashboard...[/bold cyan]")
    subprocess.run([sys.executable, "-m", "streamlit", "run", dashboard_path], check=True)


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AI-Powered IDS + DNS Anomaly Detector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--n",         type=int,  default=400,   help="Normal traffic samples (default 400)")
    parser.add_argument("--seed",      type=int,  default=42,    help="Random seed (default 42)")
    parser.add_argument("--explain",   action="store_true",      help="Generate AI threat explanations")
    parser.add_argument("--dashboard", action="store_true",      help="Launch Streamlit dashboard")
    parser.add_argument("--top",       type=int,  default=10,    help="Show top N threats (default 10)")

    args = parser.parse_args()

    if args.dashboard:
        launch_dashboard()
    else:
        run_pipeline(n=args.n, seed=args.seed, explain=args.explain, top=args.top)


if __name__ == "__main__":
    main()
