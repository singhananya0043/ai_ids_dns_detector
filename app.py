"""
AI-Powered IDS + DNS Anomaly Detector — Streamlit Dashboard

Run:
    streamlit run app.py
    OR
    python -X utf8 main.py --dashboard
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import time
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data.simulator         import NetworkFlowSimulator, DNSSimulator
from features.ids_features   import IDSFeatureExtractor
from features.dns_features   import DNSFeatureExtractor
from detection.ids_model     import IDSModel
from detection.dns_model     import DNSModel
from detection.correlator    import ThreatCorrelator
from llm.explainer           import ThreatExplainer

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="AI IDS + DNS Anomaly Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 100%);
        padding: 1.5rem 2rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        border: 1px solid #2d2d5e;
    }
    .metric-card {
        background: #1a1a2e;
        border-radius: 8px;
        padding: 1rem;
        border-left: 4px solid #4c6ef5;
        margin-bottom: 0.5rem;
    }
    .critical { border-left-color: #ff4444 !important; }
    .high     { border-left-color: #ff8800 !important; }
    .medium   { border-left-color: #ffcc00 !important; }
    .low      { border-left-color: #44bb44 !important; }
    .stButton > button {
        background: linear-gradient(135deg, #4c6ef5, #7048e8);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Severity Colors
# ─────────────────────────────────────────────

SEV_COLORS = {
    "CRITICAL": "#ff4444",
    "HIGH":     "#ff8800",
    "MEDIUM":   "#ffcc00",
    "LOW":      "#44bb44",
    "INFO":     "#888888",
}

# ─────────────────────────────────────────────
# Session State
# ─────────────────────────────────────────────

if "pipeline_run" not in st.session_state:
    st.session_state.pipeline_run = False
if "events_df" not in st.session_state:
    st.session_state.events_df = pd.DataFrame()
if "ids_metrics" not in st.session_state:
    st.session_state.ids_metrics = {}
if "dns_metrics" not in st.session_state:
    st.session_state.dns_metrics = {}
if "flow_df" not in st.session_state:
    st.session_state.flow_df = pd.DataFrame()
if "dns_df" not in st.session_state:
    st.session_state.dns_df = pd.DataFrame()
if "ids_preds" not in st.session_state:
    st.session_state.ids_preds = pd.DataFrame()
if "dns_preds" not in st.session_state:
    st.session_state.dns_preds = pd.DataFrame()


# ─────────────────────────────────────────────
# Pipeline Runner (cached by seed/n)
# ─────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def run_full_pipeline(n: int, seed: int):
    """Run end-to-end pipeline and return all results."""
    net_sim = NetworkFlowSimulator(seed=seed)
    dns_sim = DNSSimulator(seed=seed)
    flow_df = net_sim.generate(n_normal=n)
    dns_df  = dns_sim.generate(n_normal=n)

    ids_ext = IDSFeatureExtractor(window_sec=60)
    dns_ext = DNSFeatureExtractor(window_sec=60)
    ids_feat, ids_labels = ids_ext.extract_df(flow_df)
    dns_feat, dns_labels = dns_ext.extract_df(dns_df)

    ids_model = IDSModel()
    dns_model = DNSModel()
    ids_normal = ids_feat[ids_labels == "normal"].reset_index(drop=True)
    dns_normal = dns_feat[dns_labels == "normal"].reset_index(drop=True)
    ids_metrics = ids_model.train(ids_normal, ids_feat, ids_labels)
    dns_metrics = dns_model.train(dns_normal, dns_feat, dns_labels)

    ids_preds = ids_model.predict(ids_feat)
    dns_preds = dns_model.predict(dns_feat)

    flow_sorted = flow_df.sort_values("timestamp").reset_index(drop=True)
    dns_sorted  = dns_df.sort_values("timestamp").reset_index(drop=True)

    correlator = ThreatCorrelator()
    events     = correlator.correlate(flow_sorted, ids_preds, dns_sorted, dns_preds)
    events_df  = ThreatCorrelator.to_dataframe(events)

    return {
        "flow_df":     flow_sorted,
        "dns_df":      dns_sorted,
        "ids_preds":   ids_preds,
        "dns_preds":   dns_preds,
        "events_df":   events_df,
        "ids_metrics": ids_metrics,
        "dns_metrics": dns_metrics,
        "ids_labels":  ids_labels,
        "dns_labels":  dns_labels,
    }


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    n_samples = st.slider("Normal Traffic Samples", 100, 1000, 400, 50)
    seed      = st.slider("Random Seed", 1, 100, 42)
    top_n     = st.slider("Top N Threats to Show", 5, 50, 15)
    st.markdown("---")
    run_btn   = st.button("🚀 Run Detection Pipeline", use_container_width=True)
    st.markdown("---")
    st.markdown("### 📖 About")
    st.markdown("""
    **AI IDS + DNS Anomaly Detector**

    Combines two detection layers:
    - 🔴 **IDS Engine**: Network flow anomaly detection (Isolation Forest + Random Forest)
    - 🟠 **DNS Engine**: DNS threat detection (entropy, DGA, tunneling)
    - 🔗 **Correlator**: Cross-engine threat fusion
    - 🤖 **Claude AI**: Natural language explanations
    """)

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1 style="color:#00d4ff; margin:0; font-size:1.8rem;">
        🛡️ AI-Powered IDS + DNS Anomaly Detector
    </h1>
    <p style="color:#aaaacc; margin:0.3rem 0 0 0;">
        Two-Layer Network Defense · Isolation Forest · Random Forest · Claude AI
    </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Run Pipeline
# ─────────────────────────────────────────────

if run_btn:
    with st.spinner("Running detection pipeline..."):
        result = run_full_pipeline(n_samples, seed)
        st.session_state.update({
            "pipeline_run": True,
            "events_df":    result["events_df"],
            "ids_metrics":  result["ids_metrics"],
            "dns_metrics":  result["dns_metrics"],
            "flow_df":      result["flow_df"],
            "dns_df":       result["dns_df"],
            "ids_preds":    result["ids_preds"],
            "dns_preds":    result["dns_preds"],
            "ids_labels":   result["ids_labels"],
            "dns_labels":   result["dns_labels"],
        })
    st.success("Pipeline complete!")

# ─────────────────────────────────────────────
# Main Dashboard (only after run)
# ─────────────────────────────────────────────

if st.session_state.pipeline_run:
    events_df   = st.session_state.events_df
    ids_metrics = st.session_state.ids_metrics
    dns_metrics = st.session_state.dns_metrics
    flow_df     = st.session_state.flow_df
    dns_df      = st.session_state.dns_df
    ids_preds   = st.session_state.ids_preds
    dns_preds   = st.session_state.dns_preds

    # ── KPI Row ─────────────────────────────────────
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    sev_counts = events_df["severity"].value_counts() if not events_df.empty else {}

    kpi1.metric("Total Threats",  len(events_df))
    kpi2.metric("🔴 Critical",    sev_counts.get("CRITICAL", 0))
    kpi3.metric("🟠 High",        sev_counts.get("HIGH", 0))
    kpi4.metric("🟡 Medium",      sev_counts.get("MEDIUM", 0))
    kpi5.metric("IDS F1 / DNS F1",
                f"{ids_metrics.get('rf_weighted_f1','?')} / {dns_metrics.get('rf_weighted_f1','?')}")

    st.markdown("---")

    # ── Tabs ────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🚨 Threats", "📡 IDS Analysis", "🌐 DNS Analysis",
        "🔗 Correlation", "📊 Model Metrics",
    ])

    # ── Tab 1: Threat Events ─────────────────────────
    with tab1:
        st.subheader("🚨 Threat Events (sorted by score)")

        if events_df.empty:
            st.success("✅ No threats detected in this run.")
        else:
            # Severity filter
            sev_filter = st.multiselect(
                "Filter by Severity",
                ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
                default=["CRITICAL", "HIGH", "MEDIUM"],
            )
            filtered = events_df[events_df["severity"].isin(sev_filter)] if sev_filter else events_df

            # Color-coded table
            def color_severity(val):
                color = SEV_COLORS.get(val, "#888")
                return f"color: {color}; font-weight: bold"

            display_cols = ["src_ip", "severity", "threat_score",
                            "ids_attack", "dns_attack", "correlation"]
            styled = (
                filtered[display_cols]
                .head(top_n)
                .style.map(color_severity, subset=["severity"])
                .format({"threat_score": "{:.1f}"})
            )
            st.dataframe(styled, use_container_width=True, height=400)

            # ── Threat Score Distribution ────────────────
            col_a, col_b = st.columns(2)
            with col_a:
                fig = px.histogram(
                    events_df, x="threat_score", nbins=20,
                    color="severity",
                    color_discrete_map=SEV_COLORS,
                    title="Threat Score Distribution",
                    template="plotly_dark",
                )
                st.plotly_chart(fig, use_container_width=True)

            with col_b:
                fig = px.pie(
                    events_df, names="severity",
                    color="severity",
                    color_discrete_map=SEV_COLORS,
                    title="Severity Breakdown",
                    template="plotly_dark",
                )
                st.plotly_chart(fig, use_container_width=True)

            # ── Top Threats Bar ──────────────────────────
            top_threats = events_df.head(15).copy()
            top_threats["label"] = top_threats["src_ip"] + " (" + top_threats["severity"] + ")"
            fig = px.bar(
                top_threats, x="threat_score", y="label",
                orientation="h",
                color="severity",
                color_discrete_map=SEV_COLORS,
                title="Top 15 Threat Scores by Source IP",
                template="plotly_dark",
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=500)
            st.plotly_chart(fig, use_container_width=True)

            # ── AI Explanations ──────────────────────────
            st.markdown("---")
            st.subheader("🤖 AI Threat Analysis")
            top_event = events_df.iloc[0]
            with st.expander(
                f"🔍 {top_event['src_ip']} — {top_event['severity']} "
                f"(score {top_event['threat_score']:.1f})", expanded=True
            ):
                explainer = ThreatExplainer()
                expl = explainer.explain(
                    src_ip         = str(top_event["src_ip"]),
                    ids_attack     = str(top_event["ids_attack"]),
                    dns_attack     = str(top_event["dns_attack"]),
                    severity       = str(top_event["severity"]),
                    threat_score   = float(top_event["threat_score"]),
                    correlation    = str(top_event["correlation"]),
                    ids_confidence = float(top_event["ids_confidence"]),
                    dns_confidence = float(top_event["dns_confidence"]),
                )
                api_src = "🟢 Claude API" if "claude" in expl.get("source","") else "🟡 Rule-based"
                st.caption(f"Analysis engine: {api_src}")
                st.markdown(f"**Explanation:** {expl.get('explanation','')}")
                st.markdown(f"**Attack Vector:** {expl.get('attack_vector','')}")
                st.markdown(f"**Business Impact:** {expl.get('business_impact','')}")
                steps = expl.get("remediation", [])
                if steps:
                    st.markdown("**Remediation Steps:**")
                    for i, s in enumerate(steps, 1):
                        st.markdown(f"  {i}. {s}")
                ttps = expl.get("mitre_ttps", [])
                if ttps:
                    st.markdown(f"**MITRE ATT&CK:** `{'`, `'.join(ttps)}`")

    # ── Tab 2: IDS Analysis ──────────────────────────
    with tab2:
        st.subheader("📡 IDS — Network Flow Analysis")
        col1, col2 = st.columns(2)

        with col1:
            ids_dist = ids_preds["attack_type"].value_counts().reset_index()
            ids_dist.columns = ["Attack Type", "Count"]
            fig = px.bar(
                ids_dist, x="Attack Type", y="Count",
                color="Attack Type",
                title="IDS Detected Attack Types",
                template="plotly_dark",
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.histogram(
                ids_preds, x="anomaly_score", nbins=40,
                color="is_anomaly",
                title="IDS Anomaly Score Distribution",
                template="plotly_dark",
                color_discrete_map={True: "#ff4444", False: "#44bb44"},
            )
            st.plotly_chart(fig, use_container_width=True)

        # Flow timeline
        flow_with_pred = flow_df.copy()
        flow_with_pred["attack_type"] = ids_preds["attack_type"].values
        flow_with_pred["is_anomaly"]  = ids_preds["is_anomaly"].values

        fig = px.scatter(
            flow_with_pred, x="timestamp", y="byte_count",
            color="attack_type",
            size="pkt_count",
            hover_data=["src_ip", "dst_ip", "dst_port"],
            title="Network Flow Timeline — Byte Volume",
            template="plotly_dark",
            opacity=0.7,
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 3: DNS Analysis ──────────────────────────
    with tab3:
        st.subheader("🌐 DNS — Query Analysis")
        col1, col2 = st.columns(2)

        with col1:
            dns_dist = dns_preds["attack_type"].value_counts().reset_index()
            dns_dist.columns = ["Attack Type", "Count"]
            fig = px.bar(
                dns_dist, x="Attack Type", y="Count",
                color="Attack Type",
                title="DNS Detected Attack Types",
                template="plotly_dark",
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            dns_df_plot = dns_df.copy()
            dns_df_plot["domain_len"] = dns_df_plot["query_name"].str.len()
            dns_df_plot["attack_type"] = dns_preds["attack_type"].values

            fig = px.box(
                dns_df_plot, x="attack_type", y="domain_len",
                color="attack_type",
                title="Domain Name Length by Attack Type",
                template="plotly_dark",
            )
            st.plotly_chart(fig, use_container_width=True)

        # DNS payload size scatter
        dns_plot = dns_df.copy()
        dns_plot["attack_type"] = dns_preds["attack_type"].values
        fig = px.scatter(
            dns_plot, x="timestamp", y="payload_size",
            color="attack_type",
            hover_data=["src_ip", "query_name", "query_type"],
            title="DNS Query Timeline — Payload Size",
            template="plotly_dark",
            opacity=0.7,
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 4: Correlation Map ───────────────────────
    with tab4:
        st.subheader("🔗 Cross-Engine Correlation")

        if events_df.empty:
            st.info("No correlated threats to display.")
        else:
            # Heatmap: IDS attack vs DNS attack
            pivot = (
                events_df
                .groupby(["ids_attack", "dns_attack"])
                .size()
                .reset_index(name="count")
            )
            fig = px.density_heatmap(
                pivot, x="ids_attack", y="dns_attack", z="count",
                title="IDS Attack vs DNS Attack — Co-occurrence Heatmap",
                template="plotly_dark",
                color_continuous_scale="Reds",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Scatter: IDS conf vs DNS conf, colored by severity
            fig = px.scatter(
                events_df, x="ids_confidence", y="dns_confidence",
                color="severity",
                size="threat_score",
                color_discrete_map=SEV_COLORS,
                hover_data=["src_ip", "ids_attack", "dns_attack"],
                title="IDS Confidence vs DNS Confidence per Threat",
                template="plotly_dark",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Raw correlated events table
            st.markdown("**Correlated Events — Full Table**")
            st.dataframe(events_df, use_container_width=True)

    # ── Tab 5: Model Metrics ─────────────────────────
    with tab5:
        st.subheader("📊 Model Performance Metrics")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🔴 IDS Model")
            for k, v in ids_metrics.items():
                if k == "classes":
                    st.markdown(f"**Classes:** `{', '.join(v)}`")
                else:
                    st.metric(k.replace("_", " ").title(), v)

        with col2:
            st.markdown("### 🟠 DNS Model")
            for k, v in dns_metrics.items():
                if k == "classes":
                    st.markdown(f"**Classes:** `{', '.join(v)}`")
                else:
                    st.metric(k.replace("_", " ").title(), v)

        # Feature importance (RF)
        st.markdown("---")
        st.markdown("### Feature Importance (Random Forest)")
        fi_tab1, fi_tab2 = st.tabs(["IDS Features", "DNS Features"])

        # We need access to models — re-run lightweight training
        @st.cache_data(show_spinner=False)
        def get_feature_importances(n, seed):
            from features.ids_features import IDSFeatureExtractor, ALL_FEATURES as IDS_ALL
            from features.dns_features import DNSFeatureExtractor, ALL_DNS_FEATURES as DNS_ALL
            net_sim = NetworkFlowSimulator(seed=seed)
            dns_sim = DNSSimulator(seed=seed)
            fdf = net_sim.generate(n_normal=n)
            ddf = dns_sim.generate(n_normal=n)
            ie  = IDSFeatureExtractor(60)
            de  = DNSFeatureExtractor(60)
            if_, il = ie.extract_df(fdf)
            df_, dl = de.extract_df(ddf)
            im = IDSModel()
            dm = DNSModel()
            im.train(if_[il == "normal"], if_, il)
            dm.train(df_[dl == "normal"], df_, dl)
            ids_fi = pd.DataFrame({"feature": im.feature_names, "importance": im.rf_model.feature_importances_}).sort_values("importance", ascending=False)
            dns_fi = pd.DataFrame({"feature": dm.feature_names, "importance": dm.rf_model.feature_importances_}).sort_values("importance", ascending=False)
            return ids_fi, dns_fi

        ids_fi, dns_fi = get_feature_importances(n_samples, seed)

        with fi_tab1:
            fig = px.bar(ids_fi.head(15), x="importance", y="feature",
                         orientation="h", title="Top 15 IDS Features",
                         template="plotly_dark")
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)

        with fi_tab2:
            fig = px.bar(dns_fi.head(15), x="importance", y="feature",
                         orientation="h", title="Top 15 DNS Features",
                         template="plotly_dark")
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)

else:
    # ── Welcome screen ───────────────────────────────
    st.markdown("""
    <div style="text-align:center; padding: 4rem 2rem;">
        <h2 style="color:#4c6ef5;">👈 Configure settings and click <strong>Run Detection Pipeline</strong></h2>
        <p style="color:#888; font-size:1.1rem;">
            The system will simulate network traffic + DNS queries,<br>
            train detection models, correlate threats, and generate AI explanations.
        </p>
        <br>
        <table style="margin:auto; color:#aaa; border-collapse:collapse;">
            <tr><td style="padding:0.5rem 1rem;">🔴 IDS Engine</td><td>Port scan · DDoS · Brute force · Lateral movement</td></tr>
            <tr><td style="padding:0.5rem 1rem;">🟠 DNS Engine</td><td>DNS tunneling · DGA · Typosquatting · Fast-flux · Amplification</td></tr>
            <tr><td style="padding:0.5rem 1rem;">🔗 Correlator</td><td>Cross-engine threat fusion with severity scoring</td></tr>
            <tr><td style="padding:0.5rem 1rem;">🤖 Claude AI</td><td>Natural language explanations + MITRE ATT&CK mapping</td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)
