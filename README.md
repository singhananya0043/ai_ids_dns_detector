# 🛡️ AI-Powered IDS + DNS Anomaly Detector

> **Two-Layer Network Defense System** — Combining Intrusion Detection with DNS Threat Analysis, powered by Machine Learning and Claude AI.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4-orange?style=flat-square&logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?style=flat-square&logo=streamlit)
![Claude AI](https://img.shields.io/badge/Claude-Sonnet%204.6-purple?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 📌 Overview

Most modern cyberattacks **touch DNS first** — C2 beacons, malware callbacks, and data exfiltration all leverage DNS before or alongside network-level attacks. This project fuses two detection layers into a unified threat intelligence pipeline:

| Layer | Engine | Detects |
|-------|--------|---------|
| 🔴 **IDS** | Isolation Forest + Random Forest | Port scans, DDoS, Brute force, Lateral movement |
| 🟠 **DNS** | Isolation Forest + Random Forest | DNS tunneling, DGA, Typosquatting, Fast-flux, Amplification |
| 🔗 **Correlator** | Rule-based fusion engine | Cross-layer threats with unified severity scoring |
| 🤖 **Claude AI** | claude-sonnet-4-6 | Natural language explanations + MITRE ATT&CK mapping |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│               NETWORK TRAFFIC CAPTURE                │
│              (Simulated / Live pcap)                 │
└──────────────┬──────────────────┬───────────────────┘
               │                  │
       ┌───────▼──────┐   ┌───────▼──────────┐
       │  IDS ENGINE  │   │  DNS ENGINE      │
       │              │   │                  │
       │ • Flow stats │   │ • Query freq     │
       │ • Packet size│   │ • Domain entropy │
       │ • Port scan  │   │ • DGA detection  │
       │ • DDoS flags │   │ • Tunneling      │
       │ • Brute force│   │ • Typosquatting  │
       │              │   │ • Fast-flux      │
       │ Isolation    │   │ Isolation        │
       │ Forest + RF  │   │ Forest + RF      │
       └───────┬──────┘   └───────┬──────────┘
               │                  │
       ┌───────▼──────────────────▼──────────┐
       │         CORRELATION ENGINE           │
       │  Cross-link IDS alerts ↔ DNS events  │
       │  Assign unified threat score (0-10)  │
       └───────────────┬─────────────────────┘
                       │
       ┌───────────────▼─────────────────────┐
       │         CLAUDE AI LAYER             │
       │  • Natural language threat reports  │
       │  • MITRE ATT&CK TTP mapping         │
       │  • Prioritised remediation steps    │
       └───────────────┬─────────────────────┘
                       │
       ┌───────────────▼─────────────────────┐
       │       STREAMLIT DASHBOARD           │
       │  • Real-time threat feed            │
       │  • DNS query heatmap                │
       │  • Traffic anomaly graphs           │
       └─────────────────────────────────────┘
```

---

## 🎯 Detected Threats

### IDS Layer
| Attack | Description |
|--------|-------------|
| `port_scan` | Sequential SYN probing to map open services |
| `ddos` | High-volume traffic flood exhausting host resources |
| `brute_force` | Repeated login attempts on SSH / RDP / FTP |
| `lateral_movement` | Internal host pivoting via SMB / RDP / WMI |

### DNS Layer
| Attack | Description |
|--------|-------------|
| `dns_tunneling` | Data exfiltration encoded in DNS subdomains |
| `dga` | Domain Generation Algorithm — malware C2 beaconing |
| `typosquatting` | Look-alike phishing domains |
| `fast_flux` | Rapid IP rotation to evade blocklists (botnets) |
| `dns_amplification` | DNS reflection DDoS amplification |

### Correlated (Cross-Layer) Threats
| Combination | Severity | Meaning |
|-------------|----------|---------|
| `lateral_movement` + `dns_tunneling` | 🔴 CRITICAL | Active C2 channel with internal pivoting |
| `brute_force` + `dga` | 🔴 CRITICAL | Compromised host beaconing to C2 |
| `ddos` + `dns_amplification` | 🔴 CRITICAL | Coordinated volumetric attack |
| `port_scan` + `dga` | 🟠 HIGH | Malware reconnaissance phase |
| `brute_force` + `typosquatting` | 🟠 HIGH | Credential phishing campaign |

---

## 📊 Model Performance

| Metric | IDS Model | DNS Model |
|--------|-----------|-----------|
| RF Weighted F1 | **1.0** | **1.0** |
| IF Attack Recall | **~93%** | **~86%** |
| Features | 23 | 24 |
| Algorithms | Isolation Forest + Random Forest | Isolation Forest + Random Forest |

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| ML Models | `scikit-learn` (Isolation Forest, Random Forest) |
| Feature Engineering | `pandas`, `numpy`, `scipy` |
| DNS Analysis | `dnslib`, `dnspython` |
| AI Explanations | `anthropic` (claude-sonnet-4-6) with prompt caching |
| Dashboard | `streamlit`, `plotly` |
| Terminal UI | `rich` |
| Traffic Simulation | Custom Python simulator |

---

## 📁 Project Structure

```
ai_ids_dns_detector/
├── data/
│   ├── simulator.py          # Network flow + DNS attack simulator
│   └── models/               # Saved ML models (.pkl) — gitignored
├── features/
│   ├── ids_features.py       # 23 network flow features (static + temporal)
│   └── dns_features.py       # 24 DNS features (entropy, DGA score, freq)
├── detection/
│   ├── ids_model.py          # Isolation Forest + RF for IDS
│   ├── dns_model.py          # Isolation Forest + RF for DNS
│   └── correlator.py         # Cross-engine threat fusion & scoring
├── llm/
│   └── explainer.py          # Claude API + rule-based fallback
├── utils/
│   └── logger.py             # Rich terminal pretty-printing
├── app.py                    # Streamlit dashboard (5 tabs)
├── main.py                   # CLI entry point
├── requirements.txt
├── .env.example              # API key template
└── .gitignore
```

---

## 🚀 Getting Started

### 1. Clone & Install

```bash
git clone https://github.com/singhananya0043/ai_ids_dns_detector.git
cd ai_ids_dns_detector
pip install -r requirements.txt
```

### 2. Configure API Key (optional but recommended)

```bash
cp .env.example .env
# Edit .env and add your Anthropic API key:
# ANTHROPIC_API_KEY=sk-ant-...
```

> Without an API key the system falls back to rule-based explanations with MITRE ATT&CK mappings — all detection still works fully.

### 3. Run

```bash
# CLI — simulate traffic, train models, detect & correlate threats
python -X utf8 main.py

# CLI with Claude AI explanations for top threats
python -X utf8 main.py --explain

# Larger dataset
python -X utf8 main.py --n 1000 --explain

# Launch Streamlit dashboard
python -X utf8 main.py --dashboard
# OR
streamlit run app.py
```

---

## 🖥️ Dashboard

The Streamlit dashboard provides **5 interactive tabs**:

| Tab | Contents |
|-----|----------|
| 🚨 **Threats** | Ranked threat table, score distribution, severity pie chart, Claude AI analysis |
| 📡 **IDS Analysis** | Attack type breakdown, anomaly score histogram, flow timeline |
| 🌐 **DNS Analysis** | DNS attack types, domain length box plots, payload timeline |
| 🔗 **Correlation** | IDS × DNS co-occurrence heatmap, confidence scatter plot |
| 📊 **Model Metrics** | F1 scores, recall, feature importance charts |

---

## 🤖 Claude AI Explanations

When `ANTHROPIC_API_KEY` is set, the system uses **claude-sonnet-4-6** with prompt caching to generate:

- 📋 **Threat explanation** — context-aware, specific to the IDS + DNS signal combination
- ⚔️ **Attack vector** — how the attack is being carried out
- 💼 **Business impact** — potential consequences
- 🔧 **Remediation steps** — prioritised with time windows (0-15min, 1hr, 24hr, 7 days)
- 🎯 **MITRE ATT&CK** — full TTP mapping with sub-techniques

### Example Output

```
📋 Threat Analysis for 10.0.9.8  [Claude API]
  Severity   : CRITICAL  |  Score: 10.0/10
  IDS Attack : lateral_movement (95.5% confidence)
  DNS Attack : dns_tunneling (100% confidence)

  Explanation:
    Host is actively pivoting through internal network via SMB/RDP while
    simultaneously exfiltrating data through a covert DNS tunnel to an
    attacker-controlled C2 server...

  Remediation:
    1. [0-15 min]  Isolate host at switch level — do NOT power off
    2. [0-30 min]  Capture RAM dump and full disk image for forensics
    3. [1-4 hrs]   Block outbound DNS; analyse tunneling domain
    4. [4-24 hrs]  Hunt laterally; reset all credentials from this host
    5. [1-7 days]  Deploy DNS security solution; implement segmentation

  MITRE ATT&CK: T1071.004, T1048.001, T1572, T1021.002, T1550.002
```

---

## ⚙️ CLI Options

```
python -X utf8 main.py [OPTIONS]

Options:
  --n INT         Number of normal traffic samples to simulate (default: 400)
  --seed INT      Random seed for reproducibility (default: 42)
  --explain       Generate Claude AI threat explanations for top threats
  --dashboard     Launch the Streamlit dashboard
  --top INT       Number of top threats to display (default: 10)
```

---

## 🔮 Roadmap

- [ ] Live PCAP capture mode (Scapy integration)
- [ ] Real DNS resolver integration (dnspython)
- [ ] LSTM-based DGA classifier
- [ ] Threat intelligence feed integration (VirusTotal, Shodan)
- [ ] Slack / email alerting for CRITICAL events
- [ ] Docker containerisation
- [ ] Export threat reports to PDF

---

## 🙏 Acknowledgements

- [CICIDS Dataset](https://www.unb.ca/cic/datasets/) — inspiration for IDS feature design
- [Majestic Million](https://majestic.com/reports/majestic-million) — legitimate domain reference
- [Anthropic Claude](https://anthropic.com) — AI-powered threat explanations
- [MITRE ATT&CK](https://attack.mitre.org/) — threat taxonomy framework

---

## 📄 License

This project is licensed under the MIT License.

---

<p align="center">Built with 🛡️ and 🤖 by <a href="https://github.com/singhananya0043">singhananya0043</a></p>
