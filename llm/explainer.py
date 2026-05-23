"""
Claude AI Threat Explainer
Uses Claude API (claude-sonnet-4-6) with prompt caching on the system prompt.
Generates human-readable threat explanations, severity justifications,
and remediation steps for ThreatEvents.

Falls back to rule-based explanations when no API key is set.
"""

from __future__ import annotations

import os
import json
from typing import Optional, List, Dict
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
MODEL_ID          = "claude-sonnet-4-6"

SYSTEM_PROMPT = """You are an expert network security analyst specializing in intrusion detection and DNS threat analysis. Your role is to analyze network security alerts and provide:

1. **Threat Explanation**: Clear, concise description of what the detected threat means
2. **Attack Vector**: How the attack is likely being carried out
3. **Business Impact**: Potential consequences if not addressed
4. **Remediation Steps**: Concrete, prioritized actions to take (3-5 steps)
5. **MITRE ATT&CK**: Relevant tactic/technique IDs if applicable

You analyze two types of signals:
- **IDS Alerts**: Network flow anomalies (port_scan, ddos, brute_force, lateral_movement)
- **DNS Alerts**: DNS layer threats (dns_tunneling, dga, typosquatting, fast_flux, dns_amplification)

When both IDS and DNS signals come from the same host, treat it as a correlated multi-vector attack with elevated severity.

Be concise but thorough. Format your response as JSON with keys:
  explanation, attack_vector, business_impact, remediation (list), mitre_ttps (list)"""

# ─────────────────────────────────────────────
# Rule-Based Fallback Explanations
# ─────────────────────────────────────────────

FALLBACK_EXPLANATIONS: Dict[str, Dict] = {
    "port_scan": {
        "explanation":    "An external or internal host is systematically probing ports on a target to map open services.",
        "attack_vector":  "Attacker sends SYN packets to many ports in rapid succession to identify listening services.",
        "business_impact":"Reconnaissance phase — attacker is gathering intelligence before launching a targeted exploit.",
        "remediation":    [
            "Block the scanning IP at the firewall immediately",
            "Enable port-scan detection rules in your IDS/IPS",
            "Review exposed services on the target host",
            "Enable connection rate limiting on edge devices",
        ],
        "mitre_ttps": ["T1046 - Network Service Discovery"],
    },
    "ddos": {
        "explanation":    "A high-volume flood of traffic is targeting a host, overwhelming its resources.",
        "attack_vector":  "Attacker (or botnet) sends massive packet volumes to exhaust bandwidth or CPU.",
        "business_impact":"Service unavailability, potential revenue loss, SLA violations.",
        "remediation":    [
            "Activate DDoS scrubbing / upstream filtering",
            "Apply rate limiting and traffic shaping at the edge",
            "Contact ISP for upstream null-routing of attack traffic",
            "Enable anycast routing to distribute load",
        ],
        "mitre_ttps": ["T1498 - Network Denial of Service"],
    },
    "brute_force": {
        "explanation":    "Repeated login attempts from a single source indicate a credential brute-force attack.",
        "attack_vector":  "Attacker tries many password combinations against SSH/RDP/FTP/HTTP login endpoints.",
        "business_impact":"Risk of unauthorized access, lateral movement, and data breach if credentials are compromised.",
        "remediation":    [
            "Block the source IP at the firewall",
            "Enable account lockout policy after N failed attempts",
            "Implement Multi-Factor Authentication (MFA)",
            "Move SSH/RDP to non-standard ports or VPN-only access",
        ],
        "mitre_ttps": ["T1110 - Brute Force", "T1078 - Valid Accounts"],
    },
    "lateral_movement": {
        "explanation":    "An internal host is connecting to multiple other internal hosts on administrative ports.",
        "attack_vector":  "Compromised host pivoting through the network using SMB/RDP/SSH to spread access.",
        "business_impact":"Active breach — attacker moving toward high-value assets, potential ransomware deployment.",
        "remediation":    [
            "Isolate the source host from the network immediately",
            "Force password reset for all accounts on the host",
            "Conduct forensic analysis of the host",
            "Implement network segmentation and zero-trust policies",
            "Check for new scheduled tasks, services, or persistence mechanisms",
        ],
        "mitre_ttps": ["T1021 - Remote Services", "T1570 - Lateral Tool Transfer"],
    },
    "dns_tunneling": {
        "explanation":    "Data is being exfiltrated or C2 commands tunneled through DNS queries using encoded subdomains.",
        "attack_vector":  "Malware encodes data in DNS query subdomains; DNS responses carry C2 commands back.",
        "business_impact":"Active data exfiltration bypassing traditional DLP; persistent C2 channel.",
        "remediation":    [
            "Block the affected host's external DNS access",
            "Deploy DNS-layer security (Cisco Umbrella, Cloudflare Gateway)",
            "Inspect and block unusually long or high-entropy DNS queries",
            "Identify and remove the malware on the source host",
        ],
        "mitre_ttps": ["T1071.004 - Application Layer Protocol: DNS", "T1048 - Exfiltration Over Alternative Protocol"],
    },
    "dga": {
        "explanation":    "The host is querying algorithmically generated domains, a hallmark of malware C2 beaconing.",
        "attack_vector":  "Malware uses a Domain Generation Algorithm to find active C2 servers that evade blocklists.",
        "business_impact":"Active malware infection; host is communicating with attacker-controlled infrastructure.",
        "remediation":    [
            "Isolate the affected host immediately",
            "Block NXDOMAIN-heavy traffic at DNS resolver level",
            "Deploy DGA detection at DNS gateway",
            "Run full antivirus and EDR scan on the host",
        ],
        "mitre_ttps": ["T1568.002 - Dynamic Resolution: Domain Generation Algorithms"],
    },
    "fast_flux": {
        "explanation":    "A domain is rapidly cycling through many IP addresses with very low TTLs — botnet evasion tactic.",
        "attack_vector":  "Botnet uses infected machines as rotating proxies; low TTL prevents effective IP blocking.",
        "business_impact":"Difficulty blocking C2 infrastructure; indicates botnet membership.",
        "remediation":    [
            "Block the domain at DNS resolver level",
            "Report the domain to threat intel feeds",
            "Investigate the querying host for botnet infection",
            "Enable DNS-based threat intelligence feeds",
        ],
        "mitre_ttps": ["T1568.001 - Dynamic Resolution: Fast Flux DNS"],
    },
    "typosquatting": {
        "explanation":    "A user or process queried a look-alike domain designed to impersonate a legitimate site.",
        "attack_vector":  "Attacker registers misspelled domains to capture credential input or serve malware.",
        "business_impact":"Risk of credential theft, malware download, or financial fraud.",
        "remediation":    [
            "Block the typosquatted domain at DNS level",
            "Alert the user who queried it and check for credential exposure",
            "Enable browser-level phishing protection",
            "Register common misspellings of your own domains",
        ],
        "mitre_ttps": ["T1566.002 - Phishing: Spearphishing Link", "T1204 - User Execution"],
    },
    "dns_amplification": {
        "explanation":    "DNS ANY queries with spoofed source IPs are being used to amplify traffic toward a victim.",
        "attack_vector":  "Attacker spoofs victim's IP as DNS query source; large ANY responses flood the victim.",
        "business_impact":"Victim host experiences DDoS; your resolver is being abused as an amplifier.",
        "remediation":    [
            "Configure DNS resolver to reject ANY queries or rate-limit responses",
            "Block spoofed source IPs at edge (BCP38 anti-spoofing)",
            "Disable open recursive DNS resolution",
            "Contact upstream ISP about traffic filtering",
        ],
        "mitre_ttps": ["T1498.002 - Reflection Amplification"],
    },
    "correlated": {
        "explanation":    "Multiple attack vectors detected from the same host — IDS network anomaly AND DNS threat.",
        "attack_vector":  "Sophisticated multi-stage attack: network exploitation combined with DNS-based C2 or exfiltration.",
        "business_impact":"CRITICAL — active breach with command & control established. Immediate incident response required.",
        "remediation":    [
            "IMMEDIATELY isolate the source host from all networks",
            "Preserve forensic evidence (memory dump, disk image)",
            "Block all DNS from the host at resolver level",
            "Activate incident response plan and notify security team",
            "Audit all hosts the compromised host communicated with",
        ],
        "mitre_ttps": ["T1071 - Application Layer Protocol", "T1041 - Exfiltration Over C2 Channel"],
    },
}


def _fallback_explain(ids_attack: str, dns_attack: str, correlation: str) -> Dict:
    """Rule-based explanation when Claude API is unavailable."""
    # Correlated attack takes priority
    if ids_attack not in ("none", "normal") and dns_attack not in ("none", "normal"):
        base = FALLBACK_EXPLANATIONS.get("correlated", {}).copy()
        base["explanation"] = (
            f"Correlated attack: {ids_attack} (IDS) + {dns_attack} (DNS) from same host. "
            + base["explanation"]
        )
        return base

    for key in [ids_attack, dns_attack]:
        if key in FALLBACK_EXPLANATIONS:
            return FALLBACK_EXPLANATIONS[key].copy()

    return {
        "explanation":    f"Anomalous activity detected: {ids_attack or dns_attack}.",
        "attack_vector":  "Unknown vector — further analysis required.",
        "business_impact":"Potential security risk — investigate the source host.",
        "remediation":    ["Investigate the flagged host", "Review logs for additional context"],
        "mitre_ttps":     [],
    }


# ─────────────────────────────────────────────
# Claude AI Explainer
# ─────────────────────────────────────────────

class ThreatExplainer:
    """
    Generates natural-language threat explanations using Claude API.
    Falls back to rule-based explanations when no API key is configured.
    """

    def __init__(self):
        self._client = None
        if ANTHROPIC_API_KEY:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            except ImportError:
                pass

    def explain(
        self,
        src_ip:         str,
        ids_attack:     str,
        dns_attack:     str,
        severity:       str,
        threat_score:   float,
        correlation:    str,
        ids_confidence: float = 0.0,
        dns_confidence: float = 0.0,
    ) -> Dict:
        """
        Generate a threat explanation.

        Returns dict with: explanation, attack_vector, business_impact,
                           remediation (list), mitre_ttps (list), source ('claude'|'fallback')
        """
        if self._client:
            return self._claude_explain(
                src_ip, ids_attack, dns_attack, severity,
                threat_score, correlation, ids_confidence, dns_confidence,
            )
        return {
            **_fallback_explain(ids_attack, dns_attack, correlation),
            "source": "fallback",
        }

    def _claude_explain(
        self,
        src_ip:         str,
        ids_attack:     str,
        dns_attack:     str,
        severity:       str,
        threat_score:   float,
        correlation:    str,
        ids_confidence: float,
        dns_confidence: float,
    ) -> Dict:
        """Call Claude API with prompt caching on system prompt."""
        try:
            import anthropic

            user_msg = f"""Analyze this network security alert:

Source IP: {src_ip}
Severity: {severity}
Threat Score: {threat_score}/10
Correlation: {correlation}

IDS Alert: {ids_attack} (confidence: {ids_confidence:.1%})
DNS Alert: {dns_attack} (confidence: {dns_confidence:.1%})

Provide your analysis as JSON only (no markdown fences)."""

            response = self._client.messages.create(
                model=MODEL_ID,
                max_tokens=2048,
                system=[
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},  # prompt caching
                    }
                ],
                messages=[{"role": "user", "content": user_msg}],
            )

            raw = response.content[0].text.strip()
            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            result = json.loads(raw)
            result["source"] = "claude"
            return result

        except Exception as e:
            fallback = _fallback_explain(ids_attack, dns_attack, correlation)
            fallback["source"] = f"fallback (claude error: {str(e)[:60]})"
            return fallback

    def explain_batch(self, events_df: "pd.DataFrame") -> List[Dict]:
        """Explain a batch of ThreatEvent rows from a DataFrame."""
        results = []
        for _, row in events_df.iterrows():
            result = self.explain(
                src_ip         = str(row.get("src_ip", "")),
                ids_attack     = str(row.get("ids_attack", "none")),
                dns_attack     = str(row.get("dns_attack", "none")),
                severity       = str(row.get("severity", "MEDIUM")),
                threat_score   = float(row.get("threat_score", 0.0)),
                correlation    = str(row.get("correlation", "")),
                ids_confidence = float(row.get("ids_confidence", 0.0)),
                dns_confidence = float(row.get("dns_confidence", 0.0)),
            )
            results.append(result)
        return results


# ─────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    explainer = ThreatExplainer()
    print(f"Claude API available: {explainer._client is not None}\n")

    # Test correlated attack
    result = explainer.explain(
        src_ip         = "192.168.1.50",
        ids_attack     = "lateral_movement",
        dns_attack     = "dns_tunneling",
        severity       = "CRITICAL",
        threat_score   = 9.5,
        correlation    = "Host performing recon AND DNS exfiltration — active C2 channel",
        ids_confidence = 0.95,
        dns_confidence = 1.00,
    )

    print(f"Source: {result.get('source')}")
    print(f"\nExplanation:\n  {result.get('explanation')}")
    print(f"\nAttack Vector:\n  {result.get('attack_vector')}")
    print(f"\nRemediation Steps:")
    for i, step in enumerate(result.get("remediation", []), 1):
        print(f"  {i}. {step}")
    print(f"\nMITRE TTPs: {result.get('mitre_ttps')}")
