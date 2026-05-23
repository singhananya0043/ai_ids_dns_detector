"""
Traffic & DNS Attack Simulator
Generates realistic network flows + DNS queries with injected attack patterns.

Attack Types (IDS):
  - port_scan       : Sequential port probing from one source
  - ddos            : High-volume traffic flood to one target
  - brute_force     : Repeated login attempts (SSH/FTP/HTTP)
  - lateral_movement: Internal host-to-host pivoting

Attack Types (DNS):
  - dns_tunneling   : Data exfiltration via long encoded subdomains
  - dga             : Domain Generation Algorithm (malware C2 beaconing)
  - typosquatting   : Look-alike domains for phishing
  - fast_flux       : Rapid IP rotation on a domain (botnet evasion)
  - dns_amplification: DNS reflection DDoS amplification
"""

import random
import string
import math
import time
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────

@dataclass
class NetworkFlow:
    """Represents a single network flow (5-tuple + stats)."""
    timestamp: float
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str            # TCP / UDP / ICMP
    pkt_count: int
    byte_count: int
    duration: float          # seconds
    flags: str               # TCP flags string e.g. "SYN", "SYN-ACK", "FIN"
    label: str               # normal / attack type


@dataclass
class DNSEvent:
    """Represents a single DNS query/response event."""
    timestamp: float
    src_ip: str
    query_name: str          # FQDN queried
    query_type: str          # A, AAAA, TXT, MX, NS …
    response_code: str       # NOERROR, NXDOMAIN, SERVFAIL …
    response_ips: List[str]  # resolved IPs (empty for NXDOMAIN)
    ttl: int                 # seconds
    payload_size: int        # bytes
    label: str               # normal / attack type


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _rand_ip(private: bool = True) -> str:
    if private:
        prefix = random.choice(["192.168", "10.0", "172.16"])
        return f"{prefix}.{random.randint(1, 254)}.{random.randint(1, 254)}"
    return f"{random.randint(1, 223)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"


def _rand_port(well_known: bool = False) -> int:
    if well_known:
        return random.choice([22, 23, 25, 53, 80, 110, 143, 443, 445, 3306, 3389, 5900])
    return random.randint(1024, 65535)


def _entropy(s: str) -> float:
    """Shannon entropy of a string."""
    if not s:
        return 0.0
    freq = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    n = len(s)
    return -sum((v / n) * math.log2(v / n) for v in freq.values())


LEGIT_DOMAINS = [
    "google.com", "youtube.com", "facebook.com", "amazon.com",
    "microsoft.com", "apple.com", "cloudflare.com", "github.com",
    "stackoverflow.com", "wikipedia.org", "reddit.com", "twitter.com",
    "netflix.com", "linkedin.com", "instagram.com", "office365.com",
    "teams.microsoft.com", "zoom.us", "dropbox.com", "slack.com",
]

DGA_TLDS = [".com", ".net", ".org", ".info", ".biz"]


def _dga_domain(seed: int = None) -> str:
    """Generate a DGA-like domain with high entropy."""
    rng = random.Random(seed)
    length = rng.randint(10, 22)
    chars = string.ascii_lowercase + string.digits
    name = "".join(rng.choices(chars, k=length))
    tld = rng.choice(DGA_TLDS)
    return name + tld


def _tunnel_subdomain() -> str:
    """Generate a DNS-tunneling-style encoded subdomain."""
    # base32/hex-like encoding of exfiltrated data
    encoded_len = random.randint(40, 80)
    chars = string.ascii_lowercase + string.digits
    encoded = "".join(random.choices(chars, k=encoded_len))
    # split into labels of max 63 chars
    labels = [encoded[i:i+30] for i in range(0, len(encoded), 30)]
    base_domain = random.choice(["tunnel.example.com", "exfil.badactor.net", "c2srv.org"])
    return ".".join(labels) + "." + base_domain


def _typosquat_domain() -> str:
    """Generate a typosquatted version of a legit domain."""
    base = random.choice(["google", "microsoft", "amazon", "paypal", "apple"])
    typos = [
        base + "security.com",
        base + "-login.net",
        base.replace("o", "0") + ".com",
        base + "support.org",
        "www-" + base + ".com",
        base + "verify.net",
    ]
    return random.choice(typos)


# ─────────────────────────────────────────────
# Network Flow Generator
# ─────────────────────────────────────────────

class NetworkFlowSimulator:
    """Generates labeled network flow records."""

    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        self._hosts = [_rand_ip() for _ in range(20)]
        self._t = time.time() - 3600  # start 1 hour ago

    def _tick(self, delta: float = None) -> float:
        self._t += delta if delta else random.uniform(0.01, 2.0)
        return self._t

    # ── Normal traffic ──────────────────────────────

    def _normal_flow(self) -> NetworkFlow:
        src = random.choice(self._hosts)
        dst = random.choice(self._hosts + [_rand_ip(private=False)])
        return NetworkFlow(
            timestamp=self._tick(),
            src_ip=src,
            dst_ip=dst,
            src_port=_rand_port(),
            dst_port=random.choice([80, 443, 53, 22, 25, 3306]),
            protocol=random.choice(["TCP", "UDP"]),
            pkt_count=random.randint(2, 120),
            byte_count=random.randint(200, 150_000),
            duration=round(random.uniform(0.1, 30.0), 3),
            flags=random.choice(["SYN-ACK", "ACK", "FIN-ACK", "PSH-ACK"]),
            label="normal",
        )

    # ── Port Scan ────────────────────────────────────

    def _port_scan_burst(self, n: int = 30) -> List[NetworkFlow]:
        src = _rand_ip()
        dst = random.choice(self._hosts)
        flows = []
        for port in random.sample(range(1, 65535), n):
            flows.append(NetworkFlow(
                timestamp=self._tick(0.01),
                src_ip=src,
                dst_ip=dst,
                src_port=_rand_port(),
                dst_port=port,
                protocol="TCP",
                pkt_count=1,
                byte_count=random.randint(40, 80),
                duration=round(random.uniform(0.001, 0.05), 4),
                flags="SYN",
                label="port_scan",
            ))
        return flows

    # ── DDoS ─────────────────────────────────────────

    def _ddos_burst(self, n: int = 50) -> List[NetworkFlow]:
        dst = random.choice(self._hosts)
        flows = []
        for _ in range(n):
            flows.append(NetworkFlow(
                timestamp=self._tick(0.005),
                src_ip=_rand_ip(private=False),
                dst_ip=dst,
                src_port=_rand_port(),
                dst_port=random.choice([80, 443]),
                protocol=random.choice(["TCP", "UDP"]),
                pkt_count=random.randint(500, 5000),
                byte_count=random.randint(500_000, 5_000_000),
                duration=round(random.uniform(0.1, 1.0), 3),
                flags=random.choice(["SYN", "ACK", ""]),
                label="ddos",
            ))
        return flows

    # ── Brute Force ──────────────────────────────────

    def _brute_force_burst(self, n: int = 20) -> List[NetworkFlow]:
        src = _rand_ip(private=False)
        dst = random.choice(self._hosts)
        service_port = random.choice([22, 21, 3389, 5900])
        flows = []
        for _ in range(n):
            flows.append(NetworkFlow(
                timestamp=self._tick(0.3),
                src_ip=src,
                dst_ip=dst,
                src_port=_rand_port(),
                dst_port=service_port,
                protocol="TCP",
                pkt_count=random.randint(4, 10),
                byte_count=random.randint(300, 1200),
                duration=round(random.uniform(0.5, 3.0), 3),
                flags="PSH-ACK",
                label="brute_force",
            ))
        return flows

    # ── Lateral Movement ─────────────────────────────

    def _lateral_movement_burst(self, n: int = 10) -> List[NetworkFlow]:
        pivot = random.choice(self._hosts)
        targets = random.sample([h for h in self._hosts if h != pivot], min(n, len(self._hosts) - 1))
        flows = []
        for tgt in targets:
            flows.append(NetworkFlow(
                timestamp=self._tick(1.0),
                src_ip=pivot,
                dst_ip=tgt,
                src_port=_rand_port(),
                dst_port=random.choice([445, 135, 3389, 22]),
                protocol="TCP",
                pkt_count=random.randint(10, 80),
                byte_count=random.randint(1000, 50_000),
                duration=round(random.uniform(1.0, 20.0), 3),
                flags="PSH-ACK",
                label="lateral_movement",
            ))
        return flows

    # ── Public API ───────────────────────────────────

    def generate(self, n_normal: int = 400, attack_ratio: float = 0.25) -> pd.DataFrame:
        """
        Generate a mixed dataset of normal + attack flows.

        Args:
            n_normal:     Number of normal flows to generate.
            attack_ratio: Fraction of records that will be attacks.

        Returns:
            pd.DataFrame with one row per flow.
        """
        flows: List[NetworkFlow] = []

        # Normal traffic
        for _ in range(n_normal):
            flows.append(self._normal_flow())

        # Attack injections
        n_attacks = int(n_normal * attack_ratio)
        attack_generators = [
            self._port_scan_burst,
            self._ddos_burst,
            self._brute_force_burst,
            self._lateral_movement_burst,
        ]
        per_attack = max(1, n_attacks // len(attack_generators))
        for gen in attack_generators:
            try:
                batch = gen(per_attack)
            except TypeError:
                batch = [gen()]
            if isinstance(batch, list):
                flows.extend(batch)
            else:
                flows.append(batch)

        random.shuffle(flows)
        return pd.DataFrame([f.__dict__ for f in flows])


# ─────────────────────────────────────────────
# DNS Event Generator
# ─────────────────────────────────────────────

class DNSSimulator:
    """Generates labeled DNS query/response records."""

    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        self._hosts = [_rand_ip() for _ in range(15)]
        self._t = time.time() - 3600

    def _tick(self, delta: float = None) -> float:
        self._t += delta if delta else random.uniform(0.5, 5.0)
        return self._t

    def _resp_ip(self, n: int = 1) -> List[str]:
        return [_rand_ip(private=False) for _ in range(n)]

    # ── Normal DNS ───────────────────────────────────

    def _normal_query(self) -> DNSEvent:
        domain = random.choice(LEGIT_DOMAINS)
        subdomain = random.choice(["www.", "mail.", "api.", "cdn.", ""])
        return DNSEvent(
            timestamp=self._tick(),
            src_ip=random.choice(self._hosts),
            query_name=subdomain + domain,
            query_type=random.choice(["A", "AAAA", "MX", "TXT"]),
            response_code="NOERROR",
            response_ips=self._resp_ip(random.randint(1, 3)),
            ttl=random.choice([60, 300, 600, 3600]),
            payload_size=random.randint(60, 512),
            label="normal",
        )

    # ── DNS Tunneling ────────────────────────────────

    def _dns_tunnel_query(self) -> DNSEvent:
        return DNSEvent(
            timestamp=self._tick(0.1),
            src_ip=random.choice(self._hosts),
            query_name=_tunnel_subdomain(),
            query_type=random.choice(["TXT", "NULL", "CNAME"]),
            response_code="NOERROR",
            response_ips=[],
            ttl=0,
            payload_size=random.randint(400, 4096),
            label="dns_tunneling",
        )

    # ── DGA ──────────────────────────────────────────

    def _dga_query(self) -> DNSEvent:
        domain = _dga_domain(seed=random.randint(0, 99999))
        return DNSEvent(
            timestamp=self._tick(0.5),
            src_ip=random.choice(self._hosts),
            query_name=domain,
            query_type="A",
            response_code=random.choice(["NXDOMAIN", "NXDOMAIN", "NOERROR"]),
            response_ips=self._resp_ip(1) if random.random() > 0.6 else [],
            ttl=random.choice([0, 1, 5]),
            payload_size=random.randint(60, 200),
            label="dga",
        )

    # ── Typosquatting ────────────────────────────────

    def _typosquat_query(self) -> DNSEvent:
        return DNSEvent(
            timestamp=self._tick(),
            src_ip=random.choice(self._hosts),
            query_name=_typosquat_domain(),
            query_type="A",
            response_code="NOERROR",
            response_ips=self._resp_ip(1),
            ttl=random.choice([30, 60]),
            payload_size=random.randint(60, 300),
            label="typosquatting",
        )

    # ── Fast-Flux DNS ────────────────────────────────

    def _fast_flux_burst(self, n: int = 15) -> List[DNSEvent]:
        domain = f"fastflux{random.randint(1000,9999)}.net"
        events = []
        for _ in range(n):
            events.append(DNSEvent(
                timestamp=self._tick(0.2),
                src_ip=random.choice(self._hosts),
                query_name=domain,
                query_type="A",
                response_code="NOERROR",
                response_ips=self._resp_ip(random.randint(4, 8)),  # many rotating IPs
                ttl=random.randint(1, 10),                          # very low TTL
                payload_size=random.randint(100, 400),
                label="fast_flux",
            ))
        return events

    # ── DNS Amplification ────────────────────────────

    def _amplification_burst(self, n: int = 20) -> List[DNSEvent]:
        events = []
        victim_ip = _rand_ip(private=False)
        for _ in range(n):
            events.append(DNSEvent(
                timestamp=self._tick(0.05),
                src_ip=victim_ip,           # spoofed victim as src
                query_name="ANY." + random.choice(LEGIT_DOMAINS),
                query_type="ANY",
                response_code="NOERROR",
                response_ips=self._resp_ip(random.randint(5, 15)),
                ttl=random.randint(60, 3600),
                payload_size=random.randint(2000, 4096),  # amplified response
                label="dns_amplification",
            ))
        return events

    # ── Public API ───────────────────────────────────

    def generate(self, n_normal: int = 400, attack_ratio: float = 0.30) -> pd.DataFrame:
        """
        Generate a mixed DNS event dataset.

        Args:
            n_normal:     Number of normal DNS queries.
            attack_ratio: Fraction of records that will be attacks.

        Returns:
            pd.DataFrame with one row per DNS event.
        """
        events: List[DNSEvent] = []

        for _ in range(n_normal):
            events.append(self._normal_query())

        n_attacks = int(n_normal * attack_ratio)
        attack_gens = [
            self._dns_tunnel_query,
            self._dga_query,
            self._typosquat_query,
        ]
        burst_gens = [
            self._fast_flux_burst,
            self._amplification_burst,
        ]

        per_single = max(1, n_attacks // (len(attack_gens) + len(burst_gens)))

        for gen in attack_gens:
            for _ in range(per_single):
                events.append(gen())

        for gen in burst_gens:
            batch = gen(per_single)
            events.extend(batch)

        random.shuffle(events)
        return pd.DataFrame([e.__dict__ for e in events])


# ─────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Network Flow Simulation ===")
    net_sim = NetworkFlowSimulator(seed=42)
    net_df = net_sim.generate(n_normal=200)
    print(net_df["label"].value_counts())
    print(net_df.head(3))

    print("\n=== DNS Event Simulation ===")
    dns_sim = DNSSimulator(seed=42)
    dns_df = dns_sim.generate(n_normal=200)
    print(dns_df["label"].value_counts())
    print(dns_df.head(3))
