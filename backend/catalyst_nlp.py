"""Financial catalyst NLP & sentiment analyzer tailored for Indian (NSE/SEBI/RBI) & global equity news."""

from __future__ import annotations

import re

import numpy as np

_POS_TERMS: dict[str, float] = {
    "beat": 1.3, "beats": 1.4, "surpass": 1.3, "surpasses": 1.3, "outperform": 1.4,
    "upgrade": 1.6, "upgraded": 1.6, "upgrades": 1.5, "raise": 1.0, "raises": 1.2,
    "raised": 1.1, "surge": 1.3, "surges": 1.3, "surging": 1.2, "rally": 1.2,
    "rallies": 1.2, "soar": 1.3, "soars": 1.3, "jump": 1.0, "jumps": 1.0,
    "record": 0.9, "growth": 0.8, "accelerate": 1.2, "accelerating": 1.2,
    "strong": 0.9, "bullish": 1.4, "breakout": 1.2, "buyback": 1.1, "repurchase": 1.1,
    "dividend": 0.7, "bonus": 1.0, "profit": 0.8, "profitable": 0.9, "expansion": 0.9,
    "partnership": 0.9, "contract": 1.0, "order": 0.7, "approval": 1.3, "approved": 1.3,
    "upside": 1.1, "demand": 0.7, "boom": 1.1, "gain": 0.8, "gains": 0.8, "rebound": 1.0,
    "momentum": 0.8, "win": 1.0, "wins": 1.0, "overweight": 1.2, "accumulate": 1.1,
    "capex": 0.6, "ebitda": 0.5, "pat": 0.5,
}

_NEG_TERMS: dict[str, float] = {
    "miss": -1.4, "misses": -1.5, "missed": -1.4, "downgrade": -1.6, "downgraded": -1.6,
    "cut": -1.0, "cuts": -1.0, "slump": -1.3, "slumps": -1.3, "plunge": -1.5,
    "plunges": -1.5, "tumble": -1.3, "tumbles": -1.3, "tumbled": -1.3, "drop": -0.9,
    "drops": -0.9, "fall": -0.8, "falls": -0.8, "decline": -0.9, "declines": -0.9,
    "weak": -1.0, "weakness": -1.0, "bearish": -1.4, "selloff": -1.4, "sell-off": -1.4,
    "lawsuit": -1.3, "antitrust": -1.3, "probe": -1.2, "investigation": -1.2,
    "fine": -0.9, "penalty": -1.0, "recall": -1.3, "layoffs": -1.1, "crisis": -1.5,
    "warning": -1.3, "warns": -1.3, "worries": -1.0, "fear": -1.0, "fears": -1.0,
    "loss": -1.0, "losses": -1.1, "deficit": -0.9, "slowdown": -1.1, "slowing": -1.0,
    "tariff": -0.9, "tariffs": -0.9, "headwind": -1.0, "headwinds": -1.0,
    "underweight": -1.2, "fraud": -2.0, "bankruptcy": -2.0, "dilution": -1.3, "pledge": -1.1,
    "attrition": -0.8, "default": -1.6,
}

_PHRASES: dict[str, float] = {
    "raises guidance": 2.0,
    "raised guidance": 2.0,
    "lifts outlook": 1.8,
    "beats estimates": 1.8,
    "tops estimates": 1.7,
    "profit jumps": 1.6,
    "net profit rises": 1.5,
    "pat rises": 1.5,
    "order win": 1.5,
    "bags order": 1.5,
    "wins order": 1.5,
    "target raised": 1.6,
    "buy rating": 1.3,
    "bonus issue": 1.2,
    "lowers guidance": -2.0,
    "cuts guidance": -2.0,
    "misses estimates": -1.8,
    "profit falls": -1.5,
    "net profit drops": -1.5,
    "target cut": -1.6,
    "sebi notice": -1.6,
    "rbi restriction": -1.7,
    "rbi penalty": -1.4,
    "fails to": -1.0,
}

_CATALYST_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("Earnings Beat", re.compile(r"\b(beats?|tops?|surpass\w*)\s+(estimates?|expectations?|earnings|eps|revenue)|\b(net\s+profit|pat)\s+(jumps?|surges?|rises?)\b", re.I)),
    ("Earnings Miss", re.compile(r"\b(miss\w*|falls?\s+short\s+of)\s+(estimates?|expectations?|earnings|eps|revenue)|\b(net\s+profit|pat)\s+(falls?|drops?|slumps?)\b", re.I)),
    ("Order / Deal Win", re.compile(r"\b(order\s+win|bags?\s+(rs|₹|\w+\s+)?order|wins?\s+order|order\s+book|contract\s+win|multi-year\s+deal)\b", re.I)),
    ("Guidance Raise", re.compile(r"\b(rais\w+|lift\w+|boost\w+|hik\w+)\s+(guidance|outlook|forecast)\b", re.I)),
    ("Guidance Cut", re.compile(r"\b(cut\w*|lower\w*|slash\w*|trim\w*)\s+(guidance|outlook|forecast)\b", re.I)),
    ("Brokerage Upgrade", re.compile(r"\b(upgrad\w+|target\s+(price\s+)?rais\w+|overweight|buy\s+rating|outperform)\b", re.I)),
    ("Brokerage Downgrade", re.compile(r"\b(downgrad\w+|target\s+(price\s+)?cut|underweight|sell\s+rating)\b", re.I)),
    ("Dividend / Bonus", re.compile(r"\b(buyback|share\s+repurchase|dividend|bonus\s+issue|stock\s+split)\b", re.I)),
    ("Capex / Expansion", re.compile(r"\b(acquir\w+|acquisition|merger|partnership|capex|expansion|launches?|fda\s+approv\w+)\b", re.I)),
    ("SEBI / RBI / Legal Risk", re.compile(r"\b(sebi|rbi\s+(penalty|ban|restriction)|antitrust|lawsuit|probe|investigation|tariff|recall|fine|tax\s+demand)\b", re.I)),
]


def analyze_financial_news(title: str, summary: str = "") -> tuple[float, str, list[str]]:
    """Return (sentiment_score in [-1, 1], sentiment_label, catalyst_tags)."""
    raw = f"{title}. {summary}"
    combined = f"{title} {title} {summary}".lower()

    score = 0.0
    hits = 0
    for phrase, weight in _PHRASES.items():
        if phrase in combined:
            score += weight
            hits += 1

    tokens = re.findall(r"[a-z]+(?:-[a-z]+)?", combined)
    for idx, tok in enumerate(tokens):
        negated = idx > 0 and tokens[idx - 1] in {"not", "no", "never", "without", "fails"}
        val = _POS_TERMS.get(tok, 0.0) or _NEG_TERMS.get(tok, 0.0)
        if val != 0.0:
            if negated:
                val = -val * 0.7
            score += val
            hits += 1

    tags: list[str] = []
    for tag, pattern in _CATALYST_PATTERNS:
        if pattern.search(raw):
            tags.append(tag)

    if hits == 0:
        norm = 0.0
    else:
        norm = round(float(np.tanh(score / max(2.4, hits * 0.82))), 3)

    if norm >= 0.16:
        label = "Bullish"
    elif norm <= -0.16:
        label = "Bearish"
    else:
        label = "Neutral"

    return norm, label, tags[:3]
