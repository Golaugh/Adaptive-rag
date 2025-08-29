from enum import Enum, auto
import re


class QueryType(Enum):
    SIMPLE = auto()      # straightforward, no DB context
    MIDDLE = auto()      # multiple factors / some context
    COMPLEX = auto()     # full system planning


def classify_query(text: str) -> QueryType:
    t = text.lower().strip()

    # Hard cues for complex
    complex_cues = ["project plan", "timeline", "milestone", "risk", "budget", "dependency"]
    if any(k in t for k in complex_cues) or len(t) > 240:
        return QueryType.COMPLEX

    # Multiple-factor heuristics
    signals = 0
    signals += 1 if re.search(r"\b(and|or|vs\.|compare|trade[- ]off|option)\b", t) else 0
    signals += 1 if re.search(r"\bdeadline|cost|team|resource|priority|constraint\b", t) else 0
    signals += 1 if re.search(r"\bif\b.*\bthen\b", t) else 0
    signals += 1 if 100 < len(t) <= 240 else 0

    if signals >= 2:
        return QueryType.MIDDLE

    return QueryType.SIMPLE