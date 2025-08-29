import re
from typing import Optional, Tuple

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"\+?\d[\d\s().-]{7,}")


def extract_person_info(text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Very light heuristic for MVP: pull (name, email, phone) from text.
    Name: pick first capitalized two-word sequence; email & phone via regex.
    Title omitted in MVP.
    """
    email = EMAIL_RE.search(text)
    phone = PHONE_RE.search(text)

    name = None
    # Find two consecutive Capitalized words as a naive name
    tokens = re.findall(r"\b[A-Z][a-z]+\b", text)
    if len(tokens) >= 2:
        name = f"{tokens[0]} {tokens[1]}"

    return name, email.group(0) if email else None, phone.group(0) if phone else None