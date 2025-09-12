import re
from typing import Optional, Tuple

PAT = re.compile(r"^\s*@say\s*\{\s*name\s*:\s*(C\s*(\d{1,2}))\s*\}\s*(?:\|\s*(.*))?\s*$", re.IGNORECASE)
NAME_PAT = re.compile(r"\bC\s*(\d{1,2})\b", re.IGNORECASE)


def parse_schema(text: str) -> Tuple[Optional[str], bool]:
    # Strict: entire response must be a single line matching the schema
    m = PAT.match(text.strip())
    if not m:
        return None, False
    name_raw = m.group(1)
    name = "C" + str(int(re.sub(r"\s+", "", name_raw)[1:]))
    return name.upper(), True


def extract_nl_name(text: str, valid: set) -> Optional[str]:
    # Require exactly one occurrence of a Ck token in the entire text
    matches = [f"C{int(m.group(1))}" for m in NAME_PAT.finditer(text)]
    if len(matches) != 1:
        return None
    token = matches[0].upper()
    return token if token in valid else None
