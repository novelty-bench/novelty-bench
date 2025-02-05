import re


def extract_five_digits(text: str) -> str:
    # Look for exactly 5 consecutive digits
    match = re.search(r"\b\d{5}\b", text)

    # Return the matched digits if found, otherwise None
    return match.group() if match else ""


EXTRACTION_RULES = {
    "curated-89": extract_five_digits,
}
