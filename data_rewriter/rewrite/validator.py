import re

NUMBER_RE = re.compile(r"-?\d+\.?\d*")

def extract_numbers(text: str):
    return NUMBER_RE.findall(text)

def validate_numbers(original: str, rewritten: str) -> bool:
    return extract_numbers(original) == extract_numbers(rewritten)
