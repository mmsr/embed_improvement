from pathlib import Path

def count_processed(path: str) -> int:
    p = Path(path)
    if not p.exists():
        return 0

    with p.open() as f:
        return sum(1 for _ in f)
