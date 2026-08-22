from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FILE_PATH = PROJECT_ROOT / "data" / "the_verdict.txt"

raw_text = FILE_PATH.read_text(encoding="utf-8")

print(f"Total number of characters: {len(raw_text)}")
print(raw_text[:99])
