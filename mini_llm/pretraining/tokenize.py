from pathlib import Path 
import re 

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FILE_PATH = PROJECT_ROOT / "data" / "the_verdict.txt"
raw_text = FILE_PATH.read_text(encoding="utf-8")

result = re.split(r'([,.]|\s)',raw_text)
print(result)