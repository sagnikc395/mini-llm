import urllib.request
from pathlib import Path 

PROJECT_ROOT = Path(__file__).resolve().parents[2]

url = ("https://raw.githubusercontent.com/rasbt/"
"LLMs-from-scratch/main/ch02/01_main-chapter-code/"
"the-verdict.txt")

DATA_DIR = PROJECT_ROOT/ "data"
DATA_DIR.mkdir(parents=True,exist_ok=True)

FILE_PATH = DATA_DIR / "the_verdict.txt"
urllib.request.urlretrieve(url,FILE_PATH)
print("File downloaded!")