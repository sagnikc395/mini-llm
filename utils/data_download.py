# pretraining done on the "The Verdict" from Wiki

import os
import urllib.request

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

if not os.path.exists(os.path.join(DATA_DIR, "the-verdict.txt")):
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
    os.makedirs(DATA_DIR, exist_ok=True)
    file_path = os.path.join(DATA_DIR, "the-verdict.txt")
    urllib.request.urlretrieve(url, file_path)
    print(f"Downloaded data into {file_path}")
else:
    print("Pretraining files already exists!")