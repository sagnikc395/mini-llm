from pathlib import Path

from datasets import load_dataset

from llm.config import PretrainingConfig
DATASET_NAME = PretrainingConfig.dataset_name
DATA_DIR = Path(__file__).resolve().parents[2] / PretrainingConfig.data_dir


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(DATASET_NAME)
    dataset.save_to_disk(str(DATA_DIR / "common-crawl-sample"))
    print(f"Saved {DATASET_NAME} to {DATA_DIR / 'common-crawl-sample'}")


if __name__ == "__main__":
    main()
