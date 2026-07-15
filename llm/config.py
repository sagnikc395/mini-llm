from dataclasses import dataclass

@dataclass(frozen=True)
class PretrainingConfig:
    dataset_name: str | None  = "agentlans/common-crawl-sample"
    data_dir: str | None = "data"