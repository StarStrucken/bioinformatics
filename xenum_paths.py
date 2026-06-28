import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("XENUM_DATA_DIR", ROOT / "data")).expanduser()
OUTPUT_ROOT = Path(os.environ.get("XENUM_OUTPUT_DIR", ROOT / "outputs")).expanduser()
DATASETS = Path(os.environ.get("XENUM_DATASETS", ROOT / "datasets.tsv")).expanduser()

def data_dir(dataset_id):
    return DATA_ROOT / str(dataset_id)

def out_dir(dataset_id):
    p = OUTPUT_ROOT / str(dataset_id)
    p.mkdir(parents=True, exist_ok=True)
    return p

def existing_out_dir(dataset_id):
    return OUTPUT_ROOT / str(dataset_id)
