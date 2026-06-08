from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"
OUTPUT_ROOT = ROOT / "outputs"
DATASETS = ROOT / "datasets.tsv"

def data_dir(dataset_id):
    return DATA_ROOT / str(dataset_id)

def out_dir(dataset_id):
    p = OUTPUT_ROOT / str(dataset_id)
    p.mkdir(parents=True, exist_ok=True)
    return p

def existing_out_dir(dataset_id):
    return OUTPUT_ROOT / str(dataset_id)
