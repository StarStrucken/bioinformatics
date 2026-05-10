from pathlib import Path
import gzip
import hashlib
import zipfile

import requests


out = Path("downloads")
out.mkdir(exist_ok=True)

meta_path = out / "meta.tsv"

done = set()
if meta_path.exists():
    for line in meta_path.read_text().splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) == 4 and parts[2] and parts[3]:
            done.add(parts[0])

if not meta_path.exists():
    meta_path.write_text("dataset_id\turl\tsha256\tn_cells\n")

for line in Path("datasets.tsv").read_text().splitlines()[1:]:
    dataset_id, url = line.split("\t")
    zip_path = out / f"{dataset_id}.zip"

    if dataset_id in done and zip_path.exists():
        print("skip", dataset_id)
        continue

    try:
        if not zip_path.exists():
            print("download", dataset_id)
            r = requests.get(url, stream=True, timeout=60)
            r.raise_for_status()

            with zip_path.open("wb") as f:
                for chunk in r.iter_content(1024 * 1024):
                    if chunk:
                        f.write(chunk)

        with zip_path.open("rb") as f:
            sha256 = hashlib.file_digest(f, "sha256").hexdigest()

        with zipfile.ZipFile(zip_path) as z:
            cell_file = next(
                name for name in z.namelist()
                if Path(name).name in {"cells.csv.gz", "cells.csv"}
            )

            if cell_file.endswith(".gz"):
                with z.open(cell_file) as raw, gzip.GzipFile(fileobj=raw) as f:
                    n_cells = sum(1 for _ in f) - 1
            else:
                with z.open(cell_file) as f:
                    n_cells = sum(1 for _ in f) - 1

        with meta_path.open("a") as f:
            f.write(f"{dataset_id}\t{url}\t{sha256}\t{n_cells}\n")

        print("ok", dataset_id)

    except Exception as e:
        print("miss", dataset_id, e)
