from pathlib import Path
import gzip
import hashlib
import zipfile

import requests


Path("downloads").mkdir(exist_ok=True)
Path("data").mkdir(exist_ok=True)

meta_path = Path("downloads/meta.tsv")

done = set()
if meta_path.exists():
    for line in meta_path.read_text().splitlines()[1:]:
        cols = line.split("\t")
        if len(cols) == 4 and cols[2] and cols[3]:
            done.add(cols[0])
else:
    meta_path.write_text("dataset_id\turl\tsha256\tn_cells\n")

for line in Path("datasets.tsv").read_text().splitlines()[1:]:
    dataset_id, url = line.split("\t")

    zip_path = Path("downloads") / f"{dataset_id}.zip"
    data_dir = Path("data") / dataset_id

    if dataset_id in done and zip_path.exists() and data_dir.exists():
        print("skip", dataset_id)
        continue

    try:
        if not zip_path.exists():
            print("download", dataset_id)
            tmp = zip_path.with_suffix(".part")

            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with tmp.open("wb") as f:
                    for chunk in r.iter_content(1024 * 1024):
                        if chunk:
                            f.write(chunk)

            tmp.rename(zip_path)

        if not data_dir.exists():
            print("unzip", dataset_id)
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(data_dir)

        if dataset_id not in done:
            with zip_path.open("rb") as f:
                sha256 = hashlib.file_digest(f, "sha256").hexdigest()

            cell_file = next(data_dir.rglob("cells.csv.gz"), None)
            if cell_file is None:
                cell_file = next(data_dir.rglob("cells.csv"))

            if cell_file.suffix == ".gz":
                with gzip.open(cell_file, "rb") as f:
                    n_cells = sum(1 for _ in f) - 1
            else:
                with cell_file.open("rb") as f:
                    n_cells = sum(1 for _ in f) - 1

            with meta_path.open("a") as f:
                f.write(f"{dataset_id}\t{url}\t{sha256}\t{n_cells}\n")

        print("ok", dataset_id)

    except Exception as e:
        print("miss", dataset_id, e)
