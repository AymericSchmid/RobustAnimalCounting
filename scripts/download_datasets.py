import os
import subprocess
from pathlib import Path
from urllib.request import urlretrieve
import zipfile
import tarfile

from animal_counting.datasets.datasets_list import DATASETS, DATA_DIR


def download_file(url, output_path):
    print(f"Downloading {url} → {output_path}")
    urlretrieve(url, output_path)


def download_git(url, output_path):
    print(f"Cloning {url} → {output_path}")
    subprocess.run(["git", "clone", url, str(output_path)], check=True)


def extract_if_needed(file_path, extract_to):
    file_path = Path(file_path)

    if file_path.suffix == ".zip":
        print(f"Unzipping {file_path}")
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)

    elif file_path.suffix in [".gz", ".tgz"] or file_path.name.endswith(".tar.gz"):
        print(f"Extracting {file_path}")
        with tarfile.open(file_path, "r:gz") as tar_ref:
            tar_ref.extractall(extract_to)

    else:
        print(f"No extraction needed for {file_path}")


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for name, dataset in DATASETS.items():
        dataset_path = DATA_DIR / name

        # Skip if already downloaded
        if dataset_path.exists():
            print(f"[✓] {name} already exists → skipping")
            continue

        print(f"\n=== {name} ===")

        if dataset["type"] == "file":
            dataset_path.mkdir(parents=True, exist_ok=True)
            file_path = dataset_path / dataset["filename"]

            download_file(dataset["url"], file_path)

            extract_if_needed(file_path, dataset_path)

        elif dataset["type"] == "git":
            download_git(dataset["url"], dataset_path)

        elif dataset["type"] == "manual":
            print(f"[!] {name} must be downloaded manually")

    print("\nDone.")


if __name__ == "__main__":
    main()