"""
Downloads the train/test splits from the VeritaResearch/claim-extraction
composite dataset (Claimbuster + PoliClaim Gold + AVeriTeC, ~13k sentences).
Source: https://github.com/VeritaResearch/claim-extraction
"""

import argparse
import os
import urllib.request

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")

BASE_URL = "https://raw.githubusercontent.com/VeritaResearch/claim-extraction/main/data/ours"
FILES = {
    "train.csv": f"{BASE_URL}/train.csv",
    "test.csv": f"{BASE_URL}/test.csv",
}


def download(force: bool = False) -> None:
    os.makedirs(RAW_DIR, exist_ok=True)

    for filename, url in FILES.items():
        dest = os.path.join(RAW_DIR, filename)
        if os.path.exists(dest) and not force:
            print(f"  {filename} already exists, skipping (use --force to re-download)")
            continue
        print(f"  Downloading {filename}...")
        urllib.request.urlretrieve(url, dest)
        print(f"  Saved to {dest}")


def report() -> None:
    import csv

    for filename in FILES:
        path = os.path.join(RAW_DIR, filename)
        if not os.path.exists(path):
            print(f"  {filename}: not found")
            continue
        with open(path) as f:
            rows = list(csv.DictReader(f))
        claims = sum(1 for r in rows if r.get("label", r.get("is_claim", "")).strip() in ("1", "True", "true"))
        print(f"  {filename}: {len(rows)} rows, {claims} claims ({100*claims/len(rows):.1f}% positive)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download claim detection dataset")
    parser.add_argument("--force", action="store_true", help="Re-download even if files exist")
    args = parser.parse_args()

    print("Downloading dataset...")
    download(force=args.force)
    print("\nDataset summary:")
    report()
    print("\nDone. Files saved to data/raw/")
