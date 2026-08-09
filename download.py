"""Download the DigiAra model parameters and demo data from Zenodo.

This script uses only the Python standard library. Files are downloaded into
the Model_and_data directory beside this script and verified against the MD5
checksums published by Zenodo.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


ZENODO_RECORD_ID = "21824787"
ZENODO_DOI = "10.5281/zenodo.21824787"
BASE_URL = f"https://zenodo.org/records/{ZENODO_RECORD_ID}/files"
OUTPUT_DIR = Path(__file__).resolve().parent / "Model_and_data"

EXPECTED_FILES = {
    "Allzero_27448M5120.npy": "9694d08484e1a52ede3266c154337a2e",
    "Ara_stand_protein__esmft32_embeddings1.npy": "23c65ae2a8da4f4400bdf48c46c5fc09",
    "Ara_stand_protein_sorted_gl_entry.csv": "4398ceb28758fad377cea4f5477b97a4",
    "Bagging_Base.joblib": "8da8247711c04149a5cb469eeebfdd4e",
    "Demo_Unperturb_Transcriptomes.tsv": "85f953143dd9683b12f8ed5e4d8fedae",
    "ExtraTrees_Large.joblib": "2c07c982e49d2d7c948213b6d9a33752",
    "LGBM.joblib": "e1602bc7db325041118f816da9e4ee19",
    "Pst_DC3000_esmft32.mean_expand_27448.npy": "cebccf2b285dc86b3f76aea3ae658677",
    "Random_forest.joblib": "55acebaeb860c7fe92e263ba63a3b2b4",
    "Transformer.pt": "f102f76f09c4a6f9264acbfde16e13f1",
    "XGB.joblib": "ef88158af780e18dc643de6e2e9d10e9",
}

CHUNK_SIZE = 8 * 1024 * 1024
MAX_RETRIES = 3
USER_AGENT = "DigiAra-downloader/1.0"


class ChecksumError(RuntimeError):
    pass


def format_bytes(value: float) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TiB"


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def show_progress(
    downloaded: int,
    total: int | None,
    started_at: float,
    initial_size: int,
) -> None:
    elapsed = max(time.monotonic() - started_at, 0.001)
    speed = max(downloaded - initial_size, 0) / elapsed
    if total and total > 0:
        percent = min(downloaded / total * 100, 100.0)
        remaining = max(total - downloaded, 0)
        eta = remaining / speed if speed > 0 else 0
        message = (
            f"\r  {percent:6.2f}%  {format_bytes(downloaded)} / "
            f"{format_bytes(total)}  {format_bytes(speed)}/s  ETA {eta / 60:.1f} min"
        )
    else:
        message = f"\r  {format_bytes(downloaded)}  {format_bytes(speed)}/s"
    print(message, end="", flush=True)


def download_once(filename: str, part_path: Path) -> None:
    encoded_name = urllib.parse.quote(filename)
    url = f"{BASE_URL}/{encoded_name}?download=1"
    existing_size = part_path.stat().st_size if part_path.exists() else 0
    headers = {"User-Agent": USER_AGENT}
    if existing_size:
        headers["Range"] = f"bytes={existing_size}-"

    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response:
        status = getattr(response, "status", response.getcode())
        resumed = existing_size > 0 and status == 206
        if resumed:
            mode = "ab"
            downloaded = existing_size
        else:
            mode = "wb"
            downloaded = 0
        initial_size = downloaded

        content_range = response.headers.get("Content-Range")
        if content_range and "/" in content_range:
            total_text = content_range.rsplit("/", 1)[-1]
            total = int(total_text) if total_text.isdigit() else None
        else:
            content_length = response.headers.get("Content-Length")
            total = int(content_length) + downloaded if content_length else None

        started_at = time.monotonic()
        with part_path.open(mode) as handle:
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                show_progress(downloaded, total, started_at, initial_size)
        print()


def verify_file(path: Path, expected_md5: str) -> bool:
    print(f"  Verifying MD5: {path.name}")
    actual_md5 = md5sum(path)
    if actual_md5 != expected_md5:
        print(f"  Expected: {expected_md5}")
        print(f"  Found:    {actual_md5}")
        return False
    return True


def download_file(filename: str, expected_md5: str, force: bool) -> None:
    destination = OUTPUT_DIR / filename
    part_path = OUTPUT_DIR / f"{filename}.part"

    if destination.exists():
        if verify_file(destination, expected_md5):
            print(f"  Already complete: {filename}")
            return
        if not force:
            raise ChecksumError(
                f"Existing file failed checksum validation: {destination}\n"
                "Run with --force to replace it."
            )
        destination.unlink()

    print(f"Downloading: {filename}")
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            download_once(filename, part_path)
            if not verify_file(part_path, expected_md5):
                part_path.unlink(missing_ok=True)
                raise ChecksumError(f"Checksum validation failed for {filename}.")
            os.replace(part_path, destination)
            print(f"  Complete: {filename}")
            return
        except urllib.error.HTTPError as error:
            if error.code in {401, 403}:
                raise RuntimeError(
                    "Zenodo denied file access. The record files must be public "
                    "before this downloader can be used without authentication."
                ) from error
            if error.code == 416 and part_path.exists():
                if verify_file(part_path, expected_md5):
                    os.replace(part_path, destination)
                    print(f"  Recovered completed partial file: {filename}")
                    return
                part_path.unlink()
                print("  Partial file cannot be resumed; restarting from zero.")
            if attempt == MAX_RETRIES:
                raise
            print(f"  HTTP error {error.code}; retrying ({attempt}/{MAX_RETRIES})...")
        except (urllib.error.URLError, TimeoutError, ConnectionError, ChecksumError) as error:
            if attempt == MAX_RETRIES:
                raise
            print(f"  {error}; retrying ({attempt}/{MAX_RETRIES})...")
        time.sleep(5)


def verify_all() -> bool:
    all_valid = True
    for filename, expected_md5 in EXPECTED_FILES.items():
        path = OUTPUT_DIR / filename
        if not path.is_file():
            print(f"Missing: {filename}")
            all_valid = False
        elif verify_file(path, expected_md5):
            print(f"  OK: {filename}")
        else:
            all_valid = False
    return all_valid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            f"Download DigiAra model and demo files from Zenodo ({ZENODO_DOI})."
        )
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify existing files without downloading missing files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing file if it fails checksum validation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"DigiAra Zenodo record: https://doi.org/{ZENODO_DOI}")
    print(f"Destination: {OUTPUT_DIR}")
    print(f"Expected files: {len(EXPECTED_FILES)}")

    if args.verify_only:
        return 0 if verify_all() else 1

    try:
        for index, (filename, expected_md5) in enumerate(EXPECTED_FILES.items(), 1):
            print(f"\n[{index}/{len(EXPECTED_FILES)}]")
            download_file(filename, expected_md5, args.force)
    except KeyboardInterrupt:
        print("\nDownload interrupted. Run the command again to resume.")
        return 130
    except Exception as error:
        print(f"\nError: {error}", file=sys.stderr)
        print("Partial downloads are retained and can be resumed.", file=sys.stderr)
        return 1

    print("\nAll DigiAra model and demo files passed MD5 validation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
