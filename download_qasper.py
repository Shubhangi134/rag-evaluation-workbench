
"""Download the official QASPER development dataset."""

import json
import shutil
import tarfile
from pathlib import Path
from urllib.request import urlretrieve


DATA_URL = (
    "https://qasper-dataset.s3.us-west-2.amazonaws.com/"
    "qasper-train-dev-v0.3.tgz"
)

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "datasets" / "qasper"

DATA_FILE = DATA_DIR / "qasper-dev-v0.3.json"
ARCHIVE_FILE = DATA_DIR / "qasper-train-dev-v0.3.tgz"


def download_qasper():
    """Download and extract only the QASPER development data."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if DATA_FILE.exists():
        print("QASPER development dataset already exists.")
    else:
        print("Downloading QASPER dataset...")
        urlretrieve(DATA_URL, ARCHIVE_FILE)

        try:
            with tarfile.open(ARCHIVE_FILE, "r:gz") as archive:
                # Locate the development JSON inside the archive.
                member = next(
                    (
                        item
                        for item in archive.getmembers()
                        if Path(item.name).name
                        == DATA_FILE.name
                        and item.isfile()
                    ),
                    None,
                )

                if member is None:
                    raise FileNotFoundError(
                        "QASPER development JSON not found."
                    )

                source = archive.extractfile(member)

                if source is None:
                    raise FileNotFoundError(
                        "Could not read QASPER development JSON."
                    )

                with source:
                    with DATA_FILE.open("wb") as destination:
                        shutil.copyfileobj(source, destination)

        finally:
            ARCHIVE_FILE.unlink(missing_ok=True)

    # Verify that the downloaded file contains usable data.
    with DATA_FILE.open("r", encoding="utf-8") as file:
        papers = json.load(file)

    question_count = sum(
        len(paper.get("qas", []))
        for paper in papers.values()
    )

    print(f"Papers loaded: {len(papers)}")
    print(f"Questions available: {question_count}")
    print(f"Dataset file: {DATA_FILE}")


if __name__ == "__main__":
    download_qasper()

