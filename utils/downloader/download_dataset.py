import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

sys.path.append(str(ROOT_DIR))

from datasets import load_dataset

from config import (
    DATASET_REPO,
    DATASET_TYPE,
    RAW_DATASET_PATH
)

def main():

    print("Downloading dataset...")
    print("Dataset:", DATASET_REPO, DATASET_TYPE)
    print("Path:", RAW_DATASET_PATH)

    kwargs = {}

    if DATASET_TYPE:
        kwargs["name"] = DATASET_TYPE

    dataset = load_dataset(
        DATASET_REPO,
        **kwargs,
    )


    RAW_DATASET_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset.save_to_disk(
        str(RAW_DATASET_PATH)
    )

    print("Dataset download finished.")

if __name__ == "__main__":
    main()
