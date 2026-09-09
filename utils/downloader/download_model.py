import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

sys.path.append(str(ROOT_DIR))

from huggingface_hub import snapshot_download

from config import (
    MODEL_PATH,
    MODEL_REPO
)


def main():

    print("Downloading model...")
    print("Model:", MODEL_REPO)
    print("Path:", MODEL_PATH)


    snapshot_download(
        repo_id=MODEL_REPO,
        local_dir=str(MODEL_PATH),
        token=input("Please input your HF_TOKEN: ")
    )


    print("Model download finished.")


if __name__ == "__main__":
    main()
