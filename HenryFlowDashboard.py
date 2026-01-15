from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from HenryGPT.pipeline import run_pipeline as _run_pipeline


def run_pipeline() -> None:
    _run_pipeline()


if __name__ == "__main__":
    run_pipeline()