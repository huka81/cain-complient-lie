from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
SRC_DIR: Path = REPO_ROOT / "src"
SCRIPTS_DIR: Path = REPO_ROOT / "scripts"
RESULTS_DIR: Path = REPO_ROOT / "results"
FIGURES_DIR: Path = REPO_ROOT / "figures"
PAPER_DIR: Path = REPO_ROOT / "paper"
EXPERIMENTS_DIR: Path = REPO_ROOT


def ensure_import_paths() -> None:
    for path in (SRC_DIR, SCRIPTS_DIR):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


# dir_core is installed via requirements.txt (dir_core==0.4.0)
ensure_import_paths()
