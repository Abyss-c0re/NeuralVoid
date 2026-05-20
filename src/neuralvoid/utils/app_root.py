from __future__ import annotations

from pathlib import Path
from typing import Optional


def find_app_root() -> Path:
    """
    Find the root of the NeuralVoid application in a robust way.

    This function is designed to work correctly even when:
    - The command is run via `uv tool install -e .`
    - The current working directory is not inside the project
    - The package is installed in editable mode

    It returns a path such that the tool loader can find modules under
    `app_root / "tools"` or `app_root / "src" / "neuralvoid" / "tools"`.

    Priority order:
    1. Walk up from current working directory (user's intent)
    2. Walk up from the actual runtime location of the neuralvoid package
    3. Heuristics for common src-layout projects
    """
    # --- Helper to walk up looking for markers ---
    def _search(start: Path) -> Optional[Path]:
        current = start.resolve()
        for _ in range(15):  # safety limit
            # Direct hits on our tool modules (most reliable)
            if (current / "src" / "neuralvoid" / "tools" / "file_set.py").exists():
                return current / "src" / "neuralvoid"

            if (current / "tools" / "file_set.py").exists():
                return current

            # Standard project root markers
            if (current / "pyproject.toml").exists() or (current / "config.yaml").exists():
                # Prefer returning the package root if tools live under src/
                candidate = current / "src" / "neuralvoid"
                if (candidate / "tools" / "file_set.py").exists():
                    return candidate
                if (current / "tools").exists():
                    return current

            if current.parent == current:
                break
            current = current.parent
        return None

    # 1. Start from where the user is running the command
    root = _search(Path.cwd())
    if root:
        return root

    # 2. Start from where the neuralvoid package actually lives at runtime
    #    This is the key for `uv tool install -e .` and editable installs.
    try:
        import neuralvoid
        package_file = Path(neuralvoid.__file__).resolve()
        root = _search(package_file)
        if root:
            return root
    except Exception:
        pass

    # 3. Final fallback
    return Path.cwd()
