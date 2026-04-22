from neuralcore.actions.registry import tool
from neuralcore.utils.os_info import get_os_info
from pathlib import Path
from typing import Optional, List, Any
import os
import shutil


# ─────────────────────────────────────────────────────────────
# FILESYSTEM / NAVIGATION (Pure Python - Cross-platform)
# ─────────────────────────────────────────────────────────────


@tool(
    "TerminalTools",
    tags=["filesystem", "list", "navigation"],
    name="list_directory",
    description="List files and folders in a directory. "
    "Pass '.' (or omit the argument) to list the current working directory. "
    "Supports both relative and absolute paths.",
)
async def list_directory(path: str = ".") -> str:
    """List files and folders. Uses robust pathlib resolution + clear error messages."""
    try:
        # Robust path handling (handles ~, relative paths, symlinks, etc.)
        target = Path(path).expanduser().resolve(strict=False)

        if not target.exists():
            return f"❌ Path does not exist: {path} (resolved to {target})"

        if not target.is_dir():
            return f"❌ Not a directory: {path} (resolved to {target})"

        items = []
        for item in sorted(target.iterdir()):
            prefix = "📁 " if item.is_dir() else "📄 "
            items.append(f"{prefix}{item.name}")

        return "\n".join(items) if items else "Directory empty."

    except PermissionError:
        return f"❌ Permission denied: {path}"
    except Exception as e:
        return f"list_directory error: {str(e)}"


@tool(
    "TerminalTools",
    tags=["filesystem", "directory", "current"],
    name="get_current_directory",
    description="Get the current working directory path.",
)
async def get_current_directory() -> str:
    """Get current working directory."""
    return os.getcwd()


@tool(
    "TerminalTools",
    tags=["filesystem", "directory", "navigation"],
    name="change_directory",
    description="Change the current working directory.",
)
async def change_directory(path: str) -> str:
    """Change current working directory."""
    try:
        os.chdir(path)
        return f"Changed directory to '{os.getcwd()}'"
    except FileNotFoundError:
        return f"change_directory: no such file or directory: '{path}'"
    except NotADirectoryError:
        return f"change_directory: not a directory: '{path}'"
    except PermissionError:
        return f"change_directory: permission denied: '{path}'"
    except Exception as e:
        return f"change_directory error: {str(e)}"


# ─────────────────────────────────────────────────────────────
# FILE/DIR OPERATIONS
# ─────────────────────────────────────────────────────────────


@tool(
    "TerminalTools",
    tags=["filesystem", "directory", "create"],
    name="create_directory",
    description="Create a new directory (and parent folders if needed).",
)
async def create_directory(path: str) -> str:
    """Create directory."""
    try:
        os.makedirs(path, exist_ok=True)
        return f"Created directory '{path}'"
    except Exception as e:
        return f"create_directory error: {str(e)}"


@tool(
    "TerminalTools",
    tags=["filesystem", "file", "copy"],
    name="copy",
    description="Copy a file or directory.",
)
async def copy(source: str, destination: str) -> str:
    """Copy file or directory."""
    try:
        if os.path.isdir(source):
            shutil.copytree(source, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(source, destination)
        return f"Copied '{source}' → '{destination}'"
    except Exception as e:
        return f"copy error: {str(e)}"


@tool(
    "TerminalTools",
    tags=["filesystem", "file", "move"],
    name="move",
    description="Move or rename a file or directory.",
)
async def move(source: str, destination: str) -> str:
    """Move or rename file/directory."""
    try:
        shutil.move(source, destination)
        return f"Moved '{source}' → '{destination}'"
    except Exception as e:
        return f"move error: {str(e)}"


@tool(
    "TerminalTools",
    tags=["filesystem", "file", "delete"],
    name="delete_file",
    require_confirmation=True,
    description="Delete a single file (requires confirmation).",
)
async def delete_file(file_path: str) -> str:
    """Delete a file."""
    try:
        if not os.path.isfile(file_path):
            return f"File not found: '{file_path}'"
        os.remove(file_path)
        return f"Deleted file '{file_path}'"
    except Exception as e:
        return f"delete_file error: {str(e)}"


@tool(
    "TerminalTools",
    tags=["filesystem", "directory", "delete"],
    name="delete_directory",
    require_confirmation=True,
    description="Delete a directory and all its contents (requires confirmation).",
)
async def delete_directory(dir_path: str) -> str:
    """Delete directory recursively."""
    try:
        if not os.path.isdir(dir_path):
            return f"Directory not found: '{dir_path}'"
        shutil.rmtree(dir_path)
        return f"Deleted directory '{dir_path}'"
    except Exception as e:
        return f"delete_directory error: {str(e)}"


# ─────────────────────────────────────────────────────────────
# SEARCH
# ─────────────────────────────────────────────────────────────


@tool(
    "TerminalTools",
    tags=["filesystem", "search"],
    name="search_files",
    description=(
        "Recursively search for files and/or directories by name or glob pattern. "
        "Supports wildcards like '*.py', 'config*', or simple partial names. "
        "Returns full absolute paths, sorted for readability. "
        "Built for safe use in agents and multi-agent workflows."
    ),
    require_confirmation=False,
)
async def search_files(
    path: str = ".",
    name: str = "*",
    max_results: Optional[Any] = 200,  # Accept Any to handle str/int from LLM
    include_dirs: bool = True,
    include_files: bool = True,
) -> str:
    """
    Generic, reusable filesystem search tool for NeuralCore.
    No client-specific logic — pure utility.
    """
    try:
        # Safe conversion of max_results (handles str from LLM, None, or int)
        if max_results is None:
            max_results = None
        else:
            try:
                max_results = int(max_results)
                if max_results <= 0:
                    max_results = None
            except ValueError:
                max_results = 200  # fallback safe default

        search_path = Path(path).expanduser().resolve()
        if not search_path.exists():
            return f"Error: Path does not exist → {search_path}"

        if not search_path.is_dir():
            return f"Error: Path is not a directory → {search_path}"

        results: List[str] = []
        # Normalize pattern for rglob
        pattern = (
            f"*{name}*" if name and "*" not in name and "?" not in name else name or "*"
        )

        for item in search_path.rglob(pattern):
            if (item.is_file() and include_files) or (item.is_dir() and include_dirs):
                results.append(str(item))

            # Safe limit check
            if max_results is not None and len(results) >= max_results:
                results.append(
                    f"... (truncated — reached max_results limit of {max_results})"
                )
                break

        if not results:
            return f"(no matches for pattern '{name}' in '{path}')"

        # Return clean, sorted, human+agent friendly output
        return "\n".join(sorted(results))

    except PermissionError as e:
        return f"Permission denied: {str(e)}"
    except OSError as e:
        return f"Filesystem error: {type(e).__name__} - {str(e)}"
    except Exception as e:
        return f"search_files error: {type(e).__name__} - {str(e)}"


@tool(
    "TerminalTools",
    tags=["filesystem", "search", "regex"],
    name="search_text",
    description="Search for text pattern inside files.",
)
async def search_text(pattern: str, file_path: str, recursive: bool = False) -> str:
    """Search text in file(s)."""
    try:
        results = []
        paths = [file_path]
        if recursive and os.path.isdir(file_path):
            paths = []
            for root, _, files in os.walk(file_path):
                paths.extend(os.path.join(root, f) for f in files)
        for p in paths:
            if os.path.isfile(p):
                try:
                    with open(p, "r", encoding="utf-8", errors="ignore") as f:
                        for i, line in enumerate(f, 1):
                            if pattern in line:
                                results.append(f"{p}:{i}:{line.strip()}")
                except Exception:
                    continue
        return "\n".join(results) if results else "(no matches)"
    except Exception as e:
        return f"search_text error: {str(e)}"


# ─────────────────────────────────────────────────────────────
# METADATA
# ─────────────────────────────────────────────────────────────


@tool(
    "TerminalTools",
    tags=["filesystem", "metadata"],
    name="get_file_info",
    description="Get detailed information about a file or directory.",
)
async def get_file_info(path: str) -> str:
    """Get file or directory information."""
    try:
        if not os.path.exists(path):
            return f"Path not found: '{path}'"
        size = os.path.getsize(path) if os.path.isfile(path) else "-"
        mtime = os.path.getmtime(path)
        is_dir = os.path.isdir(path)
        return f"Path: {path}\nType: {'Directory' if is_dir else 'File'}\nSize: {size} bytes\nModified: {mtime}"
    except Exception as e:
        return f"get_file_info error: {str(e)}"


@tool(
    "TerminalTools",
    tags=["os", "system", "environment", "info"],
    name="get_os_info",
    description="Returns detailed, user-friendly information about the current operating system "
    "(Linux distro with PRETTY_NAME, or fallback to platform details on other OSes). "
    "Useful for adapting commands, choosing package managers, or reporting environment.",
)
async def os_info() -> str:
    """
    Async wrapper around the imported _get_distro_info().
    Keeps NeuralCore clean and reusable.
    """
    try:
        result = await get_os_info()
        return str(result)
    except Exception as e:
        import platform

        return f"OS detection error: {str(e)}. Fallback: {platform.system()} {platform.release()} ({platform.machine()})"
