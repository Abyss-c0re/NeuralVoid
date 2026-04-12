from neuralcore.actions.manager import tool
from neuralcore.utils.os_info import get_os_info
import os
import shutil


# ─────────────────────────────────────────────────────────────
# FILESYSTEM / NAVIGATION (Pure Python - Cross-platform)
# ─────────────────────────────────────────────────────────────


@tool(
    "TerminalTools",
    tags=["filesystem", "list", "navigation"],
    name="list_directory",
    description="List files and folders in a directory.",
)
async def list_directory(path: str = ".") -> str:
    """List files in directory."""
    try:
        items = os.listdir(path)
        result = []
        for item in sorted(items):
            full_path = os.path.join(path, item)
            prefix = "📁 " if os.path.isdir(full_path) else "📄 "
            result.append(f"{prefix}{item}")
        return "\n".join(result) if result else "Directory empty."
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
    description="Search for files or folders by name.",
)
async def search_files(path: str = ".", name: str = "") -> str:
    """Search files by name."""
    try:
        results = []
        for root, dirs, files in os.walk(path):
            for item in files + dirs:
                if not name or name.lower() in item.lower():
                    results.append(os.path.join(root, item))
        return "\n".join(results) if results else "(no matches)"
    except Exception as e:
        return f"search_files error: {str(e)}"


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
