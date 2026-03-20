from neuralcore.actions.manager import tool
import os
import subprocess
import shutil
from inspect import signature, _empty

# ─────────────────────────────────────────────────────────────
# FILESYSTEM / NAVIGATION
# ─────────────────────────────────────────────────────────────


@tool("TerminalTools", tags=["filesystem", "list", "navigation"], name="ls")
def exec_ls(path: str = ".") -> str:
    """List files in a directory."""
    return subprocess.run(
        ["ls", "-la", path], capture_output=True, text=True
    ).stdout.strip()


@tool("TerminalTools", tags=["filesystem", "directory", "current"], name="pwd")
def exec_pwd() -> str:
    """Print working directory."""
    return os.getcwd()


@tool("TerminalTools", tags=["filesystem", "directory", "navigation"], name="cd")
def exec_cd(path: str) -> str:
    """Change current working directory."""
    try:
        os.chdir(path)
        return f"Changed directory to: {os.getcwd()}"
    except FileNotFoundError:
        return f"cd: no such file or directory: '{path}'"
    except NotADirectoryError:
        return f"cd: not a directory: '{path}'"
    except PermissionError:
        return f"cd: permission denied: '{path}'"
    except Exception as e:
        return f"cd error: {str(e)}"


# ─────────────────────────────────────────────────────────────
# FILE OPERATIONS
# ─────────────────────────────────────────────────────────────


@tool("TerminalTools", tags=["filesystem", "file", "read"], name="cat")
def exec_cat(file_path: str) -> str:
    """Display file contents."""
    return subprocess.run(
        ["cat", file_path], capture_output=True, text=True
    ).stdout.strip()


@tool("TerminalTools", tags=["filesystem", "file", "create"], name="touch")
def exec_touch(file_path: str) -> str:
    """Create empty file or update timestamp."""
    subprocess.run(["touch", file_path], check=True)
    return f"Touched '{file_path}'"


@tool("TerminalTools", tags=["filesystem", "directory", "create"], name="mkdir")
def exec_mkdir(path: str) -> str:
    """Create a directory."""
    subprocess.run(["mkdir", "-p", path], check=True)
    return f"Directory created: '{path}'"


@tool("TerminalTools", tags=["filesystem", "file", "copy"], name="cp")
def exec_cp(source: str, destination: str) -> str:
    """Copy a file or directory."""
    subprocess.run(["cp", source, destination], check=True)
    return f"Copied '{source}' → '{destination}'"


@tool("TerminalTools", tags=["filesystem", "file", "move"], name="mv")
def exec_mv(source: str, destination: str) -> str:
    """Move or rename a file or directory."""
    subprocess.run(["mv", source, destination], check=True)
    return f"Moved '{source}' → '{destination}'"


@tool(
    "TerminalTools",
    tags=["filesystem", "file", "delete"],
    name="delete_file",
    require_confirmation=True,
)
def exec_delete_file(file_path: str) -> str:
    """Delete a file (requires confirmation)."""
    if not os.path.isfile(file_path):
        return f"File not found: '{file_path}'"
    os.remove(file_path)
    return f"Deleted file '{file_path}'"


@tool(
    "TerminalTools",
    tags=["filesystem", "directory", "delete"],
    name="delete_dir",
    require_confirmation=True,
)
def exec_delete_dir(dir_path: str) -> str:
    """Delete a directory recursively (requires confirmation)."""
    if not os.path.isdir(dir_path):
        return f"Directory not found: '{dir_path}'"
    shutil.rmtree(dir_path)
    return f"Deleted directory '{dir_path}'"


# ─────────────────────────────────────────────────────────────
# SEARCH & ANALYSIS
# ─────────────────────────────────────────────────────────────


@tool("TerminalTools", tags=["filesystem", "search"], name="find")
def exec_find(path: str = ".", name: str = "") -> str:
    """Find files optionally by name."""
    cmd = ["find", path]
    if name:
        cmd += ["-name", name]
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()


@tool("TerminalTools", tags=["filesystem", "analysis", "text"], name="wc")
def exec_wc(
    file_path: str, lines: bool = True, words: bool = True, chars: bool = False
) -> str:
    """Count lines, words, characters in a file."""
    flags = []
    if lines:
        flags.append("-l")
    if words:
        flags.append("-w")
    if chars:
        flags.append("-c")
    if not flags:
        flags = ["-lwc"]
    return subprocess.run(
        ["wc", *flags, file_path], capture_output=True, text=True
    ).stdout.strip()


@tool("TerminalTools", tags=["filesystem", "search", "regex"], name="grep")
def exec_grep(
    pattern: str, file_path: str, recursive: bool = False, case_sensitive: bool = True
) -> str:
    """Search for pattern in file or directory."""
    cmd = ["grep"]
    if not case_sensitive:
        cmd.append("-i")
    if recursive:
        cmd.append("-r")
    cmd += [pattern, file_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 1:
        return "(no matches)"
    elif result.returncode == 0:
        return result.stdout.strip()
    else:
        return f"grep error: {result.stderr.strip() or '(exit code ' + str(result.returncode) + ')'}"


# ─────────────────────────────────────────────────────────────
# TEXT UTILITIES
# ─────────────────────────────────────────────────────────────


@tool("TerminalTools", tags=["filesystem", "text"], name="head")
def exec_head(file_path: str, lines: int = 10) -> str:
    """Show first N lines of a file."""
    return subprocess.run(
        ["head", "-n", str(lines), file_path], capture_output=True, text=True
    ).stdout.rstrip()


@tool("TerminalTools", tags=["filesystem", "text"], name="tail")
def exec_tail(file_path: str, lines: int = 10) -> str:
    """Show last N lines of a file."""
    return subprocess.run(
        ["tail", "-n", str(lines), file_path], capture_output=True, text=True
    ).stdout.rstrip()


@tool("TerminalTools", tags=["filesystem", "text"], name="awk")
def exec_awk(file_path: str, script: str) -> str:
    """Process file with awk script."""
    cmd = ["awk", "-f", "-", file_path]
    result = subprocess.run(cmd, input=script, capture_output=True, text=True)
    return result.stdout or result.stderr


# ─────────────────────────────────────────────────────────────
# STRUCTURE & METADATA
# ─────────────────────────────────────────────────────────────


@tool("TerminalTools", tags=["filesystem", "directory"], name="tree")
def exec_tree(path: str = ".", max_depth: int = 3) -> str:
    """Display directory tree (requires tree command)."""
    try:
        result = subprocess.run(
            ["tree", "-L", str(max_depth), path], capture_output=True, text=True
        )
        return (
            result.stdout.rstrip()
            if result.returncode == 0
            else result.stderr.strip() or "tree command failed"
        )
    except FileNotFoundError:
        return "tree command not available"


@tool("TerminalTools", tags=["filesystem", "metadata"], name="stat")
def exec_stat(path: str) -> str:
    """Show file status."""
    return subprocess.run(
        ["stat", path], capture_output=True, text=True
    ).stdout.rstrip()


@tool("TerminalTools", tags=["filesystem", "metadata"], name="file")
def exec_file(path: str) -> str:
    """Determine file type."""
    return subprocess.run(["file", path], capture_output=True, text=True).stdout.strip()


@tool("TerminalTools", tags=["filesystem", "path"], name="realpath")
def exec_realpath(path: str) -> str:
    """Resolve absolute path."""
    result = subprocess.run(["realpath", path], capture_output=True, text=True)
    return (
        result.stdout.strip()
        if result.returncode == 0
        else f"realpath failed: {result.stderr.strip()}"
    )


@tool("TerminalTools", tags=["system", "command"], name="which")
def exec_which(command: str) -> str:
    """Locate command in PATH."""
    result = subprocess.run(["which", command], capture_output=True, text=True)
    return (
        result.stdout.strip()
        if result.returncode == 0
        else f"{command} not found in PATH"
    )
