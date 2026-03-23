from neuralcore.actions.manager import tool
import os
import subprocess
import shutil


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


# ---------------- CD ----------------
@tool("TerminalTools", tags=["filesystem", "directory", "navigation"], name="cd")
def exec_cd(path: str, as_dict: bool = False):
    """Change current working directory."""
    try:
        os.chdir(path)
        msg = f"Changed directory to '{os.getcwd()}'"
        return (
            {"status": "success", "cwd": os.getcwd(), "message": msg}
            if as_dict
            else msg
        )
    except FileNotFoundError:
        msg = f"cd: no such file or directory: '{path}'"
    except NotADirectoryError:
        msg = f"cd: not a directory: '{path}'"
    except PermissionError:
        msg = f"cd: permission denied: '{path}'"
    except Exception as e:
        msg = f"cd error: {str(e)}"
    return {"status": "error", "message": msg} if as_dict else msg


# ─────────────────────────────────────────────────────────────
# FILE OPERATIONS
# ─────────────────────────────────────────────────────────────


@tool("TerminalTools", tags=["filesystem", "directory", "create"], name="mkdir")
def exec_mkdir(path: str, as_dict: bool = False):
    """Create a directory."""
    try:
        os.makedirs(path, exist_ok=True)
        msg = f"Created directory '{path}'"
        return {"status": "success", "message": msg} if as_dict else msg
    except Exception as e:
        msg = f"mkdir error: {str(e)}"
        return {"status": "error", "message": msg} if as_dict else msg


@tool("TerminalTools", tags=["filesystem", "file", "copy"], name="cp")
def exec_cp(source: str, destination: str, as_dict: bool = False):
    """Copy a file or directory."""
    try:
        if os.path.isdir(source):
            shutil.copytree(source, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(source, destination)
        msg = f"Copied '{source}' → '{destination}'"
        return {"status": "success", "message": msg} if as_dict else msg
    except Exception as e:
        msg = f"cp error: {str(e)}"
        return {"status": "error", "message": msg} if as_dict else msg


@tool("TerminalTools", tags=["filesystem", "file", "move"], name="mv")
def exec_mv(source: str, destination: str, as_dict: bool = False):
    """Move or rename a file or directory."""
    try:
        shutil.move(source, destination)
        msg = f"Moved '{source}' → '{destination}'"
        return {"status": "success", "message": msg} if as_dict else msg
    except Exception as e:
        msg = f"mv error: {str(e)}"
        return {"status": "error", "message": msg} if as_dict else msg


@tool(
    "TerminalTools",
    tags=["filesystem", "file", "delete"],
    name="delete_file",
    require_confirmation=True,
)
def exec_delete_file(file_path: str, as_dict: bool = False):
    """Delete a file (requires confirmation)."""
    if not os.path.isfile(file_path):
        msg = f"File not found: '{file_path}'"
        return {"status": "error", "message": msg} if as_dict else msg
    try:
        os.remove(file_path)
        msg = f"Deleted file '{file_path}'"
        return {"status": "success", "message": msg} if as_dict else msg
    except Exception as e:
        msg = f"delete_file error: {str(e)}"
        return {"status": "error", "message": msg} if as_dict else msg


@tool(
    "TerminalTools",
    tags=["filesystem", "directory", "delete"],
    name="delete_dir",
    require_confirmation=True,
)
def exec_delete_dir(dir_path: str, as_dict: bool = False):
    """Delete a directory recursively (requires confirmation)."""
    if not os.path.isdir(dir_path):
        msg = f"Directory not found: '{dir_path}'"
        return {"status": "error", "message": msg} if as_dict else msg
    try:
        shutil.rmtree(dir_path)
        msg = f"Deleted directory '{dir_path}'"
        return {"status": "success", "message": msg} if as_dict else msg
    except Exception as e:
        msg = f"delete_dir error: {str(e)}"
        return {"status": "error", "message": msg} if as_dict else msg


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
