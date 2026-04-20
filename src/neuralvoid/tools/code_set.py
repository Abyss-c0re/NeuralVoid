import os
import asyncio
from typing import List
from neuralcore.actions.registry import tool
from neuralvoid.utils.file_helpers import _read_file


# ─────────────────────────────────────────────────────────────
# COMMON CODE EXTENSIONS + IGNORE PATTERNS (lightweight, pure Python)
# ─────────────────────────────────────────────────────────────
CODE_EXTENSIONS = {
    ".py",
    ".pyi",
    ".pyx",
    ".c",
    ".cpp",
    ".cc",
    ".h",
    ".hpp",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".java",
    ".kt",
    ".go",
    ".rs",
    ".php",
    ".rb",
    ".rake",
    ".cs",
    ".swift",
    ".scala",
    ".sh",
    ".bash",
    ".sql",
    ".yaml",
    ".yml",
    ".json",
    ".toml",
    ".ini",
    ".cfg",
    ".md",
    ".rst",
}
IGNORE_DIRS = {
    ".git",
    "node_modules",
    "venv",
    "env",
    "__pycache__",
    "build",
    "dist",
    ".venv",
    ".idea",
    ".vscode",
}
IGNORE_FILES = {".DS_Store", "Thumbs.db", ".gitignore"}


# ─────────────────────────────────────────────────────────────
# CODING TOOLS (ALL ASYNC)
# ─────────────────────────────────────────────────────────────


async def _index_code_files(
    agent, folder_path: str, recursive: bool = True, max_files: int = 200
) -> dict:
    """Internal helper: indexes code files into knowledge base."""
    if not os.path.isdir(folder_path):
        return {
            "error": f"Folder '{folder_path}' not found.",
            "indexed": 0,
            "skipped": 0,
        }

    indexed = 0
    skipped = 0

    for root, dirs, files in os.walk(folder_path):
        if not recursive:
            dirs.clear()
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

        for f in files:
            if f in IGNORE_FILES or not any(
                f.lower().endswith(ext) for ext in CODE_EXTENSIONS
            ):
                skipped += 1
                continue

            if indexed >= max_files:
                break

            file_path = os.path.join(root, f)
            try:
                await agent.context_manager.index_file(agent, file_path)
                indexed += 1
            except Exception:
                skipped += 1

        if indexed >= max_files:
            break

    return {"indexed": indexed, "skipped": skipped}


async def _read_codebase_content(
    folder_path: str, recursive: bool = True, max_files: int = 100
) -> List[str]:
    """Internal helper: reads code files and builds output lines."""
    if not os.path.isdir(folder_path):
        return [f"Error: Folder '{folder_path}' not found."]

    lines = [f"📂 Codebase: {os.path.abspath(folder_path)}"]
    files_read = 0

    for root, dirs, files in os.walk(folder_path):
        if not recursive:
            dirs.clear()
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

        rel = os.path.relpath(root, folder_path)
        indent = "  " * (rel.count(os.sep) + 1) if rel != "." else ""

        for f in sorted(files):
            if f in IGNORE_FILES or not any(
                f.lower().endswith(ext) for ext in CODE_EXTENSIONS
            ):
                continue
            if files_read >= max_files:
                break

            file_path = os.path.join(root, f)
            lines.append(f"{indent}📄 {file_path}")

            try:
                # Reuse the universal read_file for consistency + streaming support
                content_result = await _read_file(
                    None, file_path
                )  # agent=None since we only want raw text

                if isinstance(content_result, str):
                    preview = content_result[:400].strip()
                    if len(content_result) > 400:
                        preview += "..."
                    lines.append(f"{indent}   └─ {preview}")
                else:
                    # It's a streaming generator → consume first chunk only for preview
                    async for chunk in content_result:
                        preview = chunk[:400].strip()
                        if len(chunk) > 400:
                            preview += "..."
                        lines.append(f"{indent}   └─ {preview}")
                        break
                files_read += 1
            except Exception:
                lines.append(f"{indent}   └─ (error reading file)")

        if files_read >= max_files:
            break

    return lines


# ─────────────────────────────────────────────────────────────
# PUBLIC TOOLS (only these are decorated)
# ─────────────────────────────────────────────────────────────


@tool(
    "CodingTools",
    tags=["code", "index", "codebase", "kb"],
    name="index_codebase",
    description="Index all code files in a folder into the knowledge base.",
)
async def index_codebase(
    agent, folder_path: str, recursive: bool = True, max_files: int = 200
) -> str:
    result = await _index_code_files(agent, folder_path, recursive, max_files)
    if "error" in result:
        return result["error"]
    return f"✅ Indexed {result['indexed']} code files from '{folder_path}' (skipped {result['skipped']})"


@tool(
    "CodingTools",
    tags=["code", "read", "codebase"],
    name="read_codebase",
    description="Read and return content of all code files in a folder (reuses universal read_file).",
)
async def read_codebase(
    folder_path: str, recursive: bool = True, max_files: int = 100
) -> str:
    lines = await _read_codebase_content(folder_path, recursive, max_files)
    return "\n".join(lines)


@tool(
    "CodingTools",
    tags=["code", "list", "structure"],
    name="list_code_files",
    description="List all code files in a folder or project.",
)
async def list_code_files(folder_path: str = ".", recursive: bool = True) -> str:
    if not os.path.isdir(folder_path):
        return f"Error: Folder '{folder_path}' not found."

    results = []
    for root, dirs, files in os.walk(folder_path):
        if not recursive:
            dirs.clear()
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

        for f in files:
            if (
                any(f.lower().endswith(ext) for ext in CODE_EXTENSIONS)
                and f not in IGNORE_FILES
            ):
                results.append(os.path.join(root, f))

    return "\n".join(sorted(results)) if results else "(no code files found)"


@tool(
    "CodingTools",
    tags=["code", "search", "codebase"],
    name="search_code",
    description="Search for text inside all code files.",
)
async def search_code(
    pattern: str, folder_path: str = ".", recursive: bool = True
) -> str:
    if not os.path.isdir(folder_path):
        return f"Error: Folder '{folder_path}' not found."

    results = []
    for root, dirs, files in os.walk(folder_path):
        if not recursive:
            dirs.clear()
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

        for f in files:
            if (
                any(f.lower().endswith(ext) for ext in CODE_EXTENSIONS)
                and f not in IGNORE_FILES
            ):
                file_path = os.path.join(root, f)
                try:
                    # Reuse read_file for consistency
                    content = await _read_file(None, file_path)
                    if isinstance(content, str):
                        for i, line in enumerate(content.splitlines(), 1):
                            if pattern.lower() in line.lower():
                                results.append(f"{file_path}:{i}: {line.strip()}")
                except Exception:
                    continue
    return "\n".join(results) if results else "(no matches)"


@tool(
    "CodingTools",
    tags=["code", "structure", "list"],
    name="get_project_structure",
    description="Show clean project folder tree with only code files.",
)
async def get_project_structure(folder_path: str = ".", max_depth: int = 3) -> str:
    if not os.path.isdir(folder_path):
        return f"Error: Folder '{folder_path}' not found."

    lines = [f"📂 {os.path.basename(os.path.abspath(folder_path))}"]
    for root, dirs, files in os.walk(folder_path):
        depth = root.count(os.sep) - os.path.abspath(folder_path).count(os.sep)
        if depth > max_depth:
            continue
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        indent = "  " * depth
        for d in sorted(dirs):
            lines.append(f"{indent}📁 {d}/")
        for f in sorted(files):
            if (
                any(f.lower().endswith(ext) for ext in CODE_EXTENSIONS)
                and f not in IGNORE_FILES
            ):
                lines.append(f"{indent}📄 {f}")
    return "\n".join(lines)


@tool(
    "CodingTools",
    tags=["git", "vcs", "version"],
    name="git_status",
    description="Show current Git repository status.",
)
async def git_status(repo_path: str = ".") -> str:
    """Run git status."""
    try:
        result = await asyncio.create_subprocess_exec(
            "git",
            "-C",
            repo_path,
            "status",
            "--short",
            "-b",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await result.communicate()
        return stdout.decode().strip() or "Clean working tree."
    except Exception as e:
        return f"git_status error: {str(e)}"


@tool(
    "CodingTools",
    tags=["git", "diff"],
    name="git_diff",
    description="Show unified diff of changes (staged or unstaged).",
)
async def git_diff(repo_path: str = ".", staged: bool = False) -> str:
    args = ["git", "-C", repo_path, "diff", "--no-color"]
    if staged:
        args.insert(-1, "--cached")
    try:
        result = await asyncio.create_subprocess_exec(
            *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, _ = await result.communicate()
        return stdout.decode() or "(no changes)"
    except Exception as e:
        return f"git_diff error: {str(e)}"


@tool(
    "CodingTools",
    tags=["git", "commit"],
    name="git_commit",
    description="Commit changes with message (safe, no auto-push).",
    require_confirmation=True,
)
async def git_commit(repo_path: str = ".", message: str = "AI-assisted changes") -> str:
    try:
        await asyncio.create_subprocess_exec("git", "-C", repo_path, "add", ".")
        result = await asyncio.create_subprocess_exec(
            "git",
            "-C",
            repo_path,
            "commit",
            "-m",
            message,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await result.communicate()
        return f"Committed: {stdout.decode().strip() or stderr.decode().strip()}"
    except Exception as e:
        return f"git_commit error: {str(e)}"


@tool(
    "CodingTools",
    tags=["git", "branch"],
    name="git_branch",
    description="List branches or create/switch branch.",
)
async def git_branch(repo_path: str = ".", new_branch: str = "") -> str:
    if new_branch:
        cmd = ["git", "-C", repo_path, "checkout", "-b", new_branch]
    else:
        cmd = ["git", "-C", repo_path, "branch", "--show-current"]
    try:
        result = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, _ = await result.communicate()
        return stdout.decode().strip()
    except Exception as e:
        return f"git_branch error: {str(e)}"
