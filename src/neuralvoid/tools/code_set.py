import os
import asyncio
from typing import List
from neuralcore.utils.file_helpers import _collect_text_files, _is_text_file
from neuralcore.actions.registry import tool


from neuralcore.utils.logger import Logger

logger = Logger.get_logger()


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


@tool(
    "CodingTools",
    tags=["code", "index", "codebase", "kb"],
    name="index_codebase",
    description="Index all text files in a folder into the agent's knowledge base using read_file + GetContext.",
)
async def index_codebase(
    agent, folder_path: str = ".", recursive: bool = True, max_files: int = 150
) -> str:
    """Index → triggers internal embedding/summarization via read_file."""
    files = await _collect_text_files(folder_path, recursive, max_files)

    if not files:
        return f"✅ Indexed 0 text files from '{folder_path}' (nothing readable found)"

    indexed = 0
    skipped = 0

    for file_path in files:
        try:
            await agent.manager.execute_direct("read_file", file_path=file_path)
            indexed += 1
        except Exception as e:
            logger.debug(f"index skip {file_path}: {e}")
            skipped += 1

    return f"✅ Indexed {indexed} text files from '{folder_path}' (skipped {skipped})"


@tool(
    "CodingTools",
    tags=["code", "read", "codebase"],
    name="read_codebase",
    description="Read + preview all text files in a folder. Returns tree with first 400 chars of each",
)
async def read_codebase(
    agent, folder_path: str = ".", recursive: bool = True, max_files: int = 80
) -> str:
    """Read codebase with smart previews — uses universal read_file."""
    files = await _collect_text_files(folder_path, recursive, max_files)

    if not files:
        return f"❌ No readable text files found in '{folder_path}'."

    lines = [f"📂 Codebase: {os.path.abspath(folder_path)} ({len(files)} text files)"]

    for file_path in files:
        rel = os.path.relpath(file_path, folder_path)
        depth = rel.count(os.sep)
        indent = "  " * (depth + 1)

        lines.append(f"{indent}📄 {rel}")

        try:
            content_result = await agent.manager.execute_direct(
                "read_file", file_path=file_path
            )

            preview = ""
            if isinstance(content_result, str):
                preview = content_result[:450].strip()
            else:
                # Streaming case
                async for chunk in content_result:
                    preview = chunk[:450].strip()
                    break

            if len(preview) > 400:
                preview = preview[:400] + " …"
            lines.append(f"{indent}   └─ {preview or '(empty)'}")

        except Exception as e:
            lines.append(f"{indent}   └─ ⚠️ {str(e)[:60]}")

    return "\n".join(lines)


@tool(
    "CodingTools",
    tags=["code", "list", "structure"],
    name="list_code_files",
    description="List all text files in a folder (async content detection, no extensions).",
)
async def list_code_files(folder_path: str = ".", recursive: bool = True) -> str:
    """Pure listing — fast and extension-free."""
    files = await _collect_text_files(folder_path, recursive, max_files=1000)
    if not files:
        return "(no text files found)"
    return "\n".join(sorted(files))


@tool(
    "CodingTools",
    tags=["code", "search", "codebase"],
    name="search_code",
    description="Search text inside all text files (case-insensitive). Returns file:line:match.",
)
async def search_code(
    agent, pattern: str, folder_path: str = ".", recursive: bool = True
) -> str:
    """Grep-like search across entire codebase using read_file."""
    files = await _collect_text_files(folder_path, recursive, max_files=600)
    if not files:
        return "(no text files to search)"

    results: List[str] = []
    pattern_lower = pattern.lower()

    for file_path in files:
        try:
            content = await agent.manager.execute_direct(
                "read_file", file_path=file_path
            )
            if isinstance(content, str):
                for i, line in enumerate(content.splitlines(), 1):
                    if pattern_lower in line.lower():
                        results.append(f"{file_path}:{i}: {line.strip()[:120]}")
        except Exception:
            continue

    return "\n".join(results) if results else f"(no matches for '{pattern}')"


@tool(
    "CodingTools",
    tags=["code", "structure", "tree"],
    name="get_project_structure",
    description="Beautiful folder tree of only text files (max_depth limit). Context overview.",
)
async def get_project_structure(folder_path: str = ".", max_depth: int = 4) -> str:
    """Async text-aware project tree."""
    if not os.path.isdir(folder_path):
        return f"❌ Folder not found: {folder_path}"

    lines = [f"📂 {os.path.basename(os.path.abspath(folder_path)) or 'root'}"]

    for root, dirs, files in os.walk(folder_path):
        depth = root.count(os.sep) - os.path.abspath(folder_path).count(os.sep)
        if depth > max_depth:
            continue

        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        indent = "  " * depth

        for d in sorted(dirs):
            lines.append(f"{indent}📁 {d}/")

        for f in sorted(files):
            if f in IGNORE_FILES:
                continue
            file_path = os.path.join(root, f)
            if await _is_text_file(file_path):
                lines.append(f"{indent}📄 {f}")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# GIT TOOLS (unchanged but polished)
# ─────────────────────────────────────────────────────────────


@tool(
    "CodingTools",
    tags=["git", "vcs"],
    name="git_status",
    description="git status --short -b",
)
async def git_status(repo_path: str = ".") -> str:
    try:
        proc = await asyncio.create_subprocess_exec(
            "git",
            "-C",
            repo_path,
            "status",
            "--short",
            "-b",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        return stdout.decode().strip() or "✅ Clean working tree."
    except Exception as e:
        return f"git_status error: {e}"


@tool(
    "CodingTools",
    tags=["git", "diff"],
    name="git_diff",
    description="git diff (staged or unstaged)",
)
async def git_diff(repo_path: str = ".", staged: bool = False) -> str:
    args = ["git", "-C", repo_path, "diff", "--no-color", "--unified=3"]
    if staged:
        args.insert(-1, "--cached")
    try:
        proc = await asyncio.create_subprocess_exec(
            *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, _ = await proc.communicate()
        return stdout.decode() or "(no changes)"
    except Exception as e:
        return f"git_diff error: {e}"


@tool(
    "CodingTools",
    tags=["git", "commit"],
    name="git_commit",
    description="Stage + commit (safe, no push). require_confirmation=True",
    require_confirmation=True,
)
async def git_commit(repo_path: str = ".", message: str = "AI-assisted changes") -> str:
    try:
        await asyncio.create_subprocess_exec("git", "-C", repo_path, "add", ".")
        proc = await asyncio.create_subprocess_exec(
            "git",
            "-C",
            repo_path,
            "commit",
            "-m",
            message,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        return f"✅ Committed: {stdout.decode().strip() or stderr.decode().strip()}"
    except Exception as e:
        return f"git_commit error: {e}"


@tool(
    "CodingTools",
    tags=["git", "branch"],
    name="git_branch",
    description="List or create/switch branch",
)
async def git_branch(repo_path: str = ".", new_branch: str = "") -> str:
    cmd = (
        ["git", "-C", repo_path, "checkout", "-b", new_branch]
        if new_branch
        else ["git", "-C", repo_path, "branch", "--show-current"]
    )
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, _ = await proc.communicate()
        return stdout.decode().strip()
    except Exception as e:
        return f"git_branch error: {e}"
