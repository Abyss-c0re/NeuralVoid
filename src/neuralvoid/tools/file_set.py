import os
import asyncio
import aiofiles

from typing import AsyncIterable, List
from neuralcore.actions.registry import tool, sequenced

from neuralcore.utils.file_helpers import _read_file

from neuralcore.utils.logger import Logger

logger = Logger.get_logger()

# ─────────────────────────────────────────────────────────────
# File Editing Tools – Optimized for Streaming + Batching
# ─────────────────────────────────────────────────────────────

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


@tool(
    "FileEditingTools",
    tags=["file", "write", "edit"],
    name="write_file",
    description="Write or append text content to a file.",
)
async def write_file(file_path: str, content: str, append: bool = False) -> str:
    mode = "a" if append else "w"
    try:
        async with aiofiles.open(file_path, mode, encoding="utf-8") as f:
            if content and not content.endswith("\n"):
                content += "\n"
            await f.write(content)
        action = "Appended to" if append else "Wrote"
        return f"{action} '{file_path}' ({len(content)} chars)"
    except Exception as e:
        return f"Error writing '{file_path}': {str(e)}"


@tool(
    "FileEditingTools",
    tags=["file", "edit"],
    name="replace_block",
    description="Replace exact text block inside a file.",
)
async def replace_block(
    file_path: str, old_content: str, new_content: str, replace_all: bool = False
) -> str:
    try:
        async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
            text = await f.read()
        count = text.count(old_content)
        if count == 0:
            return f"Error: old_content not found in '{file_path}'"
        if count > 1 and not replace_all:
            return f"Error: old_content appears {count} times. Set replace_all=True."
        new_text = text.replace(old_content, new_content, count if replace_all else 1)
        async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
            await f.write(new_text)
        replaced = count if replace_all else 1
        return f"Replaced {replaced} occurrence(s) in '{file_path}'"
    except FileNotFoundError:
        return f"File not found: '{file_path}'"
    except Exception as e:
        return f"Error replacing in '{file_path}': {str(e)}"


@tool(
    "FileEditingTools",
    tags=["file", "read"],
    name="read_file",
    description="Universal file reader. Automatically detects type and streams large files. "
    "Supports: .txt, .md, .py, .pdf, .docx, .png, .jpg, .jpeg, .webp, etc.",
)
async def read_file(
    agent, file_path: str, image_prompt: str = "Describe this image in detail."
) -> str | AsyncIterable[str]:

    return await _read_file(agent, file_path, image_prompt)


@tool(
    "FileEditingTools",
    tags=["file", "read", "batch"],
    name="read_multiple_files",
    description="Read multiple files at once using the universal read_file. "
    "Optionally triggers summarization/indexing of the results.",
)
async def read_multiple_files(
    agent,
    files: List[str],
    summary: bool = False,
) -> str:
    """Reads multiple files via the universal read_file tool.
    If summary=True, it triggers GetContext to create a consolidated summary."""
    if not files:
        return "No files provided."

    indexed_files = []
    errors = []

    for file_path in files:
        try:
            # Use execute_direct to leverage the full universal read_file (streaming + dispatch)
            await agent.manager.execute_direct(
                "read_file",
                file_path=file_path,
            )

            file_name = os.path.basename(file_path)
            indexed_files.append(file_name)

        except Exception as e:
            errors.append(f"{os.path.basename(file_path)}: {str(e)}")

    # Final response
    if summary and indexed_files:
        try:
            # Trigger summarization over the just-read files
            summary_result = await agent.manager.execute_direct(
                "GetContext",  # assuming this tool exists in your registry
                query=" ".join(indexed_files),  # or better query if you have one
            )
            return (
                f"✅ Read and summarized {len(indexed_files)} files:\n{summary_result}"
            )
        except Exception as e:
            logger.warning(f"Summary failed after reading files: {e}")

    if errors:
        error_msg = f" ({len(errors)} errors)" if errors else ""
        return f"✅ Indexed {len(indexed_files)} files{error_msg}: " + ", ".join(
            indexed_files
        )

    return "✅ Files indexed: " + ", ".join(indexed_files)


@tool(
    "FileEditingTools",
    tags=["file", "folder", "read"],
    name="read_folder",
    description="Recursively read all text/code files in a folder using read_multiple_files. "
    "Returns a simple confirmation that the folder content was indexed.",
)
async def read_folder(
    agent,
    folder_path: str,
    recursive: bool = True,
    max_files: int = 50,  # reasonable default to prevent explosion
) -> str:
    """Collects all readable files in the folder and delegates to read_multiple_files."""
    if not os.path.isdir(folder_path):
        return f"Error: Folder '{folder_path}' not found."

    files_to_read: List[str] = []
    files_read = 0

    for root, dirs, files in os.walk(folder_path):
        if not recursive:
            dirs.clear()

        # Skip ignored directories
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

        for f in sorted(files):
            file_path = os.path.join(root, f)

            # Only read text/code-like files (you can expand this list)
            if any(
                f.lower().endswith(ext)
                for ext in [
                    ".txt",
                    ".md",
                    ".py",
                    ".js",
                    ".ts",
                    ".json",
                    ".yaml",
                    ".yml",
                    ".sh",
                    ".html",
                    ".css",
                    ".rst",
                    ".toml",
                    ".ini",
                    ".cfg",
                ]
            ):
                if files_read >= max_files:
                    break
                files_to_read.append(file_path)
                files_read += 1

        if files_read >= max_files:
            break

    if not files_to_read:
        return (
            f"Folder '{os.path.basename(folder_path)}' contains no readable text files."
        )

    # Delegate to the new batch reader
    folder_name = os.path.basename(folder_path) or "root"
    try:
        await agent.manager.execute_direct(
            "read_multiple_files",
            files=files_to_read,
            summary=False,
        )
        return f"✅ Content of folder '{folder_name}' was indexed ({len(files_to_read)} files)."

    except Exception as e:
        logger.error(f"read_folder failed for '{folder_path}': {e}", exc_info=True)
        return f"Error processing folder '{folder_name}': {str(e)}"


@tool(
    "FileEditingTools",
    tags=["file", "edit", "diff"],
    name="apply_diff",
    description="Apply a unified diff patch to a file (safe preview first).",
    require_confirmation=True,
)
async def apply_diff(file_path: str, diff_content: str) -> str:
    try:
        check = await asyncio.create_subprocess_exec(
            "git",
            "apply",
            "--check",
            "--unidiff-zero",
            "-",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
        )
        await check.communicate(diff_content.encode())

        result = await asyncio.create_subprocess_exec(
            "git",
            "apply",
            "--unidiff-zero",
            "-",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
        )
        await result.communicate(diff_content.encode())
        return f"Successfully applied diff to '{file_path}'"
    except Exception as e:
        return f"apply_diff failed: {str(e)}"


@tool(
    "FileEditingTools",
    tags=["file", "edit", "regex"],
    name="regex_replace",
    description="Regex-based find and replace in a file (with dry-run option).",
)
async def regex_replace(
    file_path: str, pattern: str, replacement: str, dry_run: bool = True
) -> str:
    import re

    try:
        async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
            content = await f.read()
        new_content, count = re.subn(pattern, replacement, content)
        if dry_run:
            return f"Dry-run: would replace {count} occurrence(s) in '{file_path}'"
        async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
            await f.write(new_content)
        return f"Replaced {count} occurrence(s) using regex in '{file_path}'"
    except Exception as e:
        return f"regex_replace error: {str(e)}"


@sequenced(
    name="find_and_read_file",
    description="Search for a file by name and automatically read the first match",
    set_name="FileEditingTools",
    tags=["file", "search", "read", "workflow"],
    propagate=False,
    output_from="read_file",
    dependencies={
        "search_files": {"name": "input"},
        "read_file": {"file_path": "search_files"},
    },
    steps=["search_files", "read_file"],
)
def find_and_read_file():
    pass
