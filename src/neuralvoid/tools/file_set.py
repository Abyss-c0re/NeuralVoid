import os
import asyncio
import aiofiles

from typing import AsyncIterable
from neuralcore.actions.registry import tool, sequenced

from neuralvoid.utils.file_helpers import _read_file

from neuralcore.utils.logger import Logger

logger = Logger.get_logger()

# ─────────────────────────────────────────────────────────────
# File Editing Tools – Optimized for Streaming + Batching
# ─────────────────────────────────────────────────────────────


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
    tags=["file", "folder", "read"],
    name="read_folder",
    description="Recursively read folder structure and content of text files.",
)
async def read_folder(
    folder_path: str, recursive: bool = True, max_files: int = 30
) -> str | AsyncIterable[str]:
    """Optimized streaming folder output."""
    if not os.path.isdir(folder_path):
        return f"Error: Folder '{folder_path}' not found."

    async def stream_folder():
        yield f"📂 Folder: {os.path.abspath(folder_path)}\n"
        files_read = 0

        for root, dirs, files in os.walk(folder_path):
            if not recursive:
                dirs.clear()

            rel_path = os.path.relpath(root, folder_path)
            indent = "  " * (rel_path.count(os.sep) + 1) if rel_path != "." else "  "

            if rel_path != ".":
                yield f"{indent}📁 {os.path.basename(root)}/\n"

            for f in sorted(files):
                file_path = os.path.join(root, f)
                yield f"{indent}📄 {f}\n"

                # Preview only text files (limited)
                if files_read < max_files and any(
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
                    ]
                ):
                    try:
                        content_result = await read_file(file_path)

                        if isinstance(content_result, str):
                            if not content_result.startswith("Error:"):
                                preview = content_result.strip()[:280]
                                if len(content_result) > 280:
                                    preview += "..."
                                yield f"{indent}   └─ Preview: {preview}\n"
                                files_read += 1
                        elif hasattr(content_result, "__aiter__"):
                            async for chunk in content_result:
                                preview = chunk.strip()[:280]
                                if len(chunk) > 280:
                                    preview += "..."
                                yield f"{indent}   └─ Preview: {preview}\n"
                                files_read += 1
                                break
                    except Exception:
                        pass

    return stream_folder()


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
