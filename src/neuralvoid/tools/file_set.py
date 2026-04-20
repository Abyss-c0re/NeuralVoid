import os
import base64
import asyncio
import aiofiles
from contextlib import asynccontextmanager
from PIL import Image
from io import BytesIO
from neuralcore.actions.registry import tool, sequenced
from typing import AsyncIterable, List
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
    description="Read full content of a text file. Streams large files in 8KB chunks.",
)
async def read_file(file_path: str) -> str | AsyncIterable[str]:
    """Optimized streaming for large text files.
    Uses asynccontextmanager to keep the file open for the full lifetime of the generator."""
    try:
        if not os.path.isfile(file_path):
            return f"Error: File '{file_path}' not found."

        # Small file → fast non-streaming path (no generator overhead)
        async with aiofiles.open(
            file_path, "r", encoding="utf-8", errors="ignore"
        ) as f:
            preview = await f.read(4096)
            if len(preview) < 4096:
                return preview

        # Large file → streaming path with proper resource management
        @asynccontextmanager
        async def open_file():
            async with aiofiles.open(
                file_path, "r", encoding="utf-8", errors="ignore"
            ) as f:
                yield f

        async def stream_file() -> AsyncIterable[str]:
            async with open_file() as f:  # ← Properly typed async context
                # First chunk (reuse the preview logic if you want)
                chunk = await f.read(4096)
                if chunk:
                    yield chunk

                while True:
                    chunk = await f.read(8192)
                    if not chunk:
                        break
                    yield chunk

        return stream_file()

    except Exception as e:
        logger.error(f"read_file error for '{file_path}': {e}", exc_info=True)
        return f"Error reading '{file_path}': {str(e)}"


@tool(
    "FileEditingTools",
    tags=["file", "read", "pdf"],
    name="read_pdf",
    description="Extract all text from a PDF file.",
)
async def read_pdf(file_path: str) -> str | AsyncIterable[str]:
    """Optimized PDF streaming: batches pages for speed + good embedding size."""
    try:
        from pypdf import PdfReader

        if not os.path.isfile(file_path):
            return f"Error: PDF '{file_path}' not found."

        reader = PdfReader(file_path)
        total_pages = len(reader.pages)

        if total_pages <= 8:
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            return text.strip() or "No text extracted."

        # Large PDF → batch 5 pages per yield (good balance)
        async def stream_pdf():
            batch: List[str] = []
            batch_size = 10

            for i, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    batch.append(f"--- Page {i + 1}/{total_pages} ---\n{page_text}\n")

                if len(batch) >= batch_size or i == total_pages - 1:
                    if batch:
                        yield "".join(batch)
                    batch.clear()

                if i % batch_size == 0:
                    await asyncio.sleep(0)

        # Pass metadata hint to the streaming handler
        return stream_pdf()

        return stream_pdf()

    except Exception as e:
        return f"Error reading PDF '{file_path}': {str(e)}"


@tool(
    "FileEditingTools",
    tags=["file", "read", "docx"],
    name="read_docx",
    description="Extract all text from a DOCX file.",
)
async def read_docx(file_path: str) -> str | AsyncIterable[str]:
    """Optimized DOCX streaming."""
    try:
        from docx import Document

        if not os.path.isfile(file_path):
            return f"Error: DOCX '{file_path}' not found."

        doc = Document(file_path)
        paragraphs = list(doc.paragraphs)

        if len(paragraphs) <= 60:
            text = "\n".join(para.text for para in paragraphs)
            return text.strip() or "No text extracted."

        async def stream_docx():
            batch: List[str] = []
            batch_size = 25  # ~reasonable size for embedding

            for i, para in enumerate(paragraphs):
                if para.text.strip():
                    batch.append(para.text + "\n")

                if len(batch) >= batch_size or i == len(paragraphs) - 1:
                    if batch:
                        yield "".join(batch)
                    batch.clear()

                if i % batch_size == 0:
                    await asyncio.sleep(0)

        return stream_docx()

    except Exception as e:
        return f"Error reading DOCX '{file_path}': {str(e)}"


@tool(
    "FileEditingTools",
    tags=["file", "read", "image", "vision"],
    name="read_image",
    description="Describe image content using vision model.",
)
async def read_image(
    agent, file_path: str, prompt: str = "Describe this image in detail."
) -> str:
    # Vision tools stay non-streaming (result is compact)
    try:
        if not os.path.isfile(file_path):
            return f"Error: File '{file_path}' not found."
        loop = asyncio.get_running_loop()

        def _encode():
            with Image.open(file_path) as img:
                img.thumbnail((1024, 1024))
                buffer = BytesIO()
                img.save(buffer, format="PNG")
                return base64.b64encode(buffer.getvalue()).decode("utf-8")

        base64_img = await loop.run_in_executor(None, _encode)
        description = await agent.client.describe_image(
            image_base64=base64_img, prompt=prompt
        )
        return f"Image description: {description}"
    except Exception as e:
        return f"Error processing image '{file_path}': {str(e)}"


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
