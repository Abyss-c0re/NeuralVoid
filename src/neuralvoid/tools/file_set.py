from neuralcore.actions.manager import tool
import aiofiles


# ─────────────────────────────────────────────────────────────
# File Writing Tools
# ─────────────────────────────────────────────────────────────


@tool(
    "FileEditingTools",
    name="write_file",
    description="Create or overwrite a file with full content.",
)
def exec_write_file(file_path: str, content: str, append: bool = False) -> str:
    """Write or append content to a file."""
    mode = "a" if append else "w"
    try:
        with open(file_path, mode, encoding="utf-8") as f:
            if content and not content.endswith("\n"):
                content += "\n"
            f.write(content)

        action = "Appended to" if append else "Wrote"
        return (
            f"{action} '{file_path}' "
            f"({len(content)} characters, {content.count(chr(10)) + 1} lines)"
        )
    except Exception as e:
        return f"Error writing file '{file_path}': {str(e)}"


@tool(
    "FileEditingTools",
    name="replace_block",
    description="Replace an exact block of text in a file; safe for LLM code edits.",
)
def exec_replace_block(
    file_path: str, old_content: str, new_content: str, replace_all: bool = False
) -> str:
    """Replace one or all occurrences of a text block in a file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        count = text.count(old_content)
        if count == 0:
            return f"Error: old_content not found in '{file_path}'"
        if count > 1 and not replace_all:
            return f"Error: old_content appears {count} times (not unique). Set replace_all=True to replace all occurrences."

        new_text = text.replace(old_content, new_content, count if replace_all else 1)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_text)

        replaced = count if replace_all else 1
        return f"Replaced {replaced} occurrence(s) in '{file_path}'"
    except FileNotFoundError:
        return f"File not found: '{file_path}'"
    except Exception as e:
        return f"Error during replace: {str(e)}"


# ─────────────────────────────────────────────────────────────
# File Reading Tools
# ─────────────────────────────────────────────────────────────


@tool(
    "FileEditingTools",
    name="open_file_sync",
    description="Read a file synchronously.",
)
def open_file_sync(file_path: str) -> str:
    """Read the content of a file synchronously."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: File '{file_path}' not found."
    except Exception as e:
        return f"Error while reading file '{file_path}': {str(e)}"


@tool(
    "FileEditingTools",
    name="open_file_async",
    description="Read a file asynchronously.",
)
async def open_file_async(file_path: str) -> str:
    """Read the content of a file asynchronously."""
    try:
        async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
            return await f.read()
    except FileNotFoundError:
        return f"Error: File '{file_path}' not found."
    except Exception as e:
        return f"Error while reading file '{file_path}': {str(e)}"
