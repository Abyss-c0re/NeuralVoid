from neuralcore.utils.file_utils import *
from neuralcore.actions.actions import Action, ActionSet


# Synchronous File Open Action
open_file_sync_action = Action(
    name="open_file_sync",
    description=(
        "Open a file synchronously and read its content. "
        "Use this for blocking file I/O operations where you need to read file contents in a single step."
    ),
    tags=[
        "file",
        "filesystem",
        "read",
        "open",
        "sync",
        "blocking",
        "text",
        "content",
        "file_io",
    ],
    parameters={
        "file_path": {
            "type": "string",
            "description": "Path to the file to open and read.",
        },
    },
    executor=open_file_sync,
    required=["file_path"],
)

# Asynchronous File Open Action
open_file_async_action = Action(
    name="open_file_async",
    description=(
        "Open a file asynchronously and read its content. "
        "Ideal for non-blocking file I/O operations where you need to perform other tasks while waiting for file reading to complete."
    ),
    tags=[
        "file",
        "filesystem",
        "read",
        "open",
        "async",
        "non-blocking",
        "text",
        "content",
        "file_io",
    ],
    parameters={
        "file_path": {
            "type": "string",
            "description": "Path to the file to open and read asynchronously.",
        },
    },
    executor=open_file_async,
    required=["file_path"],
)

write_file_action = Action(
    name="write_file",
    description=(
        "Create or overwrite a file with full content. "
        "Use this when generating code files (Python, JS, etc.), configs, "
        "or any multi-line text. LLM can output clean, properly indented code "
        "directly in the 'content' parameter — no escaping needed."
    ),
    tags=[
        "file",
        "filesystem",
        "write",
        "create",
        "save",
        "generate",
        "code",
        "script",
        "config",
        "project",
        "overwrite",
        "append",
        "developer",
        "programming",
        "text",
        "source",
    ],
    parameters={
        "file_path": {
            "type": "string",
            "description": "Path to the file to create/overwrite",
        },
        "content": {
            "type": "string",
            "description": (
                "Full file content. Paste entire code blocks here. "
                "Supports newlines, quotes, indentation — everything."
            ),
        },
        "append": {
            "type": "boolean",
            "description": "Append to existing file instead of overwriting",
            "default": False,
        },
    },
    executor=exec_write_file,
    required=["file_path", "content"],
)

replace_block_action = Action(
    name="replace_block",
    description=(
        "Surgically edit a file by replacing an exact code block or text section. "
        "Best tool for LLM code editing. "
        "1. Use 'cat' or 'read_file' first to see the code. "
        "2. Copy the EXACT old block (including indentation) into 'old_content'. "
        "3. Paste the new version into 'new_content'. "
        "Works perfectly for functions, classes, JSON blocks, etc. "
        "Safe: refuses if block is not unique unless replace_all=True."
    ),
    tags=[
        "file",
        "filesystem",
        "edit",
        "modify",
        "replace",
        "patch",
        "update",
        "refactor",
        "code",
        "function",
        "class",
        "json",
        "config",
        "source",
        "developer",
        "programming",
        "text",
        "surgical",
    ],
    parameters={
        "file_path": {"type": "string", "description": "Path to the file to edit"},
        "old_content": {
            "type": "string",
            "description": (
                "Exact text to find and replace (multi-line code block OK). "
                "Copy-paste directly from cat output — no escaping needed."
            ),
        },
        "new_content": {
            "type": "string",
            "description": (
                "New text to insert in its place (full new code block). "
                "Must have correct indentation and formatting."
            ),
        },
        "replace_all": {
            "type": "boolean",
            "description": "Replace ALL occurrences instead of just the first",
            "default": False,
        },
    },
    executor=exec_replace_block,
    required=["file_path", "old_content", "new_content"],
)


# ─────────────────────────────────────────────────────────────
# Putting it all together
# ─────────────────────────────────────────────────────────────
def get_file_actions():
    file_tools = ActionSet(
        name="FileEditingTools",
        description=(
            "Safe, targeted tools for creating and modifying file contents: "
            "write new files from scratch, or perform precise block replacements within existing files. "
            "Does **not** include deletion, moving, copying, directory creation, or navigation commands."
        ),
    )

    for act in [
        open_file_sync_action,
        open_file_async_action,
        write_file_action,
        replace_block_action,
    ]:
        file_tools.add(act)
    return file_tools
