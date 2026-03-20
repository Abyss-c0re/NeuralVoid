import json

from typing import Optional

def _format_block(title: str, body: str, icon: str = "") -> str:
    title_line = f"**{icon} {title}**".strip()
    body = str(body).strip()
    return f"\n\n---\n{title_line}\n\n{body}\n---\n"

def _format_text(text: str) -> str:
    lines = text.split("\n")
    formatted = []
    in_code = False

    for line in lines:
        line = line.rstrip()

        if line.startswith("```"):
            in_code = not in_code
            formatted.append(line)
            continue

        if in_code:
            formatted.append(line)
            continue

        if line.strip().startswith(tuple(f"{i}." for i in range(1, 10))):
            num, rest = line.split(".", 1)
            formatted.append(f"**{num}.** {rest.strip()}")
        else:
            formatted.append(line)

    if in_code:
        formatted.append("```")

    return "\n".join(formatted).strip()

def _build_tool_markdown(
    name: str,
    args: dict,
    level: str,
    result: Optional[str] = None,
    confirmation: Optional[str] = None,
    error: bool = False,
    error_message: Optional[str] = None,
) -> str:
    """Pure function: returns markdown to append. No UI calls."""

    if level == "off":
        return ""

    parts: list[str] = []

    if result is None and confirmation is None:
        if level in ("compact", "full"):
            parts.append(
                f"\n\n🔧 **Calling tool:** `{name}`\n"
                f"```json\n{json.dumps(args, indent=2)}\n```"
            )

    if confirmation is not None:
        parts.append(
            f"\n⚠ **Confirmation required**\n{confirmation}\n\n"
            f"Type **YES** to approve."
        )

    if result is not None:
        if error:
            if level in ("compact", "full"):
                msg = error_message or result
                parts.append(f"\n❌ **Tool `{name}` failed**\n```\n{msg}\n```")
        else:
            if level == "full":
                parts.append(f"\n✅ **Result from `{name}`:**\n```\n{result}\n```")

    return "".join(parts)