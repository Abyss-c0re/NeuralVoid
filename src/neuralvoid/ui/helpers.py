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
    """Pure Markdown version — no HTML. Works perfectly in any TUI."""
    if level == "off":
        return ""

    parts: list[str] = []

    if result is None and confirmation is None:
        # Live tool call (partial)
        if level in ("compact", "full"):
            parts.append(
                f"🔧 **Calling tool:** `{name}`\n"
                f"```json\n{json.dumps(args, indent=2, ensure_ascii=False)}\n```"
            )

    elif confirmation is not None:
        parts.append(
            f"⚠️ **Confirmation required for `{name}`**\n{confirmation}\n\n"
            f"Type **YES** to approve."
        )

    else:
        # Final result
        status = "❌" if error else "✅"
        title = f"{status} Tool `{name}` {'failed' if error else 'completed'}"

        parts.append(title)

        if args and level in ("compact", "full"):
            parts.append(
                f"**Arguments**\n```json\n{json.dumps(args, indent=2, ensure_ascii=False)}\n```"
            )

        if result is not None:
            if error:
                msg = error_message or result
                parts.append(f"**Error**\n```\n{msg}\n```")
            else:
                if level == "full":
                    parts.append(f"**Result**\n```\n{result}\n```")
                else:  # compact
                    preview = result[:400] + ("..." if len(result) > 400 else "")
                    parts.append(f"**Result preview**\n```\n{preview}\n```")

    return "\n\n" + "\n\n".join(parts) + "\n"
