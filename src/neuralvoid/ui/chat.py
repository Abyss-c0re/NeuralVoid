import asyncio
import json

from typing import Optional, Union, List, Any, Dict

from textual.app import App, ComposeResult
from textual.widgets import Input, Markdown
from textual.containers import VerticalScroll
from textual.binding import Binding

from neuralcore.core.client import LLMClient
from neuralcore.actions.actions import ActionSet
from neuralcore.actions.manager import DynamicActionManager

from neuralvoid.ui.rendering import set_renderer_app, get_renderer

from neuralcore.cognition.memory import ContextManager
from neuralcore.agents.agent_core import AgentRunner
from neuralcore.utils.logger import Logger

ToolProvider = Union[ActionSet, DynamicActionManager, list[dict[str, Any]]]

logger = Logger.get_logger()

# ============================================================
# Message widget
# ============================================================


class Message(Markdown):
    """Markdown-rendered chat message."""

    def __init__(self, role: str, content: str = ""):
        self.role = role
        self.buffer = content
        super().__init__(self.render_markdown())

    def render_markdown(self) -> str:

        if self.role == "user":
            prefix = "🧑 **You**: "
        elif self.role == "assistant":
            prefix = "🤖 **Assistant**: "
        elif self.role == "system":
            prefix = "💻 **System**: "
        else:
            prefix = ""

        return prefix + self.buffer


# ============================================================
# Chat container
# ============================================================


class ChatView(VerticalScroll):
    """Scrollable chat history."""

    def add(self, widget):
        self.mount(widget)
        self.scroll_end()


# ============================================================
# Main App
# ============================================================


class LLMChatApp(App):
    client: LLMClient
    system_prompt: Optional[str]
    waiting_for_confirmation: bool = False
    pending_confirmation: Optional[dict] = None
    _current_stream_task: asyncio.Task | None = None
    UPDATE_INTERVAL = 0.01

    BINDINGS = [
        Binding("ctrl+l", "clear_chat", "Clear"),
        Binding("ctrl+c", "quit", "Quit"),
        Binding("escape", "stop_stream", "Stop generation", show=True, priority=True),
    ]

    CSS = """
    Screen {
        layout: vertical;
    }

    #chat {
        height: 1fr;
        padding: 1;
    }

    Input {
        dock: bottom;
    }
    """

    def __init__(
        self,
        client: LLMClient,
        system_prompt: Optional[str] = None,
        tools: Optional[
            Union[ActionSet, "DynamicActionManager", List[Dict[str, Any]]]
        ] = None,
        context_manager: Optional[ContextManager] = None,
        tool_rendering: Optional[str] = "off",
        max_iterations: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        super().__init__()

        self.client = client
        self.system_prompt = system_prompt
        self.tools = tools
        self.context_manager = context_manager
        self.conversation = []

        self.rendering = get_renderer()
        self.tool_rendering = tool_rendering  # off | info | full
        self.max_iterations = (
            max_iterations
            if max_iterations is not None
            else getattr(client, "max_iterations", 20)
        )
        self.temperature = (
            temperature
            if temperature is not None
            else getattr(client, "temperature", 0.7)
        )
        self.max_tokens = (
            max_tokens
            if max_tokens is not None
            else getattr(client, "max_tokens", 32000)
        )

    def compose(self) -> ComposeResult:
        self.chat = ChatView(id="chat")
        yield self.chat
        yield Input(placeholder="Ask something...")

    async def on_mount(self):
        set_renderer_app(self, Message)
        await self.rendering.start_worker()

        self.chat.add(
            Message(
                "assistant",
                f"Connected to **{self.client.model}**\n\n"
                "Type a message and press **Enter**.\n"
                "Commands: **stop** / **cancel** → stop current stream\n"
                "**exit** → close app",
            )
        )

        # ← NEW: Auto-focus input on startup
        self.query_one(Input).focus()

    def action_stop_stream(self) -> None:
        if self._current_stream_task is None:
            self.notify("No active generation to stop.", timeout=2.5)
            return

        if not self._current_stream_task.done():
            cancelled = self._current_stream_task.cancel("user requested stop via Esc")
            if cancelled:
                self.notify(
                    "🛑 Generation cancelled (Esc pressed)",
                    severity="warning",
                    timeout=4,
                )
            else:
                self.notify("Task already finishing …")
        else:
            self.notify("Generation already finished.")

    async def _handle_confirmation_response(self, user_input: str) -> bool:
        """
        Returns True if this input was treated as confirmation answer
        """
        if not self.waiting_for_confirmation:
            return False

        approved = user_input.strip().upper() in {"YES", "Y", "OK", "CONFIRM"}

        info = self.pending_confirmation
        if not info:
            self.waiting_for_confirmation = False
            return False

        tool_call_id = info["tool_call_id"]
        name = info["name"]
        args = info["args"]
        action = info["action"]  # ← the Action object
        assistant_msg = info["assistant_msg"]
        original_tool_calls = info["tool_calls"]

        if approved:
            try:
                real_executor = action.executor  # ← this is the callable we want

                if asyncio.iscoroutinefunction(real_executor):
                    result = await real_executor(**args)
                else:
                    result = real_executor(**args)

            except Exception as exc:
                result = f"Error during confirmed execution: {exc}"

            content = str(result)
        else:
            content = "User denied the action."

        # Now we can finally send the tool result back to the model
        tool_result_msg = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": name,
            "content": content,
        }

        # Append the assistant's tool call request + our result
        self.conversation.append(
            {"role": "assistant", "tool_calls": original_tool_calls}
        )
        self.conversation.append(tool_result_msg)

        # Show feedback in UI
        feedback = f"→ Action **{name}** was {'approved' if approved else 'denied'}.\nResult: {content}"
        assistant_msg.buffer += f"\n\n{feedback}"
        assistant_msg.update(assistant_msg.render_markdown())
        self.chat.scroll_end()

        # Reset confirmation state
        self.waiting_for_confirmation = False
        self.pending_confirmation = None

        # Continue LLM generation with the new tool result in history
        asyncio.create_task(
            self.stream_llm(
                prompt="",  # we continue from history
                message=assistant_msg,  # continue in same bubble or new one
                tools=self.tools,
            )
        )

        return True

    async def on_input_submitted(self, event: Input.Submitted):
        value = event.value.strip()
        event.input.value = ""

        if not value:
            return

        cmd = value.lower()

        if cmd in ("stop", "cancel"):
            self.action_stop_stream()  # ← reuse the same logic
            return

        elif cmd == "exit":
            self.chat.add(Message("system", "👋 Exiting..."))
            await asyncio.sleep(0.3)
            self.exit()
            return

        if await self._handle_confirmation_response(value):
            return

        # Normal message flow
        self.chat.add(Message("user", value))
        self.conversation.append({"role": "user", "content": value})

        assistant_msg = Message("assistant", "")
        self.chat.add(assistant_msg)

        # ── The important change ────────────────────────────────────────
        task = asyncio.create_task(
            self.stream_llm(value, assistant_msg, self.tools), name="llm-stream"
        )
        self._current_stream_task = task

        # Optional: clean up reference when done
        def on_stream_done(fut: asyncio.Future) -> None:
            if fut is self._current_stream_task:
                self._current_stream_task = None

        task.add_done_callback(on_stream_done)

    def _build_tool_markdown(
        self,
        name: str,
        args: dict,
        result: Optional[str] = None,
        confirmation: Optional[str] = None,
        error: bool = False,
        error_message: Optional[str] = None,
    ) -> str:
        """Pure function: returns markdown to append. No UI calls."""
        level = getattr(self, "tool_rendering", "off")
        if level == "off":
            return ""

        parts: list[str] = []

        if result is None and confirmation is None:
            if level in ("info", "full"):
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
                if level in ("info", "full"):
                    msg = error_message or result
                    parts.append(f"\n❌ **Tool `{name}` failed**\n```\n{msg}\n```")
            else:
                if level == "full":
                    parts.append(f"\n✅ **Result from `{name}`:**\n```\n{result}\n```")

        return "".join(parts)

    async def stream_llm(
        self,
        prompt: str,
        message: Message,
        tools: Optional[ToolProvider] = None,
    ) -> None:
        import time

        logger.info("🚀 stream_llm START | prompt=%s...", prompt[:150])

        current_msg: Message = message
        text_buffer: str = ""
        self._last_stream_update = time.time() - 0.1

        runner = AgentRunner(
            client=self.client,
            max_iterations=self.max_iterations,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        async def _ui_update(new_buffer: str, immediate: bool = False) -> None:
            nonlocal current_msg
            now = time.time()
            if not immediate and now - self._last_stream_update < self.UPDATE_INTERVAL:
                return
            if current_msg.buffer != new_buffer:
                current_msg.buffer = new_buffer
                current_msg.update(current_msg.render_markdown())
                self.chat.scroll_end(animate=False)
                self._last_stream_update = now

        async def _new_assistant_bubble() -> Message:
            new_bubble = Message("assistant", "")
            self.chat.add(new_bubble)
            self.chat.scroll_end(animate=False)
            return new_bubble

        try:
            async for event_type, payload in runner.run(
                user_prompt=prompt,
                tools=tools or [],
                system_prompt=self.system_prompt or "",
                context_manager=self.context_manager,
            ):
                # ── Content streaming ────────────────────────────────────────
                if event_type == "content_delta":
                    text_buffer += payload
                    await _ui_update(text_buffer)

                # ── Tool lifecycle events ─────────────────────────────────────
                elif event_type == "tool_start":
                    md = self._build_tool_markdown(
                        name=payload["name"],
                        args=payload.get("args", {}),
                    )
                    text_buffer += md
                    await _ui_update(text_buffer, immediate=True)

                elif event_type == "tool_result":
                    md = self._build_tool_markdown(
                        name=payload["name"],
                        args=payload.get("args", {}),
                        result=str(payload.get("result", "")),
                        error=payload.get("error", False),
                    )
                    text_buffer += md
                    await _ui_update(text_buffer, immediate=True)

                elif event_type == "tool_calls":
                    count = len(payload)
                    text_buffer += (
                        f"\n\n**Calling {count} tool{'s' if count != 1 else ''}...**\n"
                    )
                    await _ui_update(text_buffer, immediate=True)

                # elif event_type == "tool_call_delta":
                #     func = payload.get("function", {})
                #     name = func.get("name", "unknown")
                #     args_part = func.get("arguments", "")[:60]
                #     text_buffer += f"\n🔧 **Tool**: {name}  (args: {args_part}…)"
                #     await _ui_update(text_buffer)

                # ── Confirmation handling ─────────────────────────────────────
                elif event_type == "needs_confirmation":
                    md = self._build_tool_markdown(
                        name=payload["name"],
                        args=payload["args"],
                        confirmation=payload.get("preview", ""),
                    )
                    text_buffer += (
                        f"\n\n{md}\n\n**Requires confirmation — type YES to approve**"
                    )
                    await _ui_update(text_buffer, immediate=True)

                    self.waiting_for_confirmation = True
                    self.pending_confirmation = {**payload}
                    return  # ← important: pause streaming here

                # ── Agent phases / reflection / continuation ──────────────────
                elif event_type in (
                    "reflection_triggered",
                    "progress_summary",
                    "continuation_query_added",
                ):
                    # Finalize current bubble before switching
                    if text_buffer.strip():
                        current_msg.buffer = text_buffer.strip()
                        current_msg.update(current_msg.render_markdown())

                    current_msg = await _new_assistant_bubble()
                    text_buffer = ""

                    # if event_type == "reflection_triggered":
                    #     text_buffer += f"\n\n---\n**🤔 Agent Reflection**\n{str(payload).strip()}\n---\n\n"
                    if event_type == "progress_summary":
                        text_buffer += (
                            f"\n\n**📋 First-run Summary**\n{str(payload).strip()}\n\n"
                        )
                    elif event_type == "continuation_query_added":
                        text_buffer += f"**🔄 Continuing — new sub-goal:**\n{str(payload)[:500]}...\n\n"

                    await _ui_update(text_buffer, immediate=True)

                # ── Final outputs & summaries ─────────────────────────────────
                elif event_type == "final_summary":
                    text_buffer = str(payload).strip()
                    await _ui_update(text_buffer, immediate=True)
                    self.conversation.append(
                        {"role": "assistant", "content": text_buffer}
                    )

                elif event_type == "review_phase":
                    if isinstance(payload, dict) and "summary" in payload:
                        summary_text = payload["summary"].strip()

                        # Optional: try to detect and format numbered questions nicely
                        lines = summary_text.split("\n")
                        formatted = []
                        in_code = False

                        for line in lines:
                            line = line.rstrip()
                            if line.startswith("```") or line.startswith("bash"):
                                in_code = not in_code
                                formatted.append(
                                    "```bash" if "bash" in line.lower() else "```"
                                )
                                continue

                            if in_code:
                                formatted.append(line)
                                continue

                            # Numbered items get bold labels
                            if line.strip().startswith(("1.", "2.", "3.")):
                                num, rest = line.split(".", 1)
                                rest = rest.strip()
                                formatted.append(f"**{num}.** {rest}")
                            elif line.strip():
                                formatted.append(line)
                            else:
                                formatted.append("")  # keep blank lines

                        if in_code:
                            formatted.append("```")

                        nice_summary = "\n".join(formatted).strip()

                        display_text = (
                            f"\n\n---\n**🔍 Review Phase**\n\n{nice_summary}\n---\n\n"
                        )
                    else:
                        display_text = (
                            f"\n\n---\n**🔍 Review Phase**\n{str(payload)}\n---\n\n"
                        )

                    text_buffer += display_text
                    await _ui_update(text_buffer, immediate=True)

                elif event_type == "final_answer":
                    text_buffer = str(payload).strip()
                    await _ui_update(text_buffer, immediate=True)
                    self.conversation.append(
                        {"role": "assistant", "content": text_buffer}
                    )

                # ── Meta / status / errors ────────────────────────────────────
                elif event_type == "step_start":
                    iter_num = payload.get("iteration", "?")
                    text_buffer += f"\n\n---\n**🔄 Iteration {iter_num} started**\n"
                    await _ui_update(text_buffer)

                elif event_type == "llm_finish":
                    text_buffer += "\n\n**✅ LLM generation finished**\n"
                    await _ui_update(text_buffer)

                elif event_type == "system":
                    text_buffer += f"\n\n**🖥️ System**: {payload}\n"
                    await _ui_update(text_buffer)

                elif event_type == "warning":
                    text_buffer += f"\n\n⚠️ {payload}"
                    await _ui_update(text_buffer)

                elif event_type == "cancelled":
                    text_buffer += (
                        f"\n\n🛑 **Cancelled**: {payload or 'user requested stop'}"
                    )
                    await _ui_update(text_buffer, immediate=True)
                    return

                elif event_type == "error":
                    text_buffer += f"\n\n❌ **Error**: {payload}"
                    await _ui_update(text_buffer, immediate=True)
                    return

                elif event_type == "finish":
                    text_buffer += f"\n\n**Finished**: {payload}"
                    await _ui_update(text_buffer)

        except asyncio.CancelledError:
            text_buffer += "\n\n🛑 **Agent loop cancelled by user**"
            await _ui_update(text_buffer, immediate=True)

        except Exception as exc:
            logger.exception("stream_llm crashed")
            text_buffer += f"\n\n❌ **Unexpected crash**: {exc}"
            await _ui_update(text_buffer, immediate=True)

        finally:
            if not text_buffer.strip():
                text_buffer = "No response generated."
            await _ui_update(text_buffer, immediate=True)

    async def action_clear_chat(self):
        self.chat.remove_children()
