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
from neuralvoid.ui.helpers import _build_tool_markdown, _format_block, _format_text

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

    async def stream_llm(
        self,
        prompt: str,
        message: Message,
        tools: Optional[ToolProvider] = None,
    ) -> None:
        import time
        import asyncio

        level = getattr(self, "tool_rendering", "off")

        logger.info("🚀 stream_llm START | prompt=%s...", prompt[:150])

        current_msg: Message = message
        text_buffer: str = ""
        pure_assistant_text: str = ""
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

            display_buffer = new_buffer.replace("[FINAL_ANSWER_COMPLETE]", "")

            if not immediate and now - self._last_stream_update < self.UPDATE_INTERVAL:
                return

            if current_msg.buffer != display_buffer:
                current_msg.buffer = display_buffer
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
                # ── Content streaming ─────────────────────────────
                # ── Tool lifecycle ────────────────────────────────

                if event_type == "tool_call_delta":
                    # Try to get name and args incrementally from either full function dict or payload fields
                    func = payload.get("function", {})
                    name = func.get("name") or payload.get("name") or "unknown"

                    # Build partial args dict from delta (if available)
                    args = func.get("arguments") or payload.get("arguments_delta")
                    if isinstance(args, str):
                        # show only partial delta in a minimal JSON-like snippet
                        try:
                            # attempt to parse incremental JSON, fallback to raw string
                            parsed = json.loads(args)
                            args_dict = (
                                parsed
                                if isinstance(parsed, dict)
                                else {"_partial": args}
                            )
                        except Exception:
                            args_dict = {"_partial": args}
                    elif isinstance(args, dict):
                        args_dict = args
                    else:
                        args_dict = {}

                    # Render using your existing _build_tool_markdown for consistent Markdown
                    md = _build_tool_markdown(
                        name=name,
                        args=args_dict,
                        level=level,
                        result=None,
                        confirmation=None,
                        error=False,
                    )
                    text_buffer += md
                    await _ui_update(text_buffer, immediate=True)

                elif event_type == "tool_result":
                    md = _build_tool_markdown(
                        name=payload["name"],
                        args=payload.get("args", {}),
                        result=str(payload.get("result", "")),
                        error=payload.get("error", False),
                        level=level,
                    )
                    text_buffer += md
                    await _ui_update(text_buffer, immediate=True)

                elif event_type == "tool_calls":
                    count = len(payload)
                    text_buffer += _format_block(
                        "Tool Execution",
                        f"Calling {count} tool{'s' if count != 1 else ''}...",
                        "🛠️",
                    )
                    await _ui_update(text_buffer, immediate=True)

                elif event_type == "tool_call_delta":
                    func = payload.get("function", {})
                    name = func.get("name", "unknown")
                    args_part = func.get("arguments", "")[:60]
                    text_buffer += f"\n🔧 **Tool**: {name} (args: {args_part}…)"
                    await _ui_update(text_buffer, immediate=True)

                # ── Confirmation ────────────────────────────────
                elif event_type == "needs_confirmation":
                    md = _build_tool_markdown(
                        name=payload.get("name", "unknown"),
                        args=payload.get("args", {}),
                        confirmation=payload.get("preview", ""),
                        level=level,
                    )
                    text_buffer += (
                        f"\n\n{md}\n\n**Requires confirmation — type YES to approve**"
                    )
                    await _ui_update(text_buffer, immediate=True)

                    self.waiting_for_confirmation = True
                    self.pending_confirmation = {**payload}
                    return

                # ── Reflection ──────────────────────────────────
                elif event_type == "reflection_triggered":
                    if text_buffer.strip():
                        current_msg.buffer = text_buffer.strip()
                        current_msg.update(current_msg.render_markdown())

                    current_msg = await _new_assistant_bubble()
                    text_buffer = ""
                    pure_assistant_text = ""

                    text_buffer += _format_block(
                        "Agent Reflection",
                        str(payload).strip(),
                        "🤔",
                    )
                    await _ui_update(text_buffer, immediate=True)

                # ── Review phase ────────────────────────────────
                elif event_type == "review_phase":
                    if isinstance(payload, dict) and "summary" in payload:
                        nice_summary = _format_text(payload["summary"])
                    else:
                        nice_summary = str(payload)

                    text_buffer += _format_block(
                        "Review Phase",
                        nice_summary,
                        "🔍",
                    )
                    await _ui_update(text_buffer, immediate=True)

                # ── Meta / status ───────────────────────────────
                elif event_type == "step_start":
                    iter_num = payload.get("iteration", "?")
                    text_buffer += _format_block(
                        f"Iteration {iter_num}",
                        "Started",
                        "🔄",
                    )
                    await _ui_update(text_buffer)

                elif event_type == "llm_finish":
                    text_buffer += _format_block(
                        "LLM Generation Finished",
                        "Model completed its response.",
                        "✅",
                    )
                    await _ui_update(text_buffer)

                elif event_type == "system":
                    text_buffer += _format_block("System", payload, "🖥️")
                    await _ui_update(text_buffer)

                elif event_type == "warning":
                    text_buffer += _format_block("Warning", payload, "⚠️")
                    await _ui_update(text_buffer)

                elif event_type == "cancelled":
                    text_buffer += _format_block(
                        "Cancelled",
                        payload or "user requested stop",
                        "🛑",
                    )
                    await _ui_update(text_buffer, immediate=True)
                    return

                elif event_type == "error":
                    text_buffer += _format_block("Error", payload, "❌")
                    await _ui_update(text_buffer, immediate=True)
                    return

                # ── Finish (clean + correct) ─────────────────────
                elif event_type == "finish":
                    reason = payload.get("reason", "unknown")

                    if reason == "casual_complete":
                        text_buffer = pure_assistant_text.strip()
                        await _ui_update(text_buffer, immediate=True)

                        if text_buffer:
                            self.conversation.append(
                                {"role": "assistant", "content": text_buffer}
                            )
                        return

                    summary = payload.get("summary", "")

                    if summary:
                        summary = _format_text(summary)

                        text_buffer += _format_block(
                            f"Finished — {reason.replace('_', ' ').title()}",
                            summary,
                            "🏁",
                        )
                    else:
                        text_buffer += _format_block(
                            "Finished",
                            reason,
                            "🏁",
                        )

                    await _ui_update(text_buffer, immediate=True)

        except asyncio.CancelledError:
            text_buffer += _format_block(
                "Cancelled",
                "Agent loop cancelled by user",
                "🛑",
            )
            await _ui_update(text_buffer, immediate=True)

        except Exception as exc:
            logger.exception("stream_llm crashed")
            text_buffer += _format_block(
                "Crash",
                str(exc),
                "❌",
            )
            await _ui_update(text_buffer, immediate=True)

        finally:
            if not text_buffer.strip():
                text_buffer = "No response generated."
            await _ui_update(text_buffer, immediate=True)

    async def action_clear_chat(self):
        self.chat.remove_children()
