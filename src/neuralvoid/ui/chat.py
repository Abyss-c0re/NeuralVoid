import asyncio
import json
import time

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
# Message widget – clean status footer (no iteration spam)
# ============================================================


class Message(Markdown):
    """Markdown-rendered chat message with optional status footer."""

    def __init__(self, role: str, content: str = ""):
        self.role = role
        self.buffer = content
        self.status_line: str = ""
        super().__init__(self.render_markdown())

    def update_status(self, text: str) -> None:
        """Update only the tiny status line at the bottom."""
        self.status_line = text.strip()
        self.update(self.render_markdown())

    def clear_status(self) -> None:
        self.status_line = ""
        self.update(self.render_markdown())

    def render_markdown(self) -> str:
        if self.role == "user":
            prefix = "🧑 **You**: "
        elif self.role == "assistant":
            prefix = "🤖 **Assistant**: "
        elif self.role == "system":
            prefix = "💻 **System**: "
        else:
            prefix = ""

        main = prefix + self.buffer

        if self.status_line:
            # Tiny dimmed status (never clogs the main content)
            status = f"\n\n<span style='dim'>└─ {self.status_line}</span>"
            return main + status
        return main


# ============================================================
# Chat container
# ============================================================


class ChatView(VerticalScroll):
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
    _last_assistant_msg: Optional[Message] = None

    # UI constants
    UPDATE_INTERVAL = 0.08
    SPINNERS = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

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
        tools: Optional[ToolProvider] = None,
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
        self.tool_rendering = tool_rendering
        self.max_iterations = max_iterations or getattr(client, "max_iterations", 20)
        self.temperature = temperature or getattr(client, "temperature", 0.7)
        self.max_tokens = max_tokens or getattr(client, "max_tokens", 32000)

        self._spinner_idx = 0

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
        self.query_one(Input).focus()

    def action_stop_stream(self) -> None:
        if self._current_stream_task is None or self._current_stream_task.done():
            self.notify("No active generation to stop.", timeout=2.5)
            return
        self._current_stream_task.cancel("user requested stop via Esc")
        self.notify("🛑 Generation cancelled", severity="warning", timeout=4)

    # =============================================================
    # Full confirmation handling (exactly as original)
    # =============================================================
    async def _handle_confirmation_response(self, user_input: str) -> bool:
        if not self.waiting_for_confirmation or not self.pending_confirmation:
            return False

        approved = user_input.strip().upper() in {"YES", "Y", "OK", "CONFIRM"}
        info = self.pending_confirmation

        tool_call_id = info["tool_call_id"]
        name = info["name"]
        args = info["args"]
        action = info["action"]
        assistant_msg = info["assistant_msg"]
        original_tool_calls = info["tool_calls"]

        if approved:
            try:
                real_executor = action.executor
                if asyncio.iscoroutinefunction(real_executor):
                    result = await real_executor(**args)
                else:
                    result = real_executor(**args)
            except Exception as exc:
                result = f"Error during confirmed execution: {exc}"
            content = str(result)
        else:
            content = "User denied the action."

        tool_result_msg = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": name,
            "content": content,
        }

        self.conversation.append(
            {"role": "assistant", "tool_calls": original_tool_calls}
        )
        self.conversation.append(tool_result_msg)

        feedback = f"→ Action **{name}** was {'approved' if approved else 'denied'}.\nResult: {content}"
        assistant_msg.buffer += f"\n\n{feedback}"
        assistant_msg.update(assistant_msg.render_markdown())
        self.chat.scroll_end()

        self.waiting_for_confirmation = False
        self.pending_confirmation = None

        # Continue in the SAME message bubble
        asyncio.create_task(
            self.stream_llm(
                prompt="",
                message=assistant_msg,
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
            self.action_stop_stream()
            return
        if cmd == "exit":
            self.chat.add(Message("system", "👋 Exiting..."))
            await asyncio.sleep(0.3)
            self.exit()
            return

        if await self._handle_confirmation_response(value):
            return

        # Normal message
        self.chat.add(Message("user", value))
        self.conversation.append({"role": "user", "content": value})

        assistant_msg = Message("assistant", "")
        self.chat.add(assistant_msg)
        self._last_assistant_msg = assistant_msg

        task = asyncio.create_task(
            self.stream_llm(value, assistant_msg, self.tools), name="llm-stream"
        )
        self._current_stream_task = task

        def on_done(fut):
            if fut is self._current_stream_task:
                self._current_stream_task = None
                if last := getattr(self, "_last_assistant_msg", None):
                    last.clear_status()

        task.add_done_callback(on_done)

    # =============================================================
    # FIXED STREAMING – tool args/markdown restored + clean status
    # =============================================================
    async def stream_llm(
        self,
        prompt: str,
        message: Message,
        tools: Optional[ToolProvider] = None,
        ) -> None:
        pure_assistant_text: str = ""
        text_buffer: str = ""
        self._last_stream_update = time.time() - 0.1
        self._spinner_idx = 0

        # Decide how detailed tool rendering should be ("off", "compact", "full")
        level = getattr(self, "tool_rendering_level", "compact")  # ← pick a sane default

        runner = AgentRunner(
            client=self.client,
            max_iterations=self.max_iterations,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        async def _ui_update(new_buffer: str, immediate: bool = False) -> None:
            nonlocal text_buffer
            now = time.time()
            display = new_buffer.replace("[FINAL_ANSWER_COMPLETE]", "")

            if not immediate and now - self._last_stream_update < self.UPDATE_INTERVAL:
                return

            if message.buffer != display:
                message.buffer = display
                message.update(message.render_markdown())
                self.chat.scroll_end(animate=False)
                self._last_stream_update = now

        try:
            async for event_type, payload in runner.run(
                user_prompt=prompt,
                tools=tools or [],
                system_prompt=self.system_prompt or "",
                context_manager=self.context_manager,
            ):
                # ── Main content streaming ───────────────────────────────────────
                if event_type == "content_delta":
                    pure_assistant_text += payload
                    text_buffer += payload
                    await _ui_update(text_buffer)

                # ── Tool call streaming (partial name + arguments) ───────────────
                elif event_type == "tool_call_delta":
                    func = payload.get("function", {})
                    name = func.get("name") or payload.get("name") or "unknown"

                    args = func.get("arguments") or payload.get("arguments_delta") or ""
                    if isinstance(args, str):
                        try:
                            parsed = json.loads(args)
                            args_dict = parsed if isinstance(parsed, dict) else {"_partial": args}
                        except Exception:
                            args_dict = {"_partial": args}
                    elif isinstance(args, dict):
                        args_dict = args
                    else:
                        args_dict = {}

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

                    message.update_status(
                        f"{self.SPINNERS[self._spinner_idx % len(self.SPINNERS)]} using **{name}**"
                    )
                    self._spinner_idx += 1

                # ── Tool final result ────────────────────────────────────────────
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
                    message.clear_status()

                # ── Confirmation request ─────────────────────────────────────────
                elif event_type == "needs_confirmation":
                    md = _build_tool_markdown(
                        name=payload.get("name", "unknown"),
                        args=payload.get("args", {}),
                        confirmation=payload.get("preview", ""),
                        level=level,
                    )
                    text_buffer += f"\n\n{md}\n\n**Requires confirmation — type YES to approve**"
                    await _ui_update(text_buffer, immediate=True)

                    message.update_status("⏳ Waiting for your confirmation")
                    self.waiting_for_confirmation = True
                    self.pending_confirmation = {**payload}
                    return  # pause streaming

                # ── New reasoning iteration ──────────────────────────────────────
                elif event_type == "step_start":
                    iter_num = payload.get("iteration", "?")
                    message.update_status(
                        f"{self.SPINNERS[self._spinner_idx % len(self.SPINNERS)]} Iteration {iter_num} 🔄"
                    )
                    self._spinner_idx += 1

                # ── System / warning / error / cancelled messages ────────────────
                elif event_type == "system":
                    text_buffer += _format_block("System", payload, "🖥️")
                    await _ui_update(text_buffer)

                elif event_type == "warning":
                    text_buffer += _format_block("Warning", payload, "⚠️")
                    await _ui_update(text_buffer)

                elif event_type == "cancelled":
                    text_buffer += _format_block(
                        "Cancelled", payload or "user requested stop", "🛑"
                    )
                    await _ui_update(text_buffer, immediate=True)
                    message.clear_status()
                    return

                elif event_type == "error":
                    text_buffer += _format_block("Error", payload, "❌")
                    await _ui_update(text_buffer, immediate=True)
                    message.clear_status()
                    return

                # ── Final finish handling ────────────────────────────────────────
                elif event_type == "finish":
                    reason = payload.get("reason", "unknown")

                    if reason == "casual_complete":
                        final_text = pure_assistant_text.strip()
                        if final_text:
                            text_buffer = final_text
                            await _ui_update(text_buffer, immediate=True)
                            self.conversation.append(
                                {"role": "assistant", "content": final_text}
                            )
                    else:
                        # For tool-using / structured mode: keep what we built
                        if payload.get("summary"):
                            summary = _format_text(payload["summary"])
                            text_buffer += _format_block(
                                f"Finished — {reason.replace('_', ' ').title()}",
                                summary,
                                "🏁",
                            )
                        else:
                            text_buffer += _format_block("Finished", reason, "🏁")

                        await _ui_update(text_buffer, immediate=True)

                        # Important: save final assistant message in ALL cases
                        if text_buffer.strip():
                            self.conversation.append(
                                {"role": "assistant", "content": text_buffer.strip()}
                            )

                    message.clear_status()
                    return   # ← usually good to exit loop here

        except asyncio.CancelledError:
            text_buffer += _format_block(
                "Cancelled", "Agent loop stopped by user", "🛑"
            )
            await _ui_update(text_buffer, immediate=True)
            message.clear_status()

        except Exception as exc:
            logger.exception("stream_llm crashed")
            text_buffer += _format_block("Crash", str(exc), "❌")
            await _ui_update(text_buffer, immediate=True)
            message.clear_status()

        finally:
            if not text_buffer.strip():
                text_buffer = "No response generated."
                await _ui_update(text_buffer, immediate=True)
            message.clear_status()

    async def action_clear_chat(self):
        self.chat.remove_children()


