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
    _current_stop_event: asyncio.Event | None = None
    _last_assistant_msg: Optional[Message] = None
    _pending_context: list[str]

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
        tool_info_level: Optional[str] = "compact",
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
        self.tool_info_level = tool_info_level

        self._spinner_idx = 0
        self._pending_context = []

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

    def _start_stream(self, prompt: str) -> None:
        """Start a brand-new agent turn with its own stop event."""
        assistant_msg = Message("assistant", "")
        self.chat.add(assistant_msg)
        self._last_assistant_msg = assistant_msg

        stop_event = asyncio.Event()
        self._current_stop_event = stop_event

        task = asyncio.create_task(
            self.stream_llm(
                prompt=prompt,
                message=assistant_msg,
                tools=self.tools,
                stop_event=stop_event,
            ),
            name="llm-stream",
        )
        self._current_stream_task = task

        def on_done(fut: asyncio.Task):
            if fut is not self._current_stream_task:
                return

            self._current_stream_task = None
            self._current_stop_event = None

            if last := getattr(self, "_last_assistant_msg", None):
                last.clear_status()

            # If the user typed while the agent was working, restart now
            if self._pending_context and not self.waiting_for_confirmation:
                followup_prompt = "\n".join(self._pending_context).strip()
                self._pending_context.clear()

                if followup_prompt:
                    self._start_stream(followup_prompt)

        task.add_done_callback(on_done)

    def action_stop_stream(self) -> None:
        if self._current_stream_task is None or self._current_stream_task.done():
            self.notify("No active generation to stop.", timeout=2.5)
            return

        if self._current_stop_event is not None:
            self._current_stop_event.set()

        self._current_stream_task.cancel("user requested stop via Esc")
        self.notify("🛑 Generation cancelled", severity="warning", timeout=4)

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

        asyncio.create_task(
            self.stream_llm(
                prompt="",
                message=assistant_msg,
                tools=self.tools,
                stop_event=asyncio.Event(),
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

        # Always show the user message immediately
        self.chat.add(Message("user", value))
        self.conversation.append({"role": "user", "content": value})

        # If the agent is working, queue the text as follow-up context and restart cleanly
        if (
            self._current_stream_task is not None
            and not self._current_stream_task.done()
        ):
            self._pending_context.append(value)
            self.action_stop_stream()
            return

        # No active agent: start normally
        self._start_stream(value)

    async def stream_llm(
        self,
        prompt: str,
        message: Message,
        tools: Optional[ToolProvider] = None,
        stop_event: Optional[asyncio.Event] = None,
    ) -> None:
        """Clean streaming with perfect final-answer formatting (TUI-safe)."""
        pure_assistant_text: str = ""
        tool_visual_buffer: str = ""
        self._last_stream_update = time.time() - 0.1
        self._spinner_idx = 0
        stop_event = stop_event or asyncio.Event()

        level = self.tool_info_level or "compact"

        runner = AgentRunner(
            client=self.client,
            max_iterations=self.max_iterations,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        async def _ui_update(immediate: bool = False) -> None:
            nonlocal pure_assistant_text, tool_visual_buffer
            now = time.time()
            display = (pure_assistant_text + tool_visual_buffer).replace(
                "[FINAL_ANSWER_COMPLETE]", ""
            )

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
                stop_event=stop_event,
            ):
                if stop_event.is_set():
                    return

                if event_type == "content_delta":
                    pure_assistant_text += payload
                    await _ui_update()

                elif event_type == "tool_call_delta":
                    func = payload.get("function", {})
                    name = func.get("name") or payload.get("name") or "unknown"

                    args = func.get("arguments") or payload.get("arguments_delta") or ""
                    if isinstance(args, str):
                        try:
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

                    md = _build_tool_markdown(
                        name=name,
                        args=args_dict,
                        level=level,
                        result=None,
                        confirmation=None,
                        error=False,
                    )
                    tool_visual_buffer += md
                    await _ui_update(immediate=True)

                    message.update_status(
                        f"{self.SPINNERS[self._spinner_idx % len(self.SPINNERS)]} using **{name}**"
                    )
                    self._spinner_idx += 1

                elif event_type == "tool_calls":
                    # Optional: keep this if you want a visible marker for tool batches
                    pass

                elif event_type == "needs_confirmation":
                    md = _build_tool_markdown(
                        name=payload.get("name", "unknown"),
                        args=payload.get("args", {}),
                        confirmation=payload.get("preview", ""),
                        level=level,
                    )
                    tool_visual_buffer += md
                    await _ui_update(immediate=True)
                    message.update_status("⏳ Waiting for your confirmation")
                    self.waiting_for_confirmation = True
                    self.pending_confirmation = {**payload}
                    return

                elif event_type in ("system", "warning", "cancelled", "error"):
                    icon = {
                        "system": "🖥️",
                        "warning": "⚠️",
                        "cancelled": "🛑",
                        "error": "❌",
                    }[event_type]
                    pure_assistant_text += (
                        f"\n\n{icon} **{event_type.capitalize()}**\n{payload}\n"
                    )
                    await _ui_update(immediate=True)
                    if event_type in ("cancelled", "error"):
                        message.clear_status()
                        return

                elif event_type == "step_start":
                    message.update_status(
                        f"{self.SPINNERS[self._spinner_idx % len(self.SPINNERS)]} Iteration {payload.get('iteration', '?')}"
                    )
                    self._spinner_idx += 1

                elif event_type == "final_summary":
                    pure_assistant_text += f"\n\n{payload}\n"
                    await _ui_update(immediate=True)

                elif event_type == "finish":
                    reason = payload.get("reason", "unknown")

                    if reason == "casual_complete":
                        pure_assistant_text = pure_assistant_text.strip()
                    else:
                        if tool_visual_buffer.strip():
                            tool_visual_buffer = (
                                "\n\n─── 🔧 Tool usage history ───\n\n"
                                + tool_visual_buffer
                            )

                    await _ui_update(immediate=True)

                    final_content = (pure_assistant_text + tool_visual_buffer).strip()
                    if final_content:
                        self.conversation.append(
                            {"role": "assistant", "content": final_content}
                        )

                    message.clear_status()
                    return

        except asyncio.CancelledError:
            pure_assistant_text += "\n\n🛑 **Cancelled** by user"
            await _ui_update(immediate=True)

        except Exception as exc:
            logger.exception("stream_llm crashed")
            pure_assistant_text += f"\n\n❌ **Crash**\n{exc}"
            await _ui_update(immediate=True)

        finally:
            if not (pure_assistant_text + tool_visual_buffer).strip():
                pure_assistant_text = "No response generated."
                await _ui_update(immediate=True)
            message.clear_status()

    async def action_clear_chat(self):
        self.chat.remove_children()
        self.conversation.clear()
        self._pending_context.clear()
        self.waiting_for_confirmation = False
        self.pending_confirmation = None
        self._current_stream_task = None
        self._current_stop_event = None
        self._last_assistant_msg = None
