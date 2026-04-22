import asyncio
import time
from typing import Optional, Any

from textual.app import App, ComposeResult
from textual.widgets import TextArea, Markdown
from textual.containers import VerticalScroll
from textual.binding import Binding
from textual.message import Message as TextualMessage
from textual.events import Key

from neuralcore.agents.core import Agent
from neuralcore.actions.registry import registry
from neuralcore.utils.prompt_builder import PromptBuilder


from neuralvoid.ui.rendering import set_renderer_app, get_renderer
from neuralvoid.ui.helpers import _build_tool_markdown
from neuralcore.utils.logger import Logger

logger = Logger.get_logger()


# ============================================================
# Message Widget
# ============================================================


class ChatMessage(Markdown):
    """Markdown-rendered chat message with optional status footer."""

    def __init__(self, role: str, content: str = ""):
        self.role = role
        self.buffer = content
        self.status_line: str = ""
        super().__init__(self.render_markdown())

    def update_status(self, text: str) -> None:
        self.status_line = text.strip()

        def _do_update():
            self.update(self.render_markdown())
            self.refresh()
            if self.app:
                self.app.refresh(layout=True)
                self.app.call_later(self.refresh)

        if self.app:
            self.app.call_later(_do_update)
        else:
            _do_update()

    def clear_status(self) -> None:
        self.status_line = ""
        if self.app:
            self.app.call_later(
                lambda: (
                    self.update(self.render_markdown()),
                    self.refresh(),
                    self.app.refresh(layout=True),
                )
            )
        else:
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
            status = f"\n\n<span style='dim'>└─ {self.status_line}</span>"
            return main + status
        return main


# ============================================================
# Chat Container
# ============================================================


class ChatView(VerticalScroll):
    """Chat container with reliable auto-scrolling, especially after final answer."""

    def add(self, widget) -> None:
        self.mount(widget)
        self.call_after_refresh(self._scroll_to_bottom)

    def _scroll_to_bottom(self) -> None:
        self.scroll_end(animate=False, immediate=True)
        self.refresh(layout=True)

    async def ensure_final_scroll(self) -> None:
        """Aggressive final scroll - multiple attempts with delays for Markdown layout."""
        for i in range(6):
            self.scroll_end(animate=False, immediate=True)
            self.refresh(layout=True)
            if self.app:
                self.app.refresh(layout=True)
            await asyncio.sleep(0.02 + i * 0.01)

    def on_child_update(self) -> None:
        if self._should_auto_scroll():
            self.call_after_refresh(self._scroll_to_bottom)

    def _should_auto_scroll(self) -> bool:
        if self.is_vertical_scrollbar_grabbed:
            return False
        return self.scroll_y >= self.max_scroll_y - 80


# ============================================================
# Multiline Chat Input
# ============================================================


class ChatInput(TextArea):
    class Submitted(TextualMessage):
        def __init__(self, text_area: "ChatInput") -> None:
            super().__init__()
            self.text_area = text_area
            self.value = text_area.text

    async def on_key(self, event: Key) -> None:
        if event.key == "enter":
            event.stop()
            self.post_message(self.Submitted(self))
            return

        if event.key in ("ctrl+n",):
            event.stop()
            self.insert("\n")
            return


# ============================================================
# Main App
# ============================================================


class LLMChatApp(App):
    agent: Agent
    system_prompt: Optional[str]
    waiting_for_confirmation: bool = False
    pending_confirmation: Optional[dict] = None

    _agent_task: asyncio.Task | None = None
    _current_assistant_msg: Optional[ChatMessage] = None

    _current_pure_text: str = ""
    _current_tool_buffer: str = ""
    _last_stream_update: float = 0.0
    _last_finished: bool = True

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

    TextArea {
        dock: bottom;
        height: 4;
        border: tall $primary;
        padding: 0 1;
    }
    """

    def __init__(
        self,
        agent: Agent,
        system_prompt: Optional[str] = None,
        tool_rendering: Optional[str] = "off",
        max_iterations: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tool_info_level: Optional[str] = "compact",
    ):
        super().__init__()
        self.agent = agent

        self.client = agent.client
        self.registry = registry
        self.system_prompt = system_prompt
        self.context_manager = agent.context_manager
        self.conversation = []

        self.rendering = get_renderer()
        self.tool_rendering = tool_rendering
        self.max_iterations = max_iterations or getattr(
            agent.client, "max_iterations", 20
        )
        self.temperature = temperature or getattr(agent.client, "temperature", 0.7)
        self.max_tokens = max_tokens or getattr(agent.client, "max_tokens", 32000)
        self.tool_info_level = tool_info_level

        self._spinner_idx = 0

    def compose(self) -> ComposeResult:
        self.chat = ChatView(id="chat")
        yield self.chat
        yield ChatInput(id="input")

    async def on_mount(self):
        set_renderer_app(self, ChatMessage)
        await self.rendering.start_worker()

        self.chat.add(
            ChatMessage(
                "assistant",
                f"Connected to **{self.client.model}**\n\n"
                "Type a message and press **Enter** to send.\n"
                "Use **Ctrl+n** for a new line.\n"
                "Commands: **stop** / **cancel** → stop current stream\n"
                "**exit** → close app",
            )
        )

        self._agent_task = asyncio.create_task(self.run_chat_loop(), name="chat-loop")
        self.query_one("#input").focus()

    async def on_key(self, event: Key) -> None:
        if event.key == "escape":
            await self.action_stop_stream()
            event.stop()

    # ====================== Persistent Agent Loop ======================

    async def run_chat_loop(self):
        try:
            async for event_type, payload in self.agent.run(
                user_prompt=None,
                system_prompt=self.system_prompt or "",
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                chat_mode=True,
            ):
                # logger.debug(f"[UI] Event: {event_type}")

                if event_type in (
                    "content_delta",
                    "llm_response",
                    "step_completed",
                    "final_summary",
                ) and (self._current_assistant_msg is None or self._last_finished):
                    assistant_msg = ChatMessage("assistant", "")
                    self.chat.add(assistant_msg)
                    self._current_assistant_msg = assistant_msg
                    self._current_pure_text = ""
                    self._current_tool_buffer = ""
                    self._last_finished = False

                if self._current_assistant_msg is None:
                    continue

                await self._process_agent_event(
                    event_type, payload, self._current_assistant_msg
                )

                if event_type in ("finish", "llm_response"):
                    self._last_finished = True
                    self._current_assistant_msg = None

        except asyncio.CancelledError:
            logger.debug("Chat loop cancelled")
        except Exception as e:
            logger.exception("Chat loop crashed")
            self.chat.add(ChatMessage("system", f"❌ Runner error: {e}"))

    # ====================== UI Update ======================

    async def _ui_update(self, message: ChatMessage, immediate: bool = False) -> None:
        now = time.time()
        if not immediate and now - self._last_stream_update < self.UPDATE_INTERVAL:
            return

        display = (self._current_pure_text + self._current_tool_buffer).replace(
            PromptBuilder.FINAL_ANSWER_MARKER, ""
        )

        if message.buffer != display:
            message.buffer = display
            message.update(message.render_markdown())
            self.chat.call_after_refresh(self.chat._scroll_to_bottom)
            self._last_stream_update = now

    # ====================== Process Agent Events ======================

    async def _process_agent_event(
        self, event_type: str, payload: Any, message: ChatMessage
    ) -> None:
        """Process events from the agent with proper message creation for final synthesis."""
        level = self.tool_info_level or "compact"

        if event_type == "phase_changed":
            phase = payload.get("phase", "unknown").strip()
            logger.debug(f"[UI] PHASE_CHANGED: {phase}")
            if self._current_assistant_msg:
                spinner = self.SPINNERS[self._spinner_idx % len(self.SPINNERS)]
                self._spinner_idx = (self._spinner_idx + 1) % len(self.SPINNERS)

                if "calling" in phase.lower():
                    status_text = f"🔨 {phase}"
                elif "executing" in phase.lower():
                    status_text = f"⚡ {phase.upper()}"
                elif (
                    "synthesizing" in phase.lower()
                    or "final" in phase.lower()
                    or "generating_final_answer" in phase.lower()
                ):
                    status_text = "✨ Generating final answer..."
                elif "reflecting" in phase.lower():
                    status_text = f"🤔 {phase.upper()}"
                else:
                    status_text = f"{spinner} {phase.upper()}"

                self._current_assistant_msg.update_status(status_text)
                self.chat.call_after_refresh(self.chat._scroll_to_bottom)
            return

        elif event_type == "tool_name":
            name = payload.get("name", "unknown")
            message.update_status(
                f"{self.SPINNERS[self._spinner_idx % len(self.SPINNERS)]} using **{name}**"
            )
            self._spinner_idx = (self._spinner_idx + 1) % len(self.SPINNERS)
            return

        elif event_type == "content_delta":
            # CRITICAL FIX: Always ensure we have an active message for content_delta
            if self._current_assistant_msg is None:
                assistant_msg = ChatMessage("assistant", "")
                self.chat.add(assistant_msg)
                self._current_assistant_msg = assistant_msg
                self._current_pure_text = ""
                self._current_tool_buffer = ""
                self._last_finished = False
                logger.debug("[UI] Created new assistant message for content_delta")

            self._current_pure_text += str(payload or "")
            await self._ui_update(self._current_assistant_msg)
            return

        elif event_type == "llm_response":
            full_reply = payload.get("full_reply", "").strip()
            if full_reply:
                if self._current_assistant_msg is None:
                    assistant_msg = ChatMessage("assistant", "")
                    self.chat.add(assistant_msg)
                    self._current_assistant_msg = assistant_msg
                    logger.debug("[UI] Created new assistant message for llm_response")

                # Only set if it's a real final reply (not the short "Task completed successfully")
                if len(full_reply) > 50 or "synthesis" in str(payload).lower():
                    self._current_pure_text = full_reply
                await self._ui_update(self._current_assistant_msg, immediate=True)
                self._current_assistant_msg.clear_status()
                self.chat.call_after_refresh(self.chat._scroll_to_bottom)
            return

        elif event_type == "step_completed":
            summary = payload.get("summary", "") or str(payload)
            self._current_pure_text += f"\n✅ **Step completed**\n{summary}\n\n"
            await self._ui_update(message, immediate=True)
            return

        elif event_type == "step_failed":
            error = payload.get("error", "Unknown error")
            self._current_pure_text += f"\n❌ **Step failed**\n{error}\n\n"
            await self._ui_update(message, immediate=True)
            return

        elif event_type == "planning_complete":
            steps = payload.get("steps", [])
            goal = payload.get("goal", "")
            self._current_pure_text += (
                f"\n**Planning complete**\nGoal: {goal}\n\n**Steps:**\n"
            )
            for i, step in enumerate(steps, 1):
                self._current_pure_text += f"{i}. {step}\n"
            self._current_pure_text += "\n"
            await self._ui_update(message, immediate=True)
            return

        elif event_type == "needs_confirmation":
            tool_name = payload.get("name", "unknown")
            md = _build_tool_markdown(
                name=tool_name,
                args=payload.get("args", {}),
                confirmation=payload.get("preview", ""),
                level=level,
            )
            self._current_tool_buffer += md
            await self._ui_update(message, immediate=True)

            message.update_status("⏳ Waiting for your confirmation")

            self.pending_confirmation = {**payload, "assistant_msg": message}
            self.waiting_for_confirmation = True
            return

        elif event_type == "final_summary":
            self._current_pure_text += f"\n\n{payload}\n"
            await self._ui_update(message, immediate=True)
            return

        # ====================== FINAL ANSWER ======================
        elif event_type == "finish":
            reason = payload.get("reason", "unknown")

            if reason == "casual_complete":
                self._current_pure_text = self._current_pure_text.strip()
            elif self._current_tool_buffer.strip():
                self._current_tool_buffer = (
                    "\n\n─── 🔧 Tool usage history ───\n\n" + self._current_tool_buffer
                )

            await self._ui_update(message, immediate=True)

            final_content = (
                self._current_pure_text + self._current_tool_buffer
            ).strip()
            if final_content:
                self.conversation.append(
                    {"role": "assistant", "content": final_content}
                )

            if self._current_assistant_msg:
                self._current_assistant_msg.clear_status()

            await self.chat.ensure_final_scroll()

            self._current_assistant_msg = None
            self._current_pure_text = ""
            self._current_tool_buffer = ""
            return

        elif event_type in ("cancelled", "error", "warning"):
            icon = {"cancelled": "🛑", "error": "❌", "warning": "⚠️"}.get(
                event_type, "⚠️"
            )
            self._current_pure_text += (
                f"\n\n{icon} **{event_type.capitalize()}**\n{payload}\n"
            )
            await self._ui_update(message, immediate=True)

            if event_type in ("cancelled", "error"):
                if self._current_assistant_msg:
                    self._current_assistant_msg.clear_status()
                self._current_assistant_msg = None
                self._current_pure_text = ""
                self._current_tool_buffer = ""
            return

    # ====================== Confirmation Handler ======================

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

        await self.agent.post_control(
            {
                "event": "needs_confirmation",
                "approved": approved,
                "content": content,
                "tool_call_id": tool_call_id,
                "name": name,
            }
        )

        feedback = f"→ Action **{name}** was {'approved' if approved else 'denied'}.\nResult: {content}"
        assistant_msg.buffer += f"\n\n{feedback}"
        assistant_msg.update(assistant_msg.render_markdown())
        await self.chat.ensure_final_scroll()

        self.waiting_for_confirmation = False
        self.pending_confirmation = None
        return True

    # ====================== User Input ======================

    async def on_chat_input_submitted(self, event: ChatInput.Submitted):
        value = event.value.strip()
        event.text_area.clear()
        if not value:
            return

        cmd = value.lower()
        if cmd in ("stop", "cancel"):
            await self.action_stop_stream()
            return

        if cmd == "exit":
            self.chat.add(ChatMessage("system", "👋 Exiting..."))
            await asyncio.sleep(0.3)
            self.exit()
            return

        if await self._handle_confirmation_response(value):
            return

        self.chat.add(ChatMessage("user", value))
        self.conversation.append({"role": "user", "content": value})

        assistant_msg = ChatMessage("assistant", "")
        self.chat.add(assistant_msg)

        self._current_assistant_msg = assistant_msg
        self._current_pure_text = ""
        self._current_tool_buffer = ""

        await self.agent.post_message(value)

    async def action_stop_stream(self) -> None:
        if self._current_assistant_msg is None:
            self.notify("No active generation to stop.", timeout=2.5)
            return
        await self.agent.post_control({"event": "cancelled"})
        self.notify("🛑 Generation cancelled", severity="warning", timeout=4)

    async def action_clear_chat(self):
        self.chat.remove_children()
        self.conversation.clear()
        self.waiting_for_confirmation = False
        self.pending_confirmation = None
        self._current_assistant_msg = None
        self._current_pure_text = ""
        self._current_tool_buffer = ""
