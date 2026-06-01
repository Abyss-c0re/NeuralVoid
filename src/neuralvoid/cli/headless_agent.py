from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from neuralcore import Agent, Logger

# The canonical implementation lives in NeuralHub.
# We inherit the core execution loop, bridge lifecycle, and cancellation
# handling, then layer rich CLI output, signals, and status on top.
from neuralhub.runners.headless_runner import (
    HeadlessAgentRunner as _BaseHeadlessAgentRunner,
)

logger = Logger.get_logger()


class HeadlessAgentRunner(_BaseHeadlessAgentRunner):
    """
    Rich, human-facing headless runner used by the `neuralvoid` CLI.

    Inherits from the canonical :class:`neuralhub.runners.headless_runner.HeadlessAgentRunner`
    and reuses the shared driver (`_iter_agent_events`), while providing a much
    richer experience:

      - Live terminal output for every workflow event (phase, tools, deltas, LLM replies, …)
      - Detailed throttled status files
      - SIGINT/SIGTERM handlers with cooperative shutdown
      - Stale PID guard + single-instance protection
      - Graceful bridge shutdown via ``_stop_bridge`` override
      - Final status banner + file cleanup
    """

    def __init__(
        self,
        agent: Agent,
        status_file: str | Path = "/tmp/agent.status.json",
        pid_file: str | Path = "/tmp/agent.pid",
        websocket_port: int = 8765,
        status_update_throttle_sec: float = 1.0,
        # When True the runner skips creating its own WebSocketBridge.
        # Used by AgentHub which manages bridges externally.
        skip_bridge: bool = False,
        render_events: bool = False,
    ):
        # Delegate to the canonical implementation in NeuralHub.
        # Passing the explicit status/pid paths preserves the historical
        # defaults and any overrides coming from the CLI argument parser.
        super().__init__(
            agent=agent,
            status_file=status_file,
            pid_file=pid_file,
            websocket_port=websocket_port,
            status_update_throttle_sec=status_update_throttle_sec,
            skip_bridge=skip_bridge,
            # Explicit paths take precedence; do not inject an app_root here.
            app_root=None,
            render_events=render_events,
        )

        # The base class already initializes all core state
        # (_running, _success, _start_time, _last_status_write, paths, etc.).
        # Nothing else needs to be done here for the rich subclass.

    # ============================================================
    # Status / PID
    # ============================================================

    def _write_status(
        self,
        status: str,
        *,
        prompt: Optional[str] = None,
        iteration: Optional[int] = None,
        last_tool: Optional[str] = None,
        message: Optional[str] = None,
        error: Optional[str] = None,
        phase: Optional[str] = None,
        force: bool = False,
    ) -> None:
        now = datetime.utcnow()
        now_ts = now.timestamp()

        if not force and (now_ts - self._last_status_write) < self.throttle_sec:
            return

        data = {
            "pid": os.getpid(),
            "status": status,
            "started_at": self._start_time.isoformat() + "Z"
            if self._start_time
            else None,
            "last_update": now.isoformat() + "Z",
            "prompt": prompt or "",                    # ← safe for None/empty
            "current_iteration": iteration,
            "last_tool": last_tool,
            "current_phase": phase,
            "message": message,
            "error": error,
        }

        try:
            self.status_path.parent.mkdir(parents=True, exist_ok=True)
            self.status_path.write_text(json.dumps(data, indent=2))
            self._last_status_write = now_ts
        except Exception as e:
            print(f"Warning: failed to write status file: {e}", file=sys.stderr)

    def _write_pid(self) -> None:
        try:
            self.pid_path.parent.mkdir(parents=True, exist_ok=True)
            self.pid_path.write_text(str(os.getpid()))
        except Exception as e:
            print(f"Warning: failed to write PID file: {e}", file=sys.stderr)

    def _cleanup_files(self) -> None:
        for p in (self.pid_path, self.status_path):
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass

    async def _stop_bridge(self) -> None:
        """Graceful shutdown for the rich CLI experience."""
        if self._bridge:
            await self._bridge.stop()

    # ============================================================
    # Signals
    # ============================================================

    def _setup_signal_handlers(self, loop: asyncio.AbstractEventLoop) -> None:
        def shutdown_handler(sig: Optional[int] = None):
            name = signal.Signals(sig).name if sig else "Shutdown"
            print(f"\n[{name}] Stopping agent...")

            if self._stop_event:
                self._stop_event.set()

            self._write_status(
                "shutting_down", message="Received shutdown signal", force=True
            )

            for task in asyncio.all_tasks(loop):
                if task is not asyncio.current_task():
                    task.cancel()

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, shutdown_handler, sig)

    # ============================================================
    # Main run – now supports empty / None prompt
    # ============================================================

    async def run(
        self,
        prompt: Optional[str] = None,          # ← changed: now optional
        system_prompt: str = "",
        max_tokens: int = 12000,
    ) -> bool:
        if self._running:
            raise RuntimeError("Agent is already running")

        self._running = True
        self._success = False
        self._start_time = datetime.utcnow()
        self._stop_event = asyncio.Event()

        loop = asyncio.get_running_loop()
        self._setup_signal_handlers(loop)

        # PID safety
        if self.pid_path.exists():
            try:
                old_pid = int(self.pid_path.read_text().strip())
                os.kill(old_pid, 0)
                print(f"Process {old_pid} already running → abort")
                return False
            except OSError:
                print("Removing stale PID file")
                self.pid_path.unlink(missing_ok=True)

        self._write_pid()
        initial_mode = "listening (WS-driven)" if not prompt else "task"
        self._write_status(
            "starting",
            prompt=prompt,
            message=f"Starting in {initial_mode} mode",
            force=True,
        )

        current_iteration = 0
        current_phase = "idle"

        try:
            # Use the shared driver from the NeuralHub base class.
            # This eliminates duplication of bridge + agent.run() + stop_event wiring.
            async for event_type, payload in self._iter_agent_events(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=0.3,     # preserve the historical rich-CLI temperature
            ):
                if self._stop_event.is_set():
                    print("\n🛑 Stop event received")
                    break

                # Forward important events to WebSocket via the hook
                if event_type in (
                    "tool_result",
                    "final_answer",
                    "finish",
                    "error",
                    "phase_changed",
                    "planning_complete",
                ):
                    if hasattr(self.agent, "on_background_event"):
                        await self.agent.on_background_event(event_type, payload)
                    else:
                        logger.warning(
                            "Agent does not implement on_background_event hook"
                        )

                # ── Local console + status file handling ──
                if event_type == "phase_changed":
                    current_phase = payload.get("phase", current_phase)
                    self._write_status(
                        "running",
                        prompt=prompt,
                        iteration=current_iteration,
                        phase=current_phase,
                        message=f"Phase changed to {current_phase}",
                    )
                    print(f"\n→ Phase: {current_phase.upper()}")

                elif event_type == "planning_complete":
                    steps = payload.get("steps", [])
                    goal = payload.get("goal", "")
                    print(f"\n📋 Planning complete | Goal: {goal}")
                    for i, step in enumerate(steps, 1):
                        print(f"  {i}. {step}")

                elif event_type == "step_start":
                    current_iteration = payload.get("iteration", current_iteration)
                    print(
                        f"\n[{current_iteration}] Iteration start (phase: {current_phase})"
                    )

                elif event_type == "content_delta":
                    print(payload, end="", flush=True)

                elif event_type == "tool_start":
                    name = payload.get("name", "unknown")
                    args = payload.get("args", {})
                    print(f"\n🔧 TOOL START: {name} {args}")
                    self._write_status(
                        "running",
                        prompt=prompt,
                        iteration=current_iteration,
                        last_tool=name,
                        phase=current_phase,
                        message=f"Tool started: {name}",
                    )

                elif event_type == "tool_result":
                    name = payload.get("name", "unknown")
                    result = str(payload.get("result", ""))
                    if payload.get("error"):
                        print(f"\n❌ {name} failed: {result[:300]}...")
                        self._write_status(
                            "error",
                            iteration=current_iteration,
                            phase=current_phase,
                            error=result[:300],
                        )
                    else:
                        print(f"\n✅ {name} → {result[:300]}...")
                        self._write_status(
                            "running",
                            iteration=current_iteration,
                            last_tool=name,
                            phase=current_phase,
                            message=f"{name} completed",
                        )

                elif event_type == "tool_call_delta":
                    func = payload.get("function", {})
                    name = func.get("name") or "unknown"
                    print(f"\n🔧 Tool delta: {name}")

                elif event_type == "tool_calls":
                    count = len(payload) if isinstance(payload, list) else "?"
                    print(f"\nCalling {count} tool(s)...")

                elif event_type == "reflection_triggered":
                    print("\n🤔 Reflection:\n", str(payload).strip())

                # [FIX] Handle llm_response events from chat_tool_loop.
                # In casual/chat mode the workflow yields "llm_response" with
                # the full reply — previously this was silently ignored, so
                # nothing printed to the terminal when a WS message arrived.
                #
                # Important: we also log the actual reply text via the central
                # Logger so it reliably appears in ~/.neuralcore/neuralvoid.log
                # during --deploy + websocat control (even when render_events=False).
                elif event_type == "llm_response":
                    reply = payload.get("full_reply", "") if isinstance(payload, dict) else str(payload)
                    if reply:
                        print(f"\n💬 Agent reply:\n{reply}")
                        # Always-on clean entry in the persistent log for controller visibility
                        logger.info("Agent final response (llm_response):\n%s", reply)
                    self._write_status(
                        "running",
                        prompt=prompt,
                        iteration=current_iteration,
                        phase=current_phase,
                        message="LLM response received",
                    )

                elif event_type in ("final_answer", "final_summary"):
                    # Always-on clean rendering + logging of complete model output.
                    # The heavy banner versions stay inside the render_events guard.
                    text = str(payload) if payload is not None else ""
                    print(f"\n🏁 Final answer / summary:\n{text}\n")
                    if text.strip():
                        logger.info("Agent final answer / summary:\n%s", text)

                elif event_type == "finish":
                    reason = payload.get("reason", "unknown")
                    print(f"\n🏁 Finished: {reason}")

                    if reason in ("casual_complete", "task_complete", "normal"):
                        self._success = True
                    elif reason == "max_iterations_reached":
                        print("⚠️ Max iterations reached")
                    elif reason == "reflection_stuck":
                        print("⚠️ Agent stuck in reflection loop")

                elif event_type == "needs_confirmation":
                    print("\n⚠️ Confirmation required (skipped in headless mode)")
                    self._write_status(
                        "needs_confirmation",
                        message="Dangerous tool requires user confirmation",
                        force=True,
                    )

                elif event_type == "cancelled":
                    print(f"\n🛑 Cancelled: {payload}")
                    self._write_status("cancelled", message=str(payload), force=True)
                    self._success = False
                    break

                elif event_type == "error":
                    print(f"\n❌ Error: {payload}")
                    self._write_status("error", error=str(payload), force=True)
                    self._success = False
                    break

                elif event_type == "warning":
                    print(f"\n⚠️ Warning: {payload}")

                # ============================================================
                # Optional verbose rendering (DynamicCore, final answers, etc.)
                # Controlled by --render-events (CLI) or config (headless.render_events).
                # Default OFF for clean --deploy + websocat controller-driven testing.
                # ============================================================
                if getattr(self, "render_events", False):
                    if event_type.startswith("dynamic_core"):
                        print(f"\n🧠 [DYNAMIC_CORE] {event_type}: {payload}")

                    elif isinstance(payload, dict) and any(
                        k in payload for k in ("dynamic_core", "workflow_plan", "decision_action")
                    ):
                        print(f"\n🧠 [DYNAMIC_CORE] {event_type}: {payload}")

                    if event_type == "llm_response":
                        reply = payload.get("full_reply", "") if isinstance(payload, dict) else str(payload)
                        if reply and len(reply.strip()) > 5:
                            print("\n" + "═" * 80)
                            print("💬 AGENT FINAL RESPONSE / THOUGHT (readable by controller)")
                            print("═" * 80)
                            print(reply)
                            print("═" * 80 + "\n")

                    elif event_type in ("final_answer", "final_summary"):
                        print("\n" + "🏁" * 30)
                        print("FINAL ANSWER / SUMMARY FROM AGENT")
                        print("🏁" * 30)
                        print(payload)
                        print("🏁" * 30 + "\n")

                    elif event_type == "subtask_complete":
                        print(f"\n✅✅✅ [SUBTASK COMPLETE] {payload} ✅✅✅\n")

                    elif event_type == "validation_result":
                        print(f"\n🔍🔍🔍 [VALIDATION] {payload} 🔍🔍🔍\n")

                    elif "FINAL_ANSWER_COMPLETE" in str(payload):
                        print("\n" + "🎯" * 15 + " FINAL_ANSWER_COMPLETE " + "🎯" * 15)
                        print(payload)
                        print("🎯" * 40 + "\n")

                    # Extremely prominent DynamicCore section
                    if event_type.startswith("dynamic_core") or (
                        isinstance(payload, (dict, str))
                        and any(
                            x in str(payload).lower()
                            for x in ["dynamic_core", "workflow_plan", "decision_action", "replan"]
                        )
                    ):
                        print("\n" + "🧠" * 25)
                        print(f"🧠 DYNAMIC CORE DECISION / PLAN / EVENT: {event_type}")
                        print("🧠" * 25)
                        print(payload)
                        print("🧠" * 25 + "\n")

                    # Very visible catch-all (the noisiest one)
                    if event_type not in ("content_delta", "tool_call_delta"):
                        print(f"\n[EVENT] {event_type} → {str(payload)[:1200]}\n")

        except asyncio.CancelledError:
            print("\n🛑 Cancelled by system")
            self._write_status("cancelled", force=True)
            self._success = False

        except Exception as exc:
            print(f"\n❌ Unexpected error: {exc}")
            self._write_status("error", error=str(exc), force=True)
            self._success = False

        finally:
            self._running = False

            # Full agent shutdown (BackgroundManager + all internal jobs like KB watcher, training, etc.)
            # This is critical so that Ctrl+C on the user app actually stops background work.
            try:
                await self.agent.shutdown()
            except Exception as e:
                logger.warning(f"Error during agent.shutdown() in headless runner: {e}")

            # Final status + cleanup (bridge shutdown is handled by the
            # shared _iter_agent_events + our _stop_bridge override)
            if self._success:
                self._write_status("success", force=True)
            else:
                try:
                    current = json.loads(self.status_path.read_text())
                    if current.get("status") not in (
                        "error",
                        "cancelled",
                        "shutting_down",
                    ):
                        self._write_status("failed", force=True)
                except Exception:
                    self._write_status("failed", force=True)

            self._cleanup_files()

            print("\n" + "=" * 60)
            print(f"STATUS: {'SUCCESS' if self._success else 'FAILED'}")
            print("=" * 60)

            return self._success  # type: ignore  # return in finally is intentional here for the runner pattern