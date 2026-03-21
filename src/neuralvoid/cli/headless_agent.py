from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from neuralcore.agents.core import Agent


class HeadlessAgentRunner:
    """
    Fully updated headless runner (March 2026) — now using the new Agent class.
    Features preserved: cooperative cancellation, graceful shutdown, status/PID files,
    accurate success detection, clean event handling.
    """

    def __init__(
        self,
        agent: Agent,
        status_file: str | Path = "/tmp/agent.status.json",
        pid_file: str | Path = "/tmp/agent.pid",
        status_update_throttle_sec: float = 1.0,
    ):
        self.agent = agent
        self.status_path = Path(status_file).resolve()
        self.pid_path = Path(pid_file).resolve()
        self.throttle_sec = status_update_throttle_sec

        self._last_status_write: float = 0.0
        self._running = False
        self._success = False
        self._start_time: Optional[datetime] = None

        self._stop_event: Optional[asyncio.Event] = None

    # ============================================================
    # Status / PID (unchanged)
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
            "prompt": prompt,
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

    # ============================================================
    # Signals (unchanged)
    # ============================================================

    def _setup_signal_handlers(self, loop: asyncio.AbstractEventLoop) -> None:
        def shutdown_handler(sig: Optional[int] = None):
            name = signal.Signals(sig).name if sig else "Shutdown"
            print(f"\n[{name}] Stopping agent...")

            if self._stop_event:
                self._stop_event.set()

            self._write_status(
                "shutting_down",
                message="Received shutdown signal",
                force=True,
            )

            for task in asyncio.all_tasks(loop):
                if task is not asyncio.current_task():
                    task.cancel()

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, shutdown_handler, sig)

    # ============================================================
    # Main run – now uses Agent instead of AgentRunner
    # ============================================================

    async def run(
        self,
        prompt: str,
        system_prompt: str,
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

        # ── PID safety ─────────────────────────────────────────────
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
        self._write_status("starting", prompt=prompt, force=True)

        # ── Create the new Agent instance ──────────────────────────

        current_iteration = 0
        current_phase = "idle"

        try:
            async for event_type, payload in self.agent.run(
                user_prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.3,
                max_tokens=max_tokens,
                stop_event=self._stop_event,
            ):
                if self._stop_event.is_set():
                    print("\n🛑 Stop event received")
                    break

                # ── Phase tracking (new in Agent) ──────────────────
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

                # ── Planning complete (new) ────────────────────────
                elif event_type == "planning_complete":
                    steps = payload.get("steps", [])
                    goal = payload.get("goal", "")
                    print(f"\n📋 Planning complete | Goal: {goal}")
                    print("Steps:")
                    for i, step in enumerate(steps, 1):
                        print(f"  {i}. {step}")

                # ── Iteration start ────────────────────────────────
                elif event_type == "step_start":
                    current_iteration = payload.get("iteration", current_iteration)
                    print(
                        f"\n[{current_iteration}] Iteration start (phase: {current_phase})"
                    )

                # ── Streaming content ──────────────────────────────
                elif event_type == "content_delta":
                    print(payload, end="", flush=True)

                # ── Tool lifecycle ─────────────────────────────────
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

                # ── Reflection ─────────────────────────────────────
                elif event_type == "reflection_triggered":
                    print("\n🤔 Reflection:\n", str(payload).strip())

                # ── Final summary ──────────────────────────────────
                elif event_type == "final_summary":
                    print("\n📊 FINAL REPORT\n", str(payload).strip())

                # ── Finish ─────────────────────────────────────────
                elif event_type == "finish":
                    reason = payload.get("reason", "unknown")
                    print(f"\n🏁 Finished: {reason}")

                    if reason in ("casual_complete", "task_complete", "normal"):
                        self._success = True
                    elif reason == "max_iterations_reached":
                        print("⚠️ Max iterations reached")
                    elif reason == "reflection_stuck":
                        print("⚠️ Agent stuck in reflection loop")

                # ── Confirmation (headless → log only) ─────────────
                elif event_type == "needs_confirmation":
                    print(
                        "\n⚠️ Confirmation required for dangerous tool (skipped in headless mode)"
                    )
                    self._write_status(
                        "needs_confirmation",
                        message="Dangerous tool requires user confirmation",
                        force=True,
                    )

                # ── Errors / cancel / warnings ─────────────────────
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

            return self._success
