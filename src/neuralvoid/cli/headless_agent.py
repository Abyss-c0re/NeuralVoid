from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from neuralcore.agents.agent_core import AgentRunner  # adjust import as needed


class HeadlessAgentRunner:
    """
    Runs an agent in a headless/shell-friendly way with:
    - Graceful shutdown (Ctrl+C / SIGTERM)
    - Status reporting via JSON file
    - PID file to allow external stop/status checks
    - Optional custom locations for status & pid files

    ADAPTED (March 2026) to match the improved AgentRunner:
    • Now correctly sets SUCCESS for casual/non-agentic chats ("hello", direct replies)
    • Recognises the new "finish" event (with reason="casual_complete" or "task_complete")
    • Uses real iteration number from "step_start" payload (no more off-by-one on streaming)
    • Removed obsolete handlers ("final_answer", duplicate "tool_complete", progress_summary, etc.)
    • Keeps 100% compatible yields/parameters with the current AgentRunner.run()
    """

    def __init__(
        self,
        status_file: str | Path = "/tmp/agent.status.json",
        pid_file: str | Path = "/tmp/agent.pid",
        status_update_throttle_sec: float = 1.2,
    ):
        self.status_path = Path(status_file).resolve()
        self.pid_path = Path(pid_file).resolve()
        self.throttle_sec = status_update_throttle_sec

        self._last_status_write: float = 0.0
        self._running = False
        self._success = False
        self._start_time: Optional[datetime] = None

    def _write_status(
        self,
        status: str,
        prompt: Optional[str] = None,
        iteration: Optional[int] = None,
        last_tool: Optional[str] = None,
        message: Optional[str] = None,
        error: Optional[str] = None,
        force: bool = False,
    ) -> None:
        now = datetime.utcnow()
        now_ts = now.timestamp()

        if not force and (now_ts - self._last_status_write) < self.throttle_sec:
            return

        data: dict[str, Any] = {
            "pid": os.getpid(),
            "status": status,
            "started_at": self._start_time.isoformat() + "Z"
            if self._start_time
            else None,
            "last_update": now.isoformat() + "Z",
            "prompt": prompt,
            "current_iteration": iteration,
            "last_tool": last_tool,
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

    def _setup_signal_handlers(self, loop: asyncio.AbstractEventLoop) -> None:
        def shutdown_handler(sig: Optional[int] = None):
            print(
                f"\n[{signal.Signals(sig).name if sig else 'Shutdown'}] Stopping agent..."
            )
            self._write_status("shutting_down", message="Received shutdown signal")
            for task in asyncio.all_tasks(loop):
                if task is not asyncio.current_task():
                    task.cancel()

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, shutdown_handler, sig)

    async def run(
        self,
        client: Any,
        prompt: str,
        dynamic_manager: Any,
        system_prompt: str,
        context_manager: Any,
        max_iterations: int = 25,
        max_tokens: int = 12000,
    ) -> bool:
        if self._running:
            raise RuntimeError("Agent is already running")

        self._running = True
        self._success = False
        self._start_time = datetime.utcnow()

        loop = asyncio.get_running_loop()
        self._setup_signal_handlers(loop)

        # PID / status file logic (unchanged)
        if self.pid_path.exists():
            try:
                old_pid = int(self.pid_path.read_text().strip())
                os.kill(old_pid, 0)
                print(
                    f"PID file exists and process {old_pid} is alive → refusing to start"
                )
                return False
            except OSError:
                print("Removing stale PID file")
                self.pid_path.unlink(missing_ok=True)

        self._write_pid()
        self._write_status("starting", prompt=prompt, force=True)

        runner = AgentRunner(
            client=client,
            max_iterations=max_iterations,
            max_reflections=3,
            temperature=0.3,
            max_tokens=max_tokens,
        )

        current_iteration = 0

        try:
            async for event_type, payload in runner.run(
                user_prompt=prompt,
                tools=dynamic_manager,
                system_prompt=system_prompt,
                context_manager=context_manager,
                max_tokens=max_tokens,
            ):
                # ── CRITICAL FIX: Use real iteration from AgentRunner (not per-event counter)
                if event_type == "step_start":
                    current_iteration = payload.get("iteration", current_iteration)
                    print(
                        f"\n[{current_iteration}] Starting iteration {current_iteration}"
                    )

                elif event_type == "content_delta":
                    print(payload, end="", flush=True)

                elif event_type == "tool_start":
                    print(f"\n🔧 TOOL: {payload['name']} {payload.get('args', {})}")
                    self._write_status(
                        "running",
                        prompt=prompt,
                        iteration=current_iteration,
                        last_tool=payload.get("name"),
                        message=f"Tool started: {payload.get('name')}",
                    )

                elif event_type == "tool_result":
                    name = payload.get("name", "unknown")
                    res = str(payload.get("result", ""))
                    if payload.get("error"):
                        print(f"\n❌ {name} failed: {res[:400]}")
                        self._write_status(
                            "error",
                            prompt=prompt,
                            iteration=current_iteration,
                            error=res[:300],
                        )
                    else:
                        print(
                            f"\n✅ {name} → {res[:400]}{'...' if len(res) > 400 else ''}"
                        )
                        self._write_status(
                            "running",
                            prompt=prompt,
                            iteration=current_iteration,
                            last_tool=name,
                            message=f"Tool result: {res[:150]}...",
                        )

                elif event_type == "tool_calls":
                    print(f"\nCalling {len(payload)} tool(s)...")

                elif event_type == "tool_call_delta":  # matches current AgentRunner
                    tool_name = payload.get("function", {}).get("name", "unknown")
                    print(f"\n🔧 {tool_name} (streaming tool call...)")

                elif event_type == "reflection_triggered":
                    print("\n" + "=" * 60)
                    print("🤔 AGENT REFLECTION")
                    print("=" * 60)
                    print(payload.strip())
                    print("=" * 60)

                elif event_type == "review_phase":
                    print("\n---\n🔍 REVIEW PHASE\n---\n")
                    print(str(payload).strip())
                    print("---\n")

                elif event_type == "final_summary":
                    print("\n" + "=" * 60)
                    print("📊 EXECUTION REPORT")
                    print("=" * 60)
                    print(payload.strip())
                    print("=" * 60)

                elif event_type == "finish":
                    reason = payload.get("reason", "unknown")
                    print(f"\n🏁 Loop finished: {reason}")
                    if reason in ("casual_complete", "task_complete", "normal"):
                        self._success = True  # ← CRITICAL: casual "hello" now succeeds
                    elif reason == "max_iterations_reached":
                        print("⚠️ Max iterations reached")
                    elif reason == "reflection_stuck":
                        print("⚠️ Agent stuck after reflections")

                elif event_type == "llm_finish":
                    print("\n✅ LLM finished generating response")

                elif event_type in ("error", "warning", "cancelled"):
                    print(f"\n[{event_type.upper()}] {payload}")
                    if event_type == "error":
                        self._write_status("error", error=str(payload), force=True)

                elif event_type == "needs_confirmation":
                    print("\n⚠️ Needs confirmation — skipping in headless mode")

                # Removed obsolete handlers (final_answer, progress_summary, continuation_query_added,
                # duplicate tool_complete) — they are no longer emitted by the current AgentRunner

        except asyncio.CancelledError:
            print("\nAgent loop cancelled")
            self._write_status("cancelled", message="Cancelled via signal", force=True)
            self._success = False

        except Exception as exc:
            print(f"\nUnexpected error: {exc}")
            self._write_status("error", error=str(exc), force=True)
            self._success = False

        finally:
            self._running = False
            if self._success:
                self._write_status("success", force=True)
            else:
                current = (
                    self.status_path.read_text() if self.status_path.exists() else "{}"
                )
                try:
                    data = json.loads(current)
                    if data.get("status") not in ("success", "cancelled", "error"):
                        self._write_status("failed", force=True)
                except Exception:
                    self._write_status("failed", force=True)

            self._cleanup_files()

            print("\n" + "=" * 80)
            print(f"STATUS: {'SUCCESS' if self._success else 'FAILED'}")
            print("=" * 80)

            return self._success
