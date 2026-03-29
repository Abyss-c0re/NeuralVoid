import asyncio
import json
from enum import Enum
from neuralcore.agents.state import AgentState
from neuralcore.actions.actions import ActionSet
from neuralcore.workflows.registry import workflow
from neuralcore.utils.logger import Logger

from typing import AsyncIterator, Dict, Any, List, Optional, Tuple

logger = Logger.get_logger()


class AgentFlow:
    FINAL_ANSWER_MARKER = "[FINAL_ANSWER_COMPLETE]"

    class Phase(str, Enum):
        IDLE = "idle"
        CHAT = "chat"
        PLAN = "plan"
        EXECUTE = "execute"
        WAIT = "wait"
        FINALIZE = "finalize"

    def __init__(self, agent):

        self.agent = agent
        self.engine = agent.workflow
        workflow.bind_to_engine(self.engine, instance=self)

    # ==================== SYSTEM PROMPTS ====================

    def _build_chat_system_prompt(self) -> str:
        return f"""You are a helpful Deploy Agent.
        Speak naturally and concisely.
        - Simple questions → answer directly.
        - Complex requests → call **RequestComplexAction**.
        - When you see [DEPLOYMENT COMPLETE], respond friendly.

        Current goal: {self.agent.goal or "General assistance"}"""

    def _build_sub_agent_system_prompt(
        self, task_desc: str, assigned_tools: List[str]
    ) -> str:
        tools_hint = (
            f"\n\nAvailable tools: {', '.join(assigned_tools[:15])}{', ...' if len(assigned_tools) > 15 else ''}"
            if assigned_tools
            else ""
        )
        return f"""You are a precise sub-agent executing **ONE single micro-task only**.

        TASK: {task_desc}{tools_hint}

        CRITICAL RULES:
        - Complete ONLY this exact task.
        - If the task involves reading a file, use open_file_async or open_file_sync directly. Do NOT rely on BrowseTools for obvious file operations.
        - When you have finished the task, output a short summary and end with exactly: {AgentFlow.FINAL_ANSWER_MARKER}
        - Never mention other steps or the overall project."""

    # ==================== CHAT LOOP (battle-tested - unchanged) ====================

    @workflow.set(
        "deploy_chat",
        name="deploy_chat_loop",
        toolsets=["DeployControls", "ContextManager"],
        dynamic_allowed=True,
    )
    async def _wf_deploy_chat_loop(self, iteration: int, state: AgentState):
        if iteration == 0:
            state.phase = self.Phase.CHAT
            yield ("phase_changed", {"phase": "chat"})
            logger.info(f"Agent '{self.agent.name}' → Chat mode started")

        # self.agent.manager.load_toolsets("DeployControls")

        while True:
            try:
                raw_msg = await asyncio.wait_for(
                    self.agent.message_queue.get(), timeout=5.0
                )
            except asyncio.TimeoutError:
                if (
                    getattr(self.agent, "_stop_event", None)
                    and self.agent._stop_event.is_set()
                ):
                    break
                continue
            except asyncio.CancelledError:
                break

            if isinstance(raw_msg, dict) and "event" in raw_msg:
                ev = raw_msg["event"]
                if ev in ("sub_task_completed", "sub_task_failed"):
                    yield (ev, raw_msg)
                    await self.agent.post_system_message(
                        f"[STEP {ev.replace('sub_task_', '').upper()}] {raw_msg.get('task_id')}"
                    )
                elif ev == "switch_workflow":
                    self.engine.switch_workflow(raw_msg.get("name", "deploy_chat"))
                self.agent.message_queue.task_done()
                continue

            content = (
                raw_msg.get("content", "")
                if isinstance(raw_msg, dict)
                else str(raw_msg)
            )
            if not content.strip():
                self.agent.message_queue.task_done()
                continue

            messages = await self.agent.context_manager.provide_context(
                query=content, chat=True, system_prompt=self._build_chat_system_prompt()
            )

            async for ev, pl in self._process_user_message_with_llm(messages, state):
                yield ev, pl

            self.agent.message_queue.task_done()

    # ==================== NEW ORCHESTRATOR (matches your AgentState) ====================

    @workflow.set("orchestrator", name="plan_microtasks")
    async def _wf_plan_microtasks(self, iteration: int, state: AgentState):
        if state.planned_tasks:  # already planned
            return

        state.phase = self.Phase.PLAN
        yield ("phase_changed", {"phase": "plan"})

        prompt = f"""Break this task into 4-8 small independent micro-tasks.

        TASK: {self.agent.task}

        For each micro-task suggest the most relevant tools (e.g. open_file_async, write_file, grep, replace_block, etc.).

        Return ONLY JSON:
        {{
        "microtasks": [
            {{"description": "...", "suggested_tools": ["tool1", "tool2"]}},
            ...
        ]
        }}"""

        raw = await self.agent.client.chat([{"role": "user", "content": prompt}])
        try:
            data = json.loads(raw)
            state.planned_tasks = [t["description"] for t in data.get("microtasks", [])]
            state.task_tool_assignments = {
                i: t.get("suggested_tools", [])
                for i, t in enumerate(data.get("microtasks", []))
            }
        except Exception:
            state.planned_tasks = [self.agent.task]
            state.task_tool_assignments = {0: []}

        state.current_task_index = 0
        state.task_id_map = {}
        yield (
            "info",
            f"Planned {len(state.planned_tasks)} micro-tasks with tool hints",
        )

    @workflow.set("orchestrator", name="launch_next_subtask")
    async def _wf_launch_next_subtask(self, iteration: int, state: AgentState):
        """
        Launch all remaining micro-tasks in parallel for the current task index.
        Each micro-task is mapped to a sub-agent.
        Ensures tasks are registered in self.agent.sub_tasks before continuing.
        """
        if state.current_task_index >= len(state.planned_tasks):
            state.is_complete = True
            return

        tasks_to_launch = list(
            enumerate(
                state.planned_tasks[state.current_task_index :],
                start=state.current_task_index,
            )
        )

        launched_ids = []

        for idx, task_desc in tasks_to_launch:
            assigned_tools = state.task_tool_assignments.get(idx, [])
            name = f"Step {idx + 1}/{len(state.planned_tasks)}: {task_desc[:55]}..."

            task_id = await self.agent.start_complex_deployment(
                task_description=task_desc,
                user_facing_name=name,
                assigned_tools=assigned_tools or None,
                temperature=0.25,
                custom_system_prompt=self._build_sub_agent_system_prompt(
                    task_desc, assigned_tools
                ),
            )

            # Wait for the task to appear in sub_tasks
            wait_time = 0.0
            while task_id not in self.agent.sub_tasks and wait_time < 5.0:
                await asyncio.sleep(0.05)
                wait_time += 0.05

            if task_id not in self.agent.sub_tasks:
                logger.warning(f"Task {task_id} not registered in sub_tasks after 5s.")

            # Register in orchestrator state
            launched_ids.append(task_id)
            state.task_id_map[idx] = task_id

            # Update step number safely
            if task_id in self.agent.sub_tasks:
                self.agent.sub_tasks[task_id]["step_number"] = idx + 1

            yield (
                "sub_agent_launched",
                {
                    "step": idx + 1,
                    "task_id": task_id,
                    "description": task_desc,
                    "assigned_tools": assigned_tools,
                },
            )
            logger.info(
                f"Launched sub-task {idx + 1} → {task_id} with tools: {assigned_tools}"
            )

        state.sub_task_ids = launched_ids
        state.current_task_index = len(state.planned_tasks)

    @workflow.set("orchestrator", name="wait_for_subtask")
    async def _wf_wait_for_subtask(self, iteration: int, state: AgentState):
        """
        Wait for all currently launched sub-tasks in state.sub_task_ids to complete.
        Handles multiple sub-tasks running in parallel.
        """
        if not state.sub_task_ids:
            return

        pending_tasks = set(state.sub_task_ids)

        while pending_tasks:
            for task_id in list(pending_tasks):
                task = self.agent.sub_tasks.get(task_id)
                if task and task.get("status") in ("completed", "failed", "cancelled"):
                    yield (
                        "subtask_done",
                        {"task_id": task_id, "status": task.get("status")},
                    )
                    pending_tasks.remove(task_id)
                else:
                    yield ("waiting_for_subtask", {"task_id": task_id})
            await asyncio.sleep(0.1)

        # Once all sub-tasks are done, clear the sub_task_ids list
        state.sub_task_ids = []

        # Current task index can now advance to the end of the batch
        state.current_task_index = len(state.planned_tasks)

    @workflow.set("orchestrator", name="check_orchestrator_complete")
    async def _wf_check_orchestrator_complete(self, iteration: int, state: AgentState):
        if state.current_task_index < len(state.planned_tasks):
            state.is_complete = False
            return

        # All done
        state.phase = self.Phase.FINALIZE
        yield ("phase_changed", {"phase": "finalize"})

        summary = await self._generate_user_friendly_summary(state)
        yield ("llm_response", {"full_reply": summary, "is_complete": True})
        await self.agent.context_manager.add_message("assistant", summary)

        await self.agent.post_control(
            {"event": "switch_workflow", "name": "deploy_chat"}
        )

        yield (
            "finish",
            {
                "reason": "orchestrator_complete",
                "total_steps": len(state.planned_tasks),
            },
        )

    @workflow.set(
        "sub_agent_execute",
        name="llm_stream",
        description="Streams LLM response and extracts tool calls.",
    )
    async def _wf_llm_stream(self, iteration: int, state: AgentState):
        async for ev, pl in self._llm_stream_with_tools(
            iteration, state
        ):  # ← now self.
            if ev == "llm_response" and isinstance(pl, dict):
                state.full_reply = pl.get("full_reply", "")
                state.tool_calls = pl.get("tool_calls", [])
                state.is_complete = pl.get("is_complete", False)
            yield (ev, pl)
        self.engine._log_iteration_state(iteration, state)

    # ===================================================================
    # EXECUTORS — NOW INSIDE AgentFlow (relocated)
    # ===================================================================

    async def _llm_stream_with_tools(
        self,
        iteration: int,
        state: AgentState,
        tools: Optional[ActionSet] = None,
        is_chat_mode: bool = False,
    ) -> AsyncIterator[Tuple[str, Any]]:

        if is_chat_mode:
            raise RuntimeError(
                "is_chat_mode=True should not reach _llm_stream_with_tools"
            )

        state.phase = self.Phase.EXECUTE
        yield ("phase_changed", {"phase": state.phase.value})

        messages = await self.agent.context_manager.provide_context(
            query=state.current_task or "Continue",
            max_input_tokens=self.agent.max_tokens,
            reserved_for_output=12000,
            system_prompt=self._build_objective_reminder(),
            include_logs=True,
        )

        async def executor_callback(name: str, args: dict):
            executor = self.agent.manager.get_executor(name, self.agent)
            if not executor:
                raise RuntimeError(f"No executor for tool '{name}'")
            maybe = executor(**args)
            return await maybe if asyncio.iscoroutine(maybe) else maybe

        # ====================== MAIN STREAM LOOP WITH BROWSER RESTART SUPPORT ======================
        while True:  # Allow restart for ToolBrowser
            tool_browser_detected = False
            text_buffer = ""
            tool_results = []

            queue = await self.agent.client.stream_with_tools(
                messages=messages,  # Reuse same messages on restart
                tools=tools or self.agent.manager.get_llm_tools(),
                temperature=self.agent.temperature,
                max_tokens=self.agent.max_tokens,
                tool_choice="auto",
                executor_callback=executor_callback,
            )

            try:
                async for item in self.agent.client._drain_queue(queue):
                    if item is None:
                        continue
                    if not isinstance(item, tuple) or len(item) != 2:
                        continue

                    kind, payload = item

                    if kind == "content":
                        text_buffer += payload
                        yield ("content_delta", payload)

                    elif kind in ("tool_delta", "tool_complete", "needs_confirmation"):
                        if isinstance(payload, dict):
                            tool_name = payload.get("tool_name") or payload.get("name")
                            result = payload.get("result") or payload.get("output")

                            # === TOOLBROWSER DETECTION ===
                            if tool_name and "BrowseTools":  # adjust name if needed
                                tool_browser_detected = True
                                logger.info(
                                    f"ToolBrowser detected ({tool_name}). Restarting stream with same prompt."
                                )
                                # Do NOT add to context, do NOT yield final reply yet
                                break  # break inner loop to restart

                            # Normal tool result handling
                            if result:
                                try:
                                    # ... your existing external content storage code ...
                                    title = f"{tool_name} result"
                                    summary = (
                                        result.get("summary")
                                        or result.get("message")
                                        or str(result)
                                        if isinstance(result, dict)
                                        else str(result)
                                    )
                                    await self.agent.context_manager.add_external_content(
                                        source_type=f"task_result_{tool_name}",
                                        content=f"[{title}] {summary}\nMetadata: {result if isinstance(result, dict) else {}}",
                                        metadata={"task": tool_name},
                                    )
                                except Exception as e:
                                    logger.warning(
                                        f"Failed to store external content: {e}"
                                    )

                            if isinstance(result, str) and result.strip():
                                tool_results.append(result.strip())

                    elif kind == "finish":
                        break
                    elif kind in ("error", "cancelled"):
                        yield (kind, payload)
                        return

            except asyncio.CancelledError:
                yield ("cancelled", "Task cancelled")
                return
            except Exception as e:
                logger.error(f"Stream error: {e}", exc_info=True)
                yield ("error", str(e))
                return

            # ====================== POST-STREAM HANDLING ======================
            if tool_browser_detected:
                # Restart the stream (same prompt, no context pollution)
                continue

            # Normal completion path
            final_reply = text_buffer.strip()
            if not final_reply and tool_results:
                final_reply = "\n\n".join(tool_results)
            if not final_reply:
                final_reply = "✅ Tool executed successfully."

            yield (
                "llm_response",
                {
                    "full_reply": final_reply,
                    "tool_calls": [],
                    "is_complete": True,
                },
            )

            await self.agent.context_manager.add_message("assistant", final_reply)

            # Sub-agent parent context handling (unchanged)
            if getattr(self.agent, "sub_agent", False):
                parent_agent = getattr(self.agent, "parent", None)
                if parent_agent:
                    try:
                        await parent_agent.context_manager.add_external_content(
                            source_type="sub_task_final_reply",
                            content=final_reply,
                            metadata={"origin": self.agent.agent_id},
                        )
                    except Exception as e:
                        logger.warning(f"Failed to store final reply in parent: {e}")
            break  # Exit outer loop on normal completion

    def _build_objective_reminder(self) -> str:
        """You can keep or move this helper if it exists elsewhere."""
        return f"Current goal: {self.agent.goal}"

    async def _process_user_message_with_llm(
        self, messages: List[Dict], state: AgentState
    ) -> AsyncIterator[Tuple[str, Any]]:
        """Process message in chat mode. Supports ToolBrowser restart without polluting context."""

        logger.debug("=== ENTERING _process_user_message_with_llm ===")

        async def executor_callback(name: str, args: dict):
            logger.debug(f"[TOOL CALL] Executing {name} with args {args}")
            executor = self.agent.manager.get_executor(name, self.agent)
            if not executor:
                raise RuntimeError(f"No executor for tool '{name}'")
            maybe = executor(**args)
            result = await maybe if asyncio.iscoroutine(maybe) else maybe
            return result

        if (
            hasattr(self.agent.client, "_current_stop_event")
            and self.agent.client._current_stop_event
        ):
            self.agent.client._current_stop_event.clear()

        # ====================== STREAM WITH TOOLBROWSER RESTART SUPPORT ======================
        while True:  # Allow restart when ToolBrowser is called
            tool_browser_detected = False
            text_buffer = ""
            tool_results = []
            complex_action_called = False
            complex_reason = ""

            queue = await self.agent.client.stream_with_tools(
                messages=messages,  # Reuse same messages on restart
                tools=self.agent.manager.get_action_set("DynamicCore"),
                temperature=self.agent.temperature,
                max_tokens=self.agent.max_tokens,
                tool_choice="auto",
                executor_callback=executor_callback,
            )

            try:
                async for item in self.agent.client._drain_queue(queue):
                    if item is None:
                        continue
                    if not isinstance(item, tuple) or len(item) != 2:
                        continue

                    kind, payload = item

                    if kind == "content":
                        text_buffer += payload
                        yield ("content_delta", payload)

                    elif kind in ("tool_delta", "tool_complete", "needs_confirmation"):
                        if isinstance(payload, dict):
                            tool_name = payload.get("tool_name") or payload.get("name")
                            result = payload.get("result") or payload.get("output")

                            # === TOOLBROWSER RESTART LOGIC ===
                            if tool_name and ("BrowseTools" in tool_name):
                                tool_browser_detected = True
                                logger.info(
                                    f"[ToolBrowser Restart] Detected in chat mode: {tool_name}. Restarting stream (no context added)."
                                )
                                break  # Restart stream with same prompt

                            if tool_name == "RequestComplexAction":
                                complex_action_called = True
                                complex_reason = payload.get("args", {}).get(
                                    "reason", ""
                                )

                            # Normal result handling
                            if result:
                                try:
                                    title = f"{tool_name} result"
                                    if isinstance(result, dict):
                                        summary = (
                                            result.get("summary")
                                            or result.get("message")
                                            or str(result)
                                        )
                                        metadata = result
                                    else:
                                        summary = str(result)
                                        metadata = {}

                                    await self.agent.context_manager.add_external_content(
                                        source_type=f"task_result_{tool_name}",
                                        content=f"[{title}] {summary}\nMetadata: {metadata}",
                                        metadata={
                                            "task": tool_name,
                                            **(metadata or {}),
                                        },
                                    )
                                except Exception as e:
                                    logger.warning(
                                        f"Failed to store external content: {e}"
                                    )

                            if isinstance(result, str) and result.strip():
                                tool_results.append(result.strip())

                    elif kind == "finish":
                        logger.debug("STREAM FINISHED")
                        break

                    elif kind in ("error", "cancelled"):
                        yield (kind, payload)
                        return

            except asyncio.CancelledError:
                yield ("cancelled", "Task cancelled")
                return
            except Exception as e:
                logger.error(f"Stream error: {e}", exc_info=True)
                yield ("error", str(e))
                return

            # If ToolBrowser was called → restart the stream without adding to context
            if tool_browser_detected:
                continue

            # ====================== NORMAL COMPLETION PATH ======================
            final_reply = text_buffer.strip()

            if complex_action_called:
                logger.info(
                    f"[CHAT → ORCHESTRATOR] Complex task detected: {complex_reason[:100]}..."
                )

                self.agent.task = complex_reason
                self.agent.goal = complex_reason

                try:
                    self.engine.switch_workflow("default")
                except Exception as e:
                    logger.warning(f"Direct switch failed: {e}")
                    await self.agent.post_control(
                        {"event": "switch_workflow", "name": "default"}
                    )

                final_reply = (
                    f"✅ Understood. Starting **multi-step orchestration**:\n"
                    f"**{complex_reason[:120]}{'...' if len(complex_reason) > 120 else ''}**\n\n"
                    f"Planning steps → deploying specialized sub-agents."
                )

                yield (
                    "llm_response",
                    {"full_reply": final_reply, "tool_calls": [], "is_complete": True},
                )
                await self.agent.context_manager.add_message("assistant", final_reply)
                return

            # Normal reply
            if not final_reply and tool_results:
                final_reply = "\n\n".join(tool_results)
            elif not final_reply:
                final_reply = "✅ Tool executed successfully."

            logger.info(f"FINAL REPLY being sent to user:\n{final_reply}")

            yield (
                "llm_response",
                {"full_reply": final_reply, "tool_calls": [], "is_complete": True},
            )
            await self.agent.context_manager.add_message("assistant", final_reply)
            logger.debug("Message added to context")
            break  # Exit restart loop

    async def _generate_user_friendly_summary(self, state: AgentState) -> str:
        """Generates a natural, friendly summary that will be shown to the user
        right before returning to chat mode."""

        tool_results_str = "\n".join(
            f"• {r['name']}: {str(r.get('result', ''))[:400]}"
            for r in self.agent.tool_results[-12:]  # last 12 results max
        )

        prompt = f"""You are a helpful Deploy Agent. The complex task has just finished.

    Task: {self.agent.task}
    Goal: {self.agent.goal or "General deployment assistance"}

    What was actually done (tool results):
    {tool_results_str or "No tool results recorded."}

    Write a **friendly, concise, natural** message to the user (2–6 sentences max).
    - Celebrate what was accomplished
    - Mention any important outcomes or warnings
    - End by saying we're back in normal chat mode and ask how else you can help

    Tone: professional but warm and clear. No JSON. No technical jargon unless necessary.
    """

        try:
            summary = await self.agent.client.chat(
                [{"role": "user", "content": prompt}], temperature=0.7
            )
            return summary.strip()
        except Exception:
            # Fallback
            return (
                f"✅ **Task completed successfully!**\n\n"
                f"I have finished the deployment task: **{self.agent.task}**.\n"
                f"We are now back in normal chat mode. How else can I help you?"
            )

    async def _generate_sub_agent_summary(self, state: AgentState) -> str:
        return f"✅ Sub-task completed.\n\nKey results recorded in shared context."
