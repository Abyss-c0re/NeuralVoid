import json
import asyncio

from enum import Enum
from neuralcore.agents.state import AgentState

from neuralcore.workflows.registry import workflow
from neuralcore.utils.logger import Logger
from neuralcore.workflows.executors import AgentExecutors
from neuralcore.utils.formatting import prepare_chat_messages
from neuralcore.utils.prompt_builder import PromptBuilder


from typing import List, Dict, Tuple, Any, AsyncIterator

logger = Logger.get_logger()


# ==================== CONDITIONS ====================
@workflow.condition("goal_achieved")
def goal_achieved(state: AgentState, args=None):
    """Break ONLY when we have a real final answer.
    Now much stricter for lightweight context."""
    if getattr(state, "mode", None) == "casual":
        return False

    full_reply = getattr(state, "full_reply", "").strip()
    is_complete = getattr(state, "is_complete", False)
    marker = "[FINAL_ANSWER_COMPLETE]"

    has_marker = marker in full_reply
    has_real_content = len(full_reply) > 30 and not any(
        w in full_reply.lower()
        for w in ["error", "failed", "try again", "still working"]
    )

    # Stronger check: either marker OR explicit "task complete" language + no pending sub-tasks
    explicit_done = any(
        phrase in full_reply.lower()
        for phrase in [
            "task complete",
            "goal achieved",
            "all done",
            "finished successfully",
        ]
    )

    all_subtasks_done = (
        len(state.planned_tasks) == 0
        or state.current_task_index >= len(state.planned_tasks) - 1
    )

    should_break = (
        (is_complete or has_marker or explicit_done)
        and has_real_content
        and all_subtasks_done
    )

    if should_break:
        logger.info(
            f"[GOAL ACHIEVED] Triggered | marker={has_marker} | explicit_done={explicit_done} | subtasks_done={all_subtasks_done}"
        )
    return should_break


@workflow.condition("subtask_complete")
def subtask_complete(state: AgentState, args=None):
    """Break when sub-task is really done — stricter for multi-step."""
    full_reply = getattr(state, "full_reply", "").strip()
    marker = "[FINAL_ANSWER_COMPLETE]"

    has_marker = marker in full_reply
    explicit_done = any(
        phrase in full_reply.lower()
        for phrase in ["sub-task complete", "this step done", "finished this task"]
    )

    return has_marker or explicit_done or getattr(state, "is_complete", False)


@workflow.condition("has_final_reply")
def has_final_reply(state: AgentState, args=None):
    """Chat loop break — keep simple but require meaningful content."""
    full_reply = getattr(state, "full_reply", "").strip()
    has_tools = bool(
        getattr(state, "tool_calls", None) and len(getattr(state, "tool_calls", [])) > 0
    )
    return bool(full_reply) and not has_tools and len(full_reply) > 25


@workflow.condition("max_tool_calls_reached")
def max_tool_calls_reached(state: AgentState, args=None):
    """Per-turn tool call limit (prevents runaway tool loops in one response)"""
    max_allowed = args.get("max", 10) if isinstance(args, dict) else 10
    tool_calls = getattr(state, "tool_calls", None) or []
    return len(tool_calls) >= max_allowed


@workflow.condition("too_many_empty_loops")
def too_many_empty_loops(state: AgentState, args=None):
    """Prevents infinite empty LLM replies"""
    max_allowed = args.get("max", 5) if isinstance(args, dict) else 5
    return getattr(state, "empty_loops", 0) >= max_allowed


@workflow.condition("max_action_restarts_reached")
def max_action_restarts_reached(state: AgentState, args=None):
    """Prevents excessive 'Next Action' loops"""
    max_allowed = args.get("max", 8) if isinstance(args, dict) else 8
    return getattr(state, "action_restarts", 0) >= max_allowed


@workflow.condition("chat_safety_break")
def chat_safety_break(state: AgentState, args=None):
    """
    Combined safety condition for chat loops.
    Returns True if ANY safety limit is reached.
    Logs exactly what caused the break.
    """
    reasons = []

    # 1. Check has_final_reply (normal good case)
    full_reply = getattr(state, "full_reply", "").strip()
    has_tools = bool(
        getattr(state, "tool_calls", None) and len(getattr(state, "tool_calls", [])) > 0
    )
    has_final = bool(full_reply) and not has_tools and len(full_reply) > 15

    if has_final:
        reasons.append("has_final_reply")

    # 2. Per-turn tool calls
    max_tool_per_turn = (
        args.get("max_tool_per_turn", 10) if isinstance(args, dict) else 10
    )
    tool_calls = getattr(state, "tool_calls", None) or []
    if len(tool_calls) >= max_tool_per_turn:
        reasons.append(f"max_tool_calls_reached({len(tool_calls)}/{max_tool_per_turn})")

    # 3. Too many empty loops
    max_empty = args.get("max_empty", 5) if isinstance(args, dict) else 5
    empty_loops = getattr(state, "empty_loops", 0)
    if empty_loops >= max_empty:
        reasons.append(f"too_many_empty_loops({empty_loops}/{max_empty})")

    # 4. Too many action restarts
    max_restarts = args.get("max_restarts", 8) if isinstance(args, dict) else 8
    action_restarts = getattr(state, "action_restarts", 0)
    if action_restarts >= max_restarts:
        reasons.append(f"max_action_restarts_reached({action_restarts}/{max_restarts})")

    # Log what actually caused the break
    if reasons:
        reason_str = " + ".join(reasons)
        logger.info(
            f"[SAFETY BREAK] Triggered by: {reason_str} | full_reply_len={len(full_reply)}"
        )
        return True

    return False


# ==================== LOOPS ====================
@workflow.loop("chat_task_loop", max_iterations=3, break_condition="chat_safety_break")
async def chat_task_loop(agent, state: AgentState, messages: List[Dict]):
    """Inner decorated task loop for chat"""
    async for ev, pl in agent.flow.executors.chat_loop(messages, state):
        yield ev, pl


@workflow.loop(
    "agentic_task_loop", max_iterations=12, break_condition="subtask_complete"
)
async def agentic_task_loop(agent, state: AgentState, iteration: int = 0):
    """Inner decorated task loop for sub-agents"""
    async for ev, pl in agent.flow.executors.agentic_loop(iteration, state):
        yield ev, pl


@workflow.loop("chat_tool_loop", max_iterations=50, break_condition="goal_achieved")
async def chat_tool_loop(agent, state: AgentState):
    """Outer persistent chat loop"""
    while True:
        try:
            raw_msg = await asyncio.wait_for(agent.message_queue.get(), timeout=5.0)
        except asyncio.TimeoutError:
            if getattr(agent, "_stop_event", None) and agent._stop_event.is_step():
                break
            continue
        except asyncio.CancelledError:
            break

        content = (
            raw_msg.get("content", "") if isinstance(raw_msg, dict) else str(raw_msg)
        )
        if not content.strip():
            agent.message_queue.task_done()
            continue

        state.full_reply = ""
        state.tool_calls = None

        agent.manager.reset_to_default_package("deploy_chat_loop", agent.workflow)

        # Call the nested decorated inner loop
        async for ev, pl in chat_task_loop(
            agent,
            state,
            prepare_chat_messages(
                content, system_prompt=agent.flow._build_chat_system_prompt()
            ),
        ):
            yield ev, pl

        agent.message_queue.task_done()


@workflow.loop("agentic_loop", max_iterations=30, break_condition="subtask_complete")
async def agentic_loop(agent, state: AgentState):
    """Outer agentic loop"""
    async for ev, pl in agentic_task_loop(agent, state, iteration=0):
        yield ev, pl


# ==================== MAIN FLOWS ====================
class AgentFlow:
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
        self.executors = AgentExecutors(agent, self.Phase)
        self.agent.flow = self
        self.FINAL_ANSWER_MARKER = "[FINAL_ANSWER_COMPLETE]"
        workflow.bind_to_engine(self.engine, instance=self)

    # ==================== SYSTEM PROMPTS ====================

    def _inject_final_answer_instruction(self, base_prompt: str) -> str:
        """Strong, consistent instruction for all prompts."""
        return f"""{base_prompt}

    CRITICAL FINAL ANSWER RULE:
    When you have **fully completed** the assigned task/micro-task and verified all required outputs, you MUST end your response with **exactly** this marker on its own line:

    {self.FINAL_ANSWER_MARKER}

    Do not add anything after the marker. Use it only when the goal is 100% achieved."""

    # === UPDATED PROMPTS ===
    def _build_chat_system_prompt(self) -> str:
        base = f"""You are a helpful Deploy Agent that executes commands immediately.

        RULES:
        - Use tool browser to load missing tools.
        - Keep responses short and natural after tool results.
        - Current goal: {self.agent.state.goal or "General assistance"}"""
        return self._inject_final_answer_instruction(base)

    def _build_sub_agent_system_prompt(
        self, task_desc: str, assigned_tools: List[str]
    ) -> str:
        tools_hint = (
            f"\n\nAvailable tools: {', '.join(assigned_tools[:15])}{', ...' if len(assigned_tools) > 15 else ''}"
            if assigned_tools
            else ""
        )
        base = f"""You are a precise sub-agent executing **ONE single micro-task only**.

    TASK: {task_desc}{tools_hint}

    CRITICAL RULES:
    - Complete ONLY this exact task.
    - If the task involves reading a file, use open_file_async or open_file_sync directly.
    - When you have finished the task, output a short summary and end with exactly: {self.FINAL_ANSWER_MARKER}"""
        return self._inject_final_answer_instruction(base)

    # ==================== WORKFLOWS ====================
    @workflow.step("deploy_chat", name="deploy_chat_loop")
    async def _wf_deploy_chat_loop(self, iteration: int, state: AgentState):
        """Persistent chat mode"""
        if iteration == 0:
            state.phase = self.Phase.CHAT
            yield ("phase_changed", {"phase": "chat"})
            logger.info(f"Agent '{self.agent.name}' → Chat mode started")

        while True:
            if (
                getattr(self.agent, "_stop_event", None)
                and self.agent._stop_event.is_step()
            ):
                yield ("cancelled", "User requested stop")
                break

            # Use the outer decorated loop
            async for ev, pl in self.agent.execute_loop(
                "chat_tool_loop", initial_state=state
            ):
                yield ev, pl

            await asyncio.sleep(0.01)

    @workflow.step("sub_agent_execute", name="llm_stream")
    async def _wf_llm_stream(self, iteration: int, state: AgentState):
        """Sub-agent execution step"""
        async for ev, pl in self.agent.execute_loop(
            "agentic_loop", initial_state=state
        ):
            if ev == "llm_response" and isinstance(pl, dict):
                state.full_reply = pl.get("full_reply", "")
                state.tool_calls = pl.get("tool_calls", [])
                state.is_complete = pl.get("is_complete", False)
            yield ev, pl

        self.engine._log_iteration_state(iteration, state)

    # ==================== ORCHESTRATOR ====================

    @workflow.step("orchestrator", name="plan_microtasks")
    async def _wf_plan_microtasks(self, iteration: int, state: AgentState):
        if state.planned_tasks:  # already planned
            return

        state.phase = self.Phase.PLAN
        yield ("phase_changed", {"phase": "plan"})

        prompt = f"""Break this task into 5-8 small focused micro-tasks.

        TASK: {self.agent.state.task}

        Return ONLY valid JSON in this exact format:
        {{
        "microtasks": [
            {{
            "description": "Clear one-sentence description",
            "suggested_tools": ["tool1", "tool2"],
            "depends_on": null
            }},
            {{
            "description": "...",
            "suggested_tools": ["tool3"],
            "depends_on": "step_1"     # use previous description as reference or null
            }}
        ]
        }}

        Note: Use "depends_on": null for tasks that can start immediately.
        Use the first few words of a previous task's description as "depends_on" value if it depends on it."""

        raw = await self.agent.client.chat([{"role": "user", "content": prompt}])

        try:
            data = json.loads(raw)
            microtasks = data.get("microtasks", [])

            state.planned_tasks.clear()
            state.task_tool_assignments.clear()
            state.task_dependencies.clear()

            for i, task in enumerate(microtasks):
                state.planned_tasks.append(task.get("description", f"Task {i + 1}"))
                state.task_tool_assignments[i] = task.get("suggested_tools", [])
                state.task_dependencies[i] = task.get(
                    "depends_on"
                )  # can be None or string

            state.current_task_index = 0
            state.task_id_map.clear()

            yield (
                "info",
                f"Planned {len(state.planned_tasks)} micro-tasks with dependencies",
            )

        except Exception as e:
            logger.warning(f"Planning failed: {e}. Using single task fallback.")
            state.reset_for_new_task(new_task=self.agent.state.task)
            state.planned_tasks = [self.agent.state.task]
            state.task_tool_assignments = {0: []}
            state.task_dependencies = {0: []}
            state.current_task_index = 0
            state.ensure_dependencies_structure()

    @workflow.step("orchestrator", name="launch_next_subtask")
    async def _wf_launch_next_subtask(self, iteration: int, state: AgentState):
        """Launch tasks whose dependencies are satisfied (or have no dependency)."""
        if state.current_task_index >= len(state.planned_tasks):
            state.is_complete = True
            return

        launched_this_round = 0

        for idx in range(state.current_task_index, len(state.planned_tasks)):
            depends_on = state.task_dependencies.get(idx)

            # Skip if dependency not yet satisfied
            if depends_on and depends_on not in (None, "null", ""):
                dependency_satisfied = False
                for prev_idx in range(idx):
                    prev_task_id = state.task_id_map.get(prev_idx)
                    if prev_task_id:
                        prev_status = self.agent.sub_tasks.get(prev_task_id, {}).get(
                            "status"
                        )
                        if prev_status == "completed":
                            dependency_satisfied = True
                            break
                if not dependency_satisfied:
                    continue  # wait for dependency

            # Launch this task
            assigned_tools = state.task_tool_assignments.get(idx, [])
            task_desc = state.planned_tasks[idx]
            name = f"Step {idx + 1}/{len(state.planned_tasks)}: {task_desc[:55]}..."

            task_id = await self.agent.start_complex_deployment(
                task_description=task_desc,
                user_facing_name=name,
                assigned_tools=assigned_tools or None,
                temperature=0.25,
                custom_system_prompt=self._build_sub_agent_system_prompt(
                    task_desc, assigned_tools
                ),
                depends_on=depends_on if depends_on not in (None, "null", "") else None,
            )

            state.task_id_map[idx] = task_id
            launched_this_round += 1

            if task_id in self.agent.sub_tasks:
                self.agent.sub_tasks[task_id]["step_number"] = idx + 1

            yield (
                "sub_agent_launched",
                {
                    "step": idx + 1,
                    "task_id": task_id,
                    "description": task_desc,
                    "assigned_tools": assigned_tools,
                    "depends_on": depends_on,
                },
            )
            logger.info(
                f"Launched sub-task {idx + 1} → {task_id} (depends_on={depends_on})"
            )

        # Advance the index
        state.current_task_index += launched_this_round

    @workflow.step("orchestrator", name="wait_for_subtask")
    async def _wf_wait_for_subtask(self, iteration: int, state: AgentState):
        """Wait for currently launched sub-tasks to complete."""
        if not state.task_id_map:
            return

        # Get all currently launched but not yet completed task IDs
        pending_tasks = {
            task_id
            for task_id in state.task_id_map.values()
            if self.agent.sub_tasks.get(task_id, {}).get("status")
            not in ("completed", "failed", "cancelled")
        }

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
            await asyncio.sleep(0.2)

        # Do NOT force current_task_index to the end here — let launch logic control it
        # state.sub_task_ids = []   ← removed because we now use task_id_map

    @workflow.step("orchestrator", name="check_orchestrator_complete")
    async def _wf_check_orchestrator_complete(self, iteration: int, state: AgentState):
        """Check if all planned tasks have been launched and completed."""
        # All tasks must have been launched
        if state.current_task_index < len(state.planned_tasks):
            state.is_complete = False
            return

        # All launched tasks must be finished
        all_done = all(
            self.agent.sub_tasks.get(tid, {}).get("status")
            in ("completed", "failed", "cancelled")
            for tid in state.task_id_map.values()
        )

        if not all_done:
            state.is_complete = False
            return

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

    # ==================== HELPERS (all preserved) ====================

    async def _generate_user_friendly_summary(self, state: AgentState) -> str:
        tool_results_str = "\n".join(
            f"• {r['name']}: {str(r.get('result', ''))[:400]}"
            for r in self.agent.tool_results[-12:]
        )

        prompt = f"""You are a helpful Deploy Agent. The complex task has just finished.

    Task: {self.agent.state.task}
    Goal: {self.agent.state.goal or "General deployment assistance"}

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
            return (
                f"✅ **Task completed successfully!**\n\n"
                f"I have finished the deployment task: **{self.agent.state.task}**.\n"
                f"We are now back in normal chat mode. How else can I help you?"
            )

    async def _generate_sub_agent_summary(self, state: AgentState) -> str:
        return "✅ Sub-task completed.\n\nKey results recorded in shared context."

    async def _ensure_subtasks_planned(
        self, state: AgentState, original_query: str
    ) -> AsyncIterator[Tuple[str, Any]]:
        """Generic, LLM-driven structured planning.
        Guarantees task_dependencies remains Dict[int, List[int]]."""
        yield ("phase_changed", {"phase": "planning"})

        planning_prompt = PromptBuilder.task_decomposition(original_query)

        try:
            plan_text = await self.agent.client.chat(
                planning_prompt, temperature=0.0, max_tokens=1200
            )
            plan = json.loads(plan_text.strip())

            state.planned_tasks.clear()
            state.task_tool_assignments.clear()
            state.task_dependencies.clear()

            for i, step in enumerate(plan.get("steps", [])):
                state.planned_tasks.append(step.get("description", f"Task {i + 1}"))

                # Safe conversion to List[int]
                deps_raw = step.get("dependencies", [])
                if isinstance(deps_raw, (int, str)):
                    deps = [int(deps_raw)] if str(deps_raw).isdigit() else []
                elif isinstance(deps_raw, list):
                    deps = [
                        int(d)
                        for d in deps_raw
                        if isinstance(d, (int, str)) and str(d).isdigit()
                    ]
                else:
                    deps = []

                state.task_dependencies[i] = deps
                if step.get("suggested_tool_category"):
                    state.task_tool_assignments[i] = [step["suggested_tool_category"]]

            state.ensure_dependencies_structure()  # enforce type

            logger.info(
                f"[PLANNING] Created {len(state.planned_tasks)} sub-tasks with dependencies"
            )
            for i, t in enumerate(state.planned_tasks):
                deps = state.task_dependencies.get(i, [])
                logger.debug(f"  Step {i}: {t[:80]}... | deps={deps}")

            yield ("planning_complete", {"planned_tasks": state.planned_tasks})

        except Exception as e:
            logger.warning(f"Planning failed: {e}. Using single task fallback.")
            state.planned_tasks = [original_query]
            state.task_tool_assignments = {0: []}
            state.task_dependencies = {0: []}
            state.current_task_index = 0
            state.ensure_dependencies_structure()
            yield ("planning_fallback", {"reason": str(e)})
