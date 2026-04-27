import json
import asyncio

from enum import Enum

from neuralcore.agents.state import AgentState


from neuralcore.workflows.registry import workflow
from neuralcore.utils.logger import Logger

from neuralcore.utils.prompt_builder import PromptBuilder
from neuralcore.workflows.executors import (
    plan_tasks_unified,
    goal_driven_task_loop,
    classify_intent,
)

logger = Logger.get_logger()


# ==================== CONDITIONS ====================
@workflow.condition("goal_achieved")
def goal_achieved(state: AgentState, args=None) -> bool:
    if getattr(state, "mode", None) == "casual":
        return False

    full_reply = getattr(state, "full_reply", "").strip()
    is_complete = getattr(state, "is_complete", False)

    has_real_content = len(full_reply) > 50 and not any(
        w in full_reply.lower()
        for w in ["error", "failed", "try again", "still working", "reading file again"]
    )

    explicit_done = any(
        phrase in full_reply.lower()
        for phrase in [
            "task complete",
            "goal achieved",
            "all done",
            "finished successfully",
            "tool added",
            "implemented the new",
        ]
    )

    all_subtasks_done = (
        len(state.planned_tasks) == 0
        or state.current_task_index >= len(state.planned_tasks) - 1
    )

    # ==================== NEW: Marker + Tool Success Logic (relocated here) ====================
    marker = PromptBuilder.FINAL_ANSWER_MARKER
    has_marker_in_reply = marker in full_reply

    # Clean marker from final_reply if present (do this once, early)
    if has_marker_in_reply:
        state.full_reply = full_reply.replace(marker, "").strip()  # mutate safely

    last_success = getattr(state, "last_tool_success", None)
    tool_reported_success = bool(last_success and last_success.get("success"))

    strong_completion = has_marker_in_reply or tool_reported_success

    # Marker in any past tool result (kept for robustness)
    marker_in_history = any(
        marker in str(r.get("result", "")) for r in state.tool_results
    )

    # ==================== Final Decision ====================
    should_break = (
        is_complete
        and has_real_content
        and all_subtasks_done
        and (explicit_done or marker_in_history or strong_completion)
    )

    if should_break:
        logger.info(
            f"[GOAL ACHIEVED] Triggered | "
            f"explicit={explicit_done} | "
            f"marker_in_reply={has_marker_in_reply} | "
            f"marker_in_history={marker_in_history} | "
            f"strong_completion={strong_completion} | "
            f"index={state.current_task_index}/{len(state.planned_tasks)}"
        )

    return should_break


@workflow.condition("subtask_complete")
def subtask_complete(state: AgentState, args=None):
    """Break when sub-task is really done — stricter for multi-step."""
    full_reply = getattr(state, "full_reply", "").strip()
    marker = PromptBuilder.FINAL_ANSWER_MARKER

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


# ==================== LOOPS ====================
@workflow.loop("goal_driven_loop", max_iterations=None, break_condition="goal_achieved")
async def goal_driven_loop(agent, state: AgentState):
    """Inner decorated task loop for chat"""

    yield ("phase_changed", {"phase": "thinking"})

    async for ev, pl in goal_driven_task_loop(agent, state, "goal_driven_loop"):
        yield ev, pl


@workflow.loop("chat_tool_loop", max_iterations=None)
async def chat_tool_loop(agent, state: AgentState):
    """
    Stable outer persistent chat loop.
    - One-time heavy registration at startup only.
    - CASUAL → lightweight direct reply (no inner loop).
    - TASK → delegate to goal_driven_loop without full re-init.
    """
    logger.info("[CHAT TOOL LOOP] Outer loop started — persistent mode active")

    target_loop = "chat_tool_loop"

    content = await agent.wait_for_incoming_message(
        role="user", return_content_only=True
    )

    intent = await classify_intent(agent, content)
    logger.debug(f"Intent {intent}")

    if content:
        if intent == "CASUAL":
            logger.info("[CASUAL MODE] Pure basic chat — no inner loop")
            yield ("phase_changed", {"phase": "casual_chat"})

            casual_messages = await agent.context_manager.provide_context(
                query=content,
                max_input_tokens=agent.max_tokens * 0.65,
                reserved_for_output=agent.client.max_tokens * 0.35,
                include_logs=True,
                chat=True,
            )

            logger.debug(f"Chat Context: {(str(casual_messages))}")

            final_reply = await agent.client.chat(
                casual_messages,
                temperature=0.85,
                top_p=0.95,
                max_tokens=agent.client.max_tokens * 0.4,
            )

            await agent.add_message("assistant", final_reply)
            yield ("llm_response", {"full_reply": final_reply, "is_complete": True})

            agent.message_queue.task_done()
            state.reset_for_new_task()

            state.request_loop_restart(
                reason="Casual chat completed, waiting for next message",
                target_loop=target_loop,
            )
        else:
            # === TASK-DRIVEN MODE ===
            logger.info("[TASK-DRIVEN MODE] Delegating to goal_driven_loop")
            yield ("phase_changed", {"phase": "goal_driven"})
            state.reset_for_new_task(new_task=content)

            # ====================== ONE-TIME PLANNING ======================
            if not state.planned_tasks:
                async for event, payload in plan_tasks_unified(agent, state):
                    yield event, payload

            # Forward inner loop events
            async for event, payload in agent.execute_loop(
                "goal_driven_loop", initial_state=state
            ):
                yield event, payload

                if event in ("error", "cancelled", "loop_stopped"):
                    state.request_loop_stop(
                        reason=f"Inner loop signaled {event}", target_loop=target_loop
                    )
                    return

            logger.info(
                "[TASK-DRIVEN MODE] Inner goal_driven_loop completed — back to outer chat"
            )

            # ====================== FINAL SYNTHESIS (fallback only) ======================
            if state.goal_reached:
                yield ("phase_changed", {"phase": "generating_final_answer"})

                results = await agent.context_manager.provide_context(
                    query=content,
                    lightweight_agentic=True,
                    max_input_tokens=agent.client.max_tokens * 0.65,
                    reserved_for_output=agent.client.max_tokens * 0.4,
                    return_as_string=True,
                )
                logger.debug(f"Final Context: {results}")
                final_reply = await agent.client.chat(
                    results,
                    temperature=0.0,
                    top_p=0.1,
                    max_tokens=agent.client.max_tokens,
                )
                yield (
                    "llm_response",
                    {"full_reply": final_reply, "tool_calls": [], "is_complete": True},
                )
                logger.info("Task completed successfully → full reset")
                state.reset_for_new_task()
            else:
                logger.warning(
                    "Loop ended without explicit goal or restart – forcing restart"
                )
                state.request_loop_restart(
                    reason="Fallback restart", target_loop=target_loop
                )
                yield ("phase_changed", {"phase": "restarting_loop"})

            if not state.goal_reached:
                state.status = "idle"
                state.is_complete = True
            # Restart outer loop for next user input
            state.request_loop_restart(
                reason="Inner goal_driven_loop finished, returning to chat",
                target_loop=target_loop,
            )


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
        workflow.bind_to_engine(self.engine, instance=self)

    # ==================== WORKFLOWS ====================

    @workflow.step("deploy_chat", name="deploy_chat_loop")
    async def _wf_deploy_chat_loop(self, iteration: int, state: AgentState):
        """Persistent chat mode"""
        if iteration == 0:
            state.phase = self.Phase.CHAT
            yield ("phase_changed", {"phase": "chat"})
            logger.info(f"Agent '{self.agent.name}' → Chat mode started")

            # Use the outer decorated loop
        async for ev, pl in self.agent.execute_loop(
            "chat_tool_loop", initial_state=state
        ):
            yield ev, pl

    @workflow.step("sub_agent_execute", name="llm_stream")
    async def _wf_llm_stream(self, iteration: int, state: AgentState):
        """Sub-agent execution step"""
        async for ev, pl in self.agent.execute_loop(
            "goal_driven_loop", initial_state=state
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
        """Plan complex task into micro-tasks using PromptBuilder directly."""
        if state.planned_tasks:  # already planned
            return

        state.phase = self.Phase.PLAN
        yield ("phase_changed", {"phase": "plan"})

        prompt = PromptBuilder.plan_microtasks_prompt(self.agent.state.task)

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
                state.task_dependencies[i] = task.get("depends_on")

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
        """Launch tasks whose dependencies are satisfied."""
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
                    continue

            # Launch this task
            assigned_tools = state.task_tool_assignments.get(idx, [])
            task_desc = state.planned_tasks[idx]
            name = f"Step {idx + 1}/{len(state.planned_tasks)}: {task_desc[:55]}..."

            task_id = await self.agent.start_complex_deployment(
                task_description=task_desc,
                user_facing_name=name,
                assigned_tools=assigned_tools or None,
                temperature=0.25,
                custom_system_prompt=PromptBuilder.sub_agent_system_prompt(
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

    @workflow.step("orchestrator", name="check_orchestrator_complete")
    async def _wf_check_orchestrator_complete(self, iteration: int, state: AgentState):
        """Check if all planned tasks have been launched and completed."""
        if state.current_task_index < len(state.planned_tasks):
            state.is_complete = False
            return

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
        await self.agent.add_message("assistant", summary)

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

    # ==================== HELPERS ====================

    async def _generate_user_friendly_summary(self, state: AgentState) -> str:
        """Generate natural summary after complex task using PromptBuilder directly."""
        tool_results_str = "\n".join(
            f"• {r['name']}: {str(r.get('result', ''))[:400]}"
            for r in self.agent.tool_results[-12:]
        )

        prompt = PromptBuilder.user_friendly_task_summary_prompt(
            task=self.agent.state.task,
            goal=self.agent.state.goal,
            tool_results_str=tool_results_str,
        )

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
        """Simple fallback summary for sub-agents."""
        return "✅ Sub-task completed.\n\nKey results recorded in shared context."
