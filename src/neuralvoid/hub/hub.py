"""
AgentHub — Central WebSocket hub for multi-agent communication.

Manages multiple NeuralCore Agent instances deployed in the same process,
provides a WebSocket server for inter-agent message routing, and exposes
monitoring endpoints so external clients can observe / control the swarm.

Architecture
~~~~~~~~~~~~
  External WS client ──▶ AgentHub (ws://host:hub_port)
                              │
                ┌─────────────┼─────────────────┐
                ▼             ▼                  ▼
           Agent Alpha    Agent Beta        Agent Gamma
           (bridge:8771)  (bridge:8772)     (bridge:8773)

Each agent also gets its own WebSocketBridge for direct 1:1 access.
The Hub adds a routing layer on top so that any connected client can
relay messages between agents, broadcast, or query statuses.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from websockets.asyncio.server import ServerConnection, serve

from neuralcore.agents.core import Agent
from neuralcore.bridge.websocket import WebSocketBridge
from neuralcore.utils.logger import Logger

logger = Logger.get_logger()


class AgentHub:
    """
    Central WebSocket hub for multi-agent coordination.

    Responsibilities:
      - Hold references to every deployed Agent instance.
      - Start per-agent WebSocketBridge instances on sequential ports.
      - Run a *hub-level* WebSocket server that supports relay / broadcast /
        status queries so agents (or external tools) can talk to each other.
      - Keep an append-only message log for post-mortem inspection.
    """

    # ────────────────────────────────────────────────────────────
    # Construction
    # ────────────────────────────────────────────────────────────
    def __init__(
        self,
        host: str = "127.0.0.1",
        hub_port: int = 8770,
        bridge_base_port: int = 8771,
    ):
        self.host = host
        self.hub_port = hub_port
        self._bridge_next_port = bridge_base_port

        # agent_id → Agent instance
        self.agents: Dict[str, Agent] = {}
        # agent_id → WebSocketBridge (one per agent for direct access)
        self.bridges: Dict[str, WebSocketBridge] = {}
        # asyncio tasks that keep bridges alive
        self._bridge_tasks: Dict[str, asyncio.Task] = {}

        # Hub-level server
        self._server = None
        self._hub_clients: set[ServerConnection] = set()

        # Append-only log of every inter-agent message routed through the hub
        self.message_log: List[Dict[str, Any]] = []

    # ────────────────────────────────────────────────────────────
    # Agent registration
    # ────────────────────────────────────────────────────────────
    def register_agent(self, agent: Agent, ws_port: Optional[int] = None) -> int:
        """Register an agent with the hub.

        Creates a dedicated WebSocketBridge for the agent and returns the
        port it will listen on.  The bridge is NOT started yet — call
        ``start()`` to bring everything online.
        """
        port = ws_port or self._bridge_next_port
        self._bridge_next_port = max(self._bridge_next_port, port) + 1

        self.agents[agent.agent_id] = agent
        # Attach a back-reference so the agent (and the workflow engine's
        # wait-for-agent logic) can reach the hub at runtime.
        agent.hub = self  # type: ignore[attr-defined]

        bridge = WebSocketBridge(agent=agent, host=self.host, port=port)
        self.bridges[agent.agent_id] = bridge

        logger.info(
            f"[HUB] Registered agent '{agent.name}' ({agent.agent_id}) "
            f"→ bridge ws://{self.host}:{port}"
        )
        return port

    # ────────────────────────────────────────────────────────────
    # Inter-agent messaging (the core routing primitive)
    # ────────────────────────────────────────────────────────────
    async def send_to_agent(
        self,
        target_id: str,
        content: str,
        source_id: Optional[str] = None,
    ) -> bool:
        """Route a message to *target_id*'s message queue.

        Uses ``Agent.post_message`` so the message lands in the queue AND
        the context manager — exactly the same path as a WebSocket ``send``
        command.  Returns True on success.
        """
        target = self.agents.get(target_id)
        if target is None:
            logger.warning(f"[HUB] send_to_agent: unknown target '{target_id}'")
            return False

        # Tag the message with the source agent so the receiver can tell
        # who sent it.
        tagged_content = content
        if source_id:
            tagged_content = f"[from {source_id}] {content}"

        await target.post_message(tagged_content)

        entry = {
            "source": source_id,
            "target": target_id,
            "content": content,
            "timestamp": time.time(),
        }
        self.message_log.append(entry)

        logger.info(
            f"[HUB] Relayed message {source_id or '?'} → {target_id} "
            f"({len(content)} chars)"
        )

        # Also broadcast a notification to all hub-level WS clients
        await self._broadcast_to_hub_clients(
            {"type": "relay", "source": source_id, "target": target_id,
             "content": content[:200], "timestamp": entry["timestamp"]}
        )
        return True

    async def broadcast_to_all(
        self,
        content: str,
        source_id: Optional[str] = None,
    ) -> int:
        """Broadcast a message to every registered agent (except *source_id*)."""
        count = 0
        for aid in list(self.agents):
            if aid != source_id:
                ok = await self.send_to_agent(aid, content, source_id=source_id)
                if ok:
                    count += 1
        return count

    # ────────────────────────────────────────────────────────────
    # Status helpers
    # ────────────────────────────────────────────────────────────
    def get_all_statuses(self) -> Dict[str, Dict[str, Any]]:
        """Lightweight status snapshot of every registered agent."""
        return {
            aid: agent.get_detailed_status()
            for aid, agent in self.agents.items()
        }

    def get_agent(self, agent_id: str) -> Optional[Agent]:
        return self.agents.get(agent_id)

    # ────────────────────────────────────────────────────────────
    # Hub-level WebSocket server
    # ────────────────────────────────────────────────────────────
    async def _hub_handler(self, websocket: ServerConnection):
        """Handle connections to the hub-level WebSocket server.

        Supported commands:
          relay        — route a message to a specific agent
          broadcast    — send to all agents
          status       — return status of all agents
          list_agents  — list registered agent IDs / names
          message_log  — return the inter-agent message log
        """
        self._hub_clients.add(websocket)
        logger.info("[HUB] External client connected to hub WS")

        try:
            async for raw_msg in websocket:
                try:
                    msg = json.loads(raw_msg)
                except json.JSONDecodeError:
                    await websocket.send(
                        json.dumps({"type": "error", "message": "Invalid JSON"})
                    )
                    continue

                cmd = msg.get("command", "")

                if cmd == "relay":
                    # Route a message from one agent to another
                    target = msg.get("target", "")
                    content = msg.get("content", "")
                    source = msg.get("source")
                    ok = await self.send_to_agent(target, content, source_id=source)
                    await websocket.send(json.dumps({
                        "type": "ack", "action": "relay",
                        "success": ok, "target": target,
                    }))

                elif cmd == "broadcast":
                    content = msg.get("content", "")
                    source = msg.get("source")
                    n = await self.broadcast_to_all(content, source_id=source)
                    await websocket.send(json.dumps({
                        "type": "ack", "action": "broadcast", "delivered": n,
                    }))

                elif cmd == "status":
                    await websocket.send(json.dumps({
                        "type": "status",
                        "agents": {
                            aid: s for aid, s in self.get_all_statuses().items()
                        },
                    }))

                elif cmd == "list_agents":
                    agents_list = [
                        {"id": a.agent_id, "name": a.name, "status": a.status}
                        for a in self.agents.values()
                    ]
                    await websocket.send(json.dumps({
                        "type": "agents", "data": agents_list,
                    }))

                elif cmd == "message_log":
                    limit = msg.get("limit", 50)
                    await websocket.send(json.dumps({
                        "type": "message_log",
                        "data": self.message_log[-limit:],
                    }))

                else:
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": f"Unknown hub command: {cmd}",
                    }))

        except Exception as exc:
            logger.debug(f"[HUB] Client error: {exc}")
        finally:
            self._hub_clients.discard(websocket)
            logger.info("[HUB] External client disconnected from hub WS")

    async def _broadcast_to_hub_clients(self, message: dict):
        """Push a notification to every connected hub-level WS client."""
        dead: set[ServerConnection] = set()
        raw = json.dumps(message)
        for ws in list(self._hub_clients):
            try:
                await ws.send(raw)
            except Exception:
                dead.add(ws)
        self._hub_clients -= dead

    # ────────────────────────────────────────────────────────────
    # Lifecycle
    # ────────────────────────────────────────────────────────────
    async def start(self):
        """Start the hub WebSocket server and all per-agent bridges."""
        # 1. Start each agent's dedicated WebSocket bridge
        for aid, bridge in self.bridges.items():
            task = asyncio.create_task(bridge.start(), name=f"bridge_{aid}")
            self._bridge_tasks[aid] = task
            logger.info(
                f"[HUB] Bridge for '{aid}' starting on "
                f"ws://{bridge.host}:{bridge.port}"
            )

        # 2. Start the central hub server
        self._server = await serve(self._hub_handler, self.host, self.hub_port)
        logger.info(
            f"[HUB] Hub WebSocket server live → ws://{self.host}:{self.hub_port}"
        )

    async def stop(self):
        """Gracefully tear down the hub and all bridges."""
        # Stop per-agent bridges
        for aid, bridge in self.bridges.items():
            try:
                await bridge.stop()
            except Exception:
                pass
            task = self._bridge_tasks.get(aid)
            if task and not task.done():
                task.cancel()

        # Stop hub server
        if self._server:
            self._server.close()
            await self._server.wait_closed()

        logger.info("[HUB] AgentHub stopped")

    # ────────────────────────────────────────────────────────────
    # High-level deploy helper (used by main.py)
    # ────────────────────────────────────────────────────────────
    async def deploy_all(
        self,
        prompt: Optional[str] = None,
        system_prompt: str = "",
        max_tokens: int = 12000,
    ) -> Dict[str, bool]:
        """Deploy every registered agent in parallel and run until they finish.

        Each agent runs through HeadlessAgentRunner.  The hub server and
        all bridges are started first so agents can communicate immediately.

        Returns a dict of agent_id → success bool.
        """
        from neuralvoid.cli.headless_agent import HeadlessAgentRunner

        await self.start()

        results: Dict[str, bool] = {}
        tasks: Dict[str, asyncio.Task] = {}

        # Create per-agent runner tasks
        for aid, agent in self.agents.items():
            bridge = self.bridges.get(aid)
            bridge_port = bridge.port if bridge else 8765

            # skip_bridge=True because the hub already started bridges above
            runner = HeadlessAgentRunner(
                agent=agent,
                status_file=f"/tmp/neuralvoid/{aid}.status.json",
                pid_file=f"/tmp/neuralvoid/{aid}.pid",
                websocket_port=bridge_port,
                skip_bridge=True,
            )

            # Each agent gets the same initial prompt (or None for listen mode)
            async def _run_agent(r=runner, p=prompt):
                return await r.run(
                    prompt=p,
                    system_prompt=system_prompt,
                    max_tokens=max_tokens,
                )

            tasks[aid] = asyncio.create_task(_run_agent(), name=f"agent_{aid}")
            logger.info(f"[HUB] Deployed agent '{agent.name}' ({aid})")

        print("\n" + "=" * 60)
        print(f"  AgentHub live — {len(self.agents)} agents deployed")
        print(f"  Hub WS: ws://{self.host}:{self.hub_port}")
        for aid, bridge in self.bridges.items():
            name = self.agents[aid].name
            print(f"  {name} ({aid}): ws://{bridge.host}:{bridge.port}")
        print("=" * 60 + "\n")

        # Wait for all agents to finish (or be cancelled)
        done, _ = await asyncio.wait(tasks.values(), return_when=asyncio.ALL_COMPLETED)

        for aid, task in tasks.items():
            try:
                results[aid] = task.result()
            except Exception as exc:
                logger.error(f"[HUB] Agent '{aid}' failed: {exc}")
                results[aid] = False

        await self.stop()
        return results
