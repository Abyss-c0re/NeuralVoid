# NeuralVoid Agent Testing Guide (AGENTS.md)

This document is written for AI agents (like Grok, Claude, Cursor, etc.) that will work on or test the NeuralVoid + NeuralCore system.

It explains how to run realistic tests in the exact mode we use for development and verification: **headless deploy + WebSocket user simulation**.

## 1. Core Testing Mode

The primary way to test the agent system (especially shutdown, background work, and the new `BackgroundManager`) is:

```bash
# From ProjectNexus root
uv --project NeuralCore run neuralvoid --deploy --agent agent_001
```

- `--deploy` runs the agent headlessly (no TUI).
- It uses the real config at `~/.neuralcore/config.yaml` and real LLM server.
- The agent exposes a **WebSocket bridge** (usually on port 8765 for the first agent) that you can connect to as a "user".

## 2. Interacting with the Agent via WebSocket

After launching the agent, connect to its bridge:

```python
import asyncio
import json
import websockets

async with websockets.connect("ws://127.0.0.1:8765") as ws:
    # Send a message as the user
    await ws.send(json.dumps({
        "command": "send",
        "content": "List the files in the current directory using tools if needed."
    }))

    # Listen for events
    while True:
        msg = await ws.recv()
        data = json.loads(msg)
        print(data)
```

### Useful WebSocket Commands

| Command              | Payload Example                                      | Purpose                                      |
|----------------------|------------------------------------------------------|----------------------------------------------|
| `send`               | `{"command": "send", "content": "your message"}`     | Simulate user chat / task input              |
| `stop`               | `{"command": "stop"}`                                | Gracefully stop the agent (triggers shutdown)|
| `full_state`         | `{"command": "full_state"}`                          | Get current agent state + BackgroundManager status |
| `status`             | `{"command": "status"}`                              | Lightweight status                           |
| `control`            | `{"command": "control", "payload": {...}}`           | Send low-level control events                |

## 3. Log Monitoring (Critical)

Always monitor the main log file while testing:

```bash
tail -f ~/.neuralcore/neuralvoid.log
```

**Useful filters during testing:**

```bash
# Focus on background work and shutdown
grep -E "(BackgroundManager|shutdown|watcher|goal_driven|Cancelled|ERROR|Exception)" ~/.neuralcore/neuralvoid.log

# See workflow activity
grep -E "(phase_changed|TASK-DRIVEN|goal_driven_loop|chat_tool_loop)" ~/.neuralcore/neuralvoid.log
```

## 4. What to Test in This Mode

### 4.1 Basic Flow (from default_flow.py)

When the agent is launched with `deploy_agent` workflow:

- It starts in `chat_tool_loop` (even in headless mode).
- When you send a message via WS:
  - `classify_intent` decides **CASUAL** vs **TASK**.
  - **TASK** → planning (`TaskExecutor.plan` from NeuralHub) → `goal_driven_loop` → tool use → `goal_achieved` condition.
  - After completion it restarts back into `chat_tool_loop`.

**Good test prompts:**
- Task: `"List files in the current directory using tools. Be concise."`
- Casual: `"Thanks! What's your name?"`
- Multi-turn: Send several messages in sequence and observe loop restarts.

### 4.2 BackgroundManager & Shutdown (P0 Focus)

This is currently the most important area to verify.

**What to look for in logs after sending a prompt and then `stop`:**

- `BackgroundManager:xxx Initialized`
- `submit(...)` calls for jobs (especially `job_type="watcher"` when auto_reindex is enabled)
- On `stop`:
  - `Shutting down background services...`
  - Cancellation of jobs
  - `Shutdown complete`
- Final job count should be 0 after `agent.shutdown()`

**Key files to understand:**
- `NeuralCore/src/neuralcore/core/background.py`
- `NeuralCore/src/neuralcore/agents/core.py` → `shutdown()`
- `NeuralCore/src/neuralcore/cognition/knowledge.py` → how the watcher is submitted
- `NeuralVoid/src/neuralvoid/cli/headless_agent.py` → runner shutdown path
- `NeuralVoid/src/neuralvoid/main.py` → top-level signal/atexit handling
- `NeuralHub/src/neuralhub/tasks/executor.py` → TaskExecutor (relocated agentic task orchestration; was TaskManager)
- `NeuralVoid/src/neuralvoid/workflows/default_flow.py` → uses TaskExecutor via proper imports from neuralhub

### 4.3 Common Things to Verify

1. Does sending a prompt via WS correctly trigger planning and tool use?
2. Does the agent switch between `TASK-DRIVEN` and casual modes properly?
3. On `stop` via WS, does `agent.shutdown()` get called and do all background jobs get cancelled?
4. Are there repeated calls to `start_background_services()`? (This is a known smell — we want it to be idempotent.)
5. Do any errors appear in the log during normal operation or shutdown?
6. When using `--agents` for multi-agent, does the top-level shutdown in `main.py` clean everything up?

## 5. Useful Commands

```bash
# Clean previous test agents
pkill -f "neuralvoid.*deploy" || true

# Launch with limited iterations (good for testing)
uv --project NeuralCore run neuralvoid --deploy --agent agent_001 --max-iterations 15

# Launch and capture full output
uv --project NeuralCore run neuralvoid --deploy --agent agent_001 2>&1 | tee -a ~/.neuralcore/neuralvoid.log
```

## 6. Project Structure (Relevant for Testing)

```
NeuralVoid/
├── src/neuralvoid/
│   ├── main.py                    # Entry point, signal handling, multi-agent
│   ├── cli/headless_agent.py      # Runner + shutdown logic
│   ├── ui/chat.py                 # TUI + shutdown on exit
│   └── workflows/default_flow.py  # The actual agent loops (very important)
├── pyproject.toml

NeuralCore/
├── src/neuralcore/
│   ├── agents/core.py             # Agent + BackgroundManager ownership + shutdown()
│   ├── core/background.py         # The new internal work manager
│   ├── cognition/                 # KnowledgeBase, ContextManager, Consolidator
│   └── ...
```

## 7. Current Known Gotchas (as of May 2026)

- `auto_reindex` is often disabled in `~/.neuralcore/config.yaml` → the KB watcher job may not appear.
- `start_background_services()` is currently called multiple times in some flows → causes duplicate log lines in LLM context.
- There is still a `SyntaxWarning` about `return` in `finally` in `headless_agent.py`.
- The WebSocket bridge port is usually 8765 for `agent_001` (increments for additional agents).

## 8. Recommended Testing Session Flow

1. Clean old processes.
2. Launch agent in background with the tool (or nohup).
3. Start a log monitor (filtered for `BackgroundManager|shutdown|ERROR`).
4. Run a Python WebSocket client that:
   - Connects
   - Sends 1–2 task prompts
   - Sends a casual message
   - Requests `full_state`
   - Sends `stop`
5. Analyze logs for clean shutdown and absence of duplicate services spam.
6. Kill the monitor when done.

---

**Goal of this document**: Give any future AI agent (including yourself in the next session) enough context to immediately start productive testing without having to rediscover the entire architecture from scratch.

Update this file whenever major testing patterns or gotchas change.
