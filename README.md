# NeuralVoid – Demo App for NeuralCore

**NeuralVoid** is a demonstration application built on the **[NeuralCore](https://github.com/Abyss-c0re/NeuralCore)** adaptive agentic framework. It showcases multi-agent orchestration, dynamic workflows, sub-agent deployment, background execution, and interactive chat capabilities. NeuralVoid serves as both a **demo** and an active **test harness** for the latest NeuralCore features.

## Features

* **Interactive Chat Loop** (`deploy_chat` workflow): Persistent terminal-based conversation using Textual UI. Supports natural language, tool calls, and seamless escalation to complex orchestration via `RequestComplexAction`.
* **Headless / Orchestrator Mode** (`orchestrator` workflow): Fully automated planning → micro-task decomposition → parallel sub-agent deployment → completion.
* **Sub-Agent System**: 
  - `start_complex_deployment` spawns isolated `SubAgent` instances.
  - `sub_agent_execute` workflow provides focused ReAct loops.
  - Tools are restricted per sub-agent; results are forwarded to parent with context syncing.
  - Full lifecycle management (status, cancel, purge, background tasks).
* **WorkflowEngine** (core runtime):
  - Loads workflows from `config.yaml`.
  - Supports `register_workflow`, `include`, conditional steps (`if`/`when`), retries, timeouts, `go_to`, `insert_steps`, and runtime control events.
  - Advanced condition evaluator with built-in state introspection (`needs_reflection`, `all_sub_tasks_completed`, `error_rate_high`, sub-task stats, etc.).
  - Step-level overrides (client, temperature, max_tokens, system_prompt, toolset).
  - Unified event stream with `step_start`, `step_completed`, `phase_changed`, `sub_agent_progress`, etc.
* **AgentFlow** handlers: All `_wf_*` steps are supplied by AgentFlow (decorated with `@workflow.set`). Executors remain in WorkflowEngine.
* **Dynamic Tools & File Editing**:
  - `FileEditingTools`: `write_file`, `replace_block`, `open_file_sync`, `open_file_async`.
  - Tools can be loaded/unloaded per step or per sub-agent.
  - `ToolBrowser` for runtime discovery.
* **Background Execution**: `run_background()` runs agents asynchronously with bidirectional control via message queues (`post_message`, `post_control`, `post_system_message`).
* **Status & Context**: Rich LLM-generated status reports via `get_agent_status()`. ContextManager handles pruning, external content, and task-specific contexts.
* **Config-Driven Everything**: Agents, clients, workflows, tool sets, and defaults defined in `config.yaml`.
* **Production-Ready Controls**: Sub-task tracking, cancellation, cleanup, reflection limits, iteration guards, and infinite-loop protection.

## Installation

```bash
uv sync
uv tool install -e .
```

## Quick Start

```bash
# Interactive chat mode
neuralvoid --agent agent_002

# Headless mode
neuralvoid --deploy "Analyze logs and summarize issues" \
           --agent agent_001 \
           --status_file /tmp/agent_status.json \
           --pid_file /tmp/agent.pid
```

## Project Structure

```
.
├── config.yaml                  # Agent, client, workflow, and tool configuration
├── LICENSE
├── neuralcore.log               # Log file
├── pyproject.toml
├── README.md
├── setup.py
├── src/
│   └── neuralvoid/
│       ├── cli/                 # CLI parsers & headless runner
│       ├── tools/               # Toolsets (file, terminal, agentic tools)
│       ├── ui/                  # Chat UI and rendering helpers
│       ├── workflows/           # Predefined agent workflows
│       └── main.py              # Entry point
└── uv.lock                      # UV package lock file
```

## How It Works

1. **Configuration Loading** – Agents, clients, workflows, and tools are defined in `config.yaml`.

2. **Agent Initialization** – `Agent` (or `SubAgent`) is created with `WorkflowEngine`, `DynamicActionManager`, and `ContextManager`.

3. **Workflow Execution** (`WorkflowEngine.run()`):
   - Resolves steps (supports `include` for reusable workflow fragments).
   - Evaluates conditions using rich state introspection (`needs_reflection`, `all_sub_tasks_completed`, `error_rate_high`, sub-task stats, etc.).
   - Applies step-level overrides (client, temperature, max_tokens, system_prompt, toolset).
   - Configures tools dynamically for the current step.
   - Executes the corresponding handler from `AgentFlow`.
   - Handles runtime control events (`go_to`, `insert_steps`, `switch_workflow`, control queue messages, etc.).
   - Includes safeguards against infinite loops and excessive steps per iteration.

4. **Orchestrator Workflow** (`orchestrator`):
   - `plan_microtasks` → LLM breaks the main task into smaller micro-tasks with suggested tools.
   - `launch_next_subtask` → Spawns multiple `SubAgent` instances in parallel via `start_complex_deployment`.
   - `wait_for_subtask` → Monitors completion of launched sub-tasks and emits progress events.
   - `check_orchestrator_complete` → Generates a friendly user summary and switches back to chat mode.

5. **Chat Workflow** (`deploy_chat`):
   - Runs a continuous message queue loop.
   - Processes user messages and tool calls.
   - Escalates complex requests to the orchestrator using `RequestComplexAction`.

6. **Sub-Agent Workflow** (`sub_agent_execute`):
   - Focused ReAct-style loop using the `llm_stream` step.
   - Uses restricted tools and a lightweight system prompt.
   - Automatically forwards tool results and final replies to the parent agent’s context.

7. **Control & Communication**:
   - Message queues enable bidirectional communication (user input, system alerts, control events).
   - Background execution (`run_background`) forwards all workflow events as control messages.
   - Runtime workflow switching and dynamic step insertion are fully supported.

## Configuration

All system behavior is driven by a single `config.yaml` file located in the project root. It defines clients, tools, workflows, and agents.

### Important: Tokenizer Configuration

**The tokenizer must be explicitly specified** for every client in `config.yaml`.  
You can provide either:
- A path to a local JSON tokenizer file, or
- A Hugging Face tokenizer tag (e.g. `Qwen/Qwen2.5-7B-Instruct`)

Example:
```yaml
clients:
  main:
    type: chat
    base_url: http://localhost:1212/v1
    model: qwen3.5-9b
    tokenizer: /home/user/Dev/AI/model/tokenizer.json     # ← Required: local JSON file
    # tokenizer: Qwen/Qwen2.5-7B-Instruct                 # Alternative: HF tag

  reasoning:
    type: chat
    model: qwen3.5-9b
    tokenizer: /home/user/Dev/AI/model/tokenizer.json     # ← Also required here
    temperature: 0.4
    max_tokens: 64000


workflows:
  orchestrator:
    name: "orchestrator"
    description: "Plan → launch sub-agents → wait → complete"
    steps:
      - name: plan_microtasks
        # Step-level overrides
        overrides:
          temperature: 0.3
          max_tokens: 8000
          system_prompt: "You are an expert task decomposition assistant."

      - name: launch_next_subtask
        overrides:
          temperature: 0.25
          client: reasoning          # Use the "reasoning" client for this step

      - name: wait_for_subtask
        # Conditional execution
        if:
          all_sub_tasks_completed: true
        retries: 3
        timeout: 30

      - name: check_orchestrator_complete
        overrides:
          temperature: 0.7
          toolset: "DeployControls"   # Switch toolset for final summary step

  deploy_chat:
    steps:
      - name: deploy_chat_loop
        overrides:
          temperature: 0.85
          max_tokens: 12000
```

## Notes

- NeuralVoid is primarily a **demo and test harness** for the NeuralCore framework.
- The core framework (`neuralcore/`) is designed to be reusable in other agentic applications.
- Detailed logging with iteration snapshots helps with debugging complex multi-agent workflows.
- File editing tools are optimized for safe, LLM-driven code changes.

**Compatibility Note**:  
The software was primarily tested with **llama.cpp** (local inference server).  
However, because it uses the standard **OpenAI Python client** under the hood, it is compatible with many other providers such as OpenAI, Groq, Together.ai, Fireworks, and any OpenAI-compatible endpoint.


