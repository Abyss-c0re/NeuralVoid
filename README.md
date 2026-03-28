# NeuralVoid – Demo App for NeuralCore

**NeuralVoid** is a demonstration application built on the **[NeuralCore](https://github.com/Abyss-c0re/NeuralCore)** adaptive agentic framework. It showcases multi-agent orchestration, dynamic workflows, and interactive chat capabilities. This project is designed as both a **demo** and a **test harness** for NeuralCore features.

## Features

* **Interactive Chat Loop**: Terminal-based **Textual UI** for natural agent conversations.
* **Headless Mode**: Deploy agents automatically with a prompt and optional status tracking.
* **Highly Adjustable Workflows**: Add `.py` workflow files in `src/neuralvoid/workflows/` and customize each step:

  * Client / LLM instance → via `overrides["client"]`
  * Temperature → `overrides["temperature"]`
  * Max tokens → `overrides["max_tokens"]`
  * System prompt → `overrides["system_prompt"]`
  * Toolsets / tools → via `overrides["toolset"]` or `agent.manager.configure_for_step()`
  * Step retries → `retries` in step config
  * Timeout per step → `timeout` in step config
  * Conditional execution → `if` / `when` with built-in or custom conditions
  * Insert / jump steps → `insert_steps` events, `go_to` events
  * Sub-agent behavior → light tool adjustment for special steps (like `llm_stream`)
* **Easy Tool Creation**: Add Python tool files in `src/neuralvoid/tools/` to dynamically expose new agent capabilities.
* **Dynamic Tools**: Load tools from `src/neuralvoid/tools/` (internal or external).
* **Multi-Agent Deployment**: Spawn sub-agents, orchestrate complex tasks, and override agent configuration dynamically.
* **Configurable Clients & Agents**: Different LLM clients and agents can be defined for reasoning, chat, or specialized tasks.

Everything else (iteration limits, workflow switching, logging, etc.) is global or controlled by runtime events.

## Installation

Install **NeuralVoid** in editable mode using **uv** (Ultralight Package Manager):

```bash
uv sync
uv tool install -e .
```

After installation, you can call `neuralvoid` from anywhere in your terminal.

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

## How NeuralVoid Works

1. **Load Configuration**: Agents, clients, workflows, and tools are defined in `config.yaml`.
2. **Initialize Agent**: The selected agent is loaded with its workflow and assigned tools.
3. **Run Mode**:

   * **Interactive**: Terminal chat UI for natural conversation.
   * **Headless**: Agents run automatically on a given prompt.
4. **Workflow & Tools**: Agents execute tasks through **highly adjustable workflows**; steps can override clients, temperature, tokens, system prompts, toolsets, retries, timeouts, and sub-agent behaviors.
5. **Easy Tool Creation**: New tools can be added as Python files in the `tools` folder and immediately used by agents.
6. **Multi-Agent Orchestration**: Complex tasks are split into micro-tasks, assigned to sub-agents, and executed sequentially or in parallel.

## Notes

* NeuralVoid is a **demo and test harness** for NeuralCore.
* Internal logging tracks tool execution, multi-agent coordination, and workflow progress.
* Defaults for iterations, tokens, and temperature are configurable per agent in `config.yaml`.
