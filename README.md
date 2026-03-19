# NeuralVoid

A powerful AI agent framework built on **NeuralCore**, designed for interactive terminal-based LLM interaction and headless autonomous task execution.

## 🚀 Features

- **Interactive Chat Interface**: Talk directly with LLM agents in your terminal
- **Headless Agent Mode**: Deploy autonomous agents for background processing and automation
- **Extensible Tool System**: Currently supports `terminal` and `file` tool sets, easily expandable
- **Multi-Model Support**: Configurable support for different LLM types (chat, reasoning, embeddings)
- **Context-Aware Memory**: Built-in context management for conversation continuity

## 📦 Installation

```bash
# Using uv (recommended)
uv tool install .

# Or using pip
pip install -e .
```

## ⚙️ Configuration

Configuration is stored in `config.yaml` within the project directory or at `$HOME/.neuralcore/config.yaml`.

### Key Configuration Sections:

- **Clients**: Define your LLM connections (chat, reasoning, embeddings)
- **Tools**: Enable/disable tool sets (`terminal`, `file`)
- **Agents**: Configure headless agent behavior
- **App**: Set application-level parameters (iterations, tokens, logging)

## 🎯 Usage

### Interactive Chat Mode

```bash
# Basic usage
neuralvoid

# With custom config
neuralvoid --config <path/to/config.yaml>
```

### Headless Agent Mode

```bash
# Deploy a headless agent
neuralvoid --deploy "Task description here" \
  --status-file status.json \
  --pid-file pid.json

# With specific iteration limits
neuralvoid --deploy "..." --max-iterations 10 --throttle-sec 5
```

### Command Line Interface

```bash
# Available CLI options:
#   --config, -c       Path to configuration file (optional)
#   --deploy, -d       Deploy headless agent with prompt
#   --status-file, -sf File for status updates
#   --pid-file, -pf    File for process ID tracking
#   --throttle-sec     Throttle interval in seconds
#   --max-iterations   Maximum iterations (overrides config)
```

## 🏗️ Architecture

### Project Structure:

```
neuralvoid/
├── src/neuralvoid/
│   ├── cli/              # Command-line argument parsing
│   ├── tools/            # Terminal and file tool implementations
│   └── ui/               # UI rendering and chat interface
├── config.yaml           # Application configuration
├── pyproject.toml        # Project metadata and dependencies
└── setup.py              # Legacy setup script (for pip compatibility)
```

### Core Components:

- **ActionRegistry**: Manages tool registration and execution
- **DynamicActionManager**: Handles runtime tool dynamics
- **ContextManager**: Maintains conversation context and memory
- **ToolBrowser**: Provides a browser-like interface to available tools
- **ClientFactory**: Abstracts LLM client connections (chat, reasoning, embeddings)

## 🔧 Tool Sets

Currently supported:

| Set | Description |
|-----|-------------|
| `terminal` | Terminal operations (run commands, check output, etc.) |
| `file` | File system operations (read, write, list, search, etc.) |

To enable additional tool sets, modify the `tools.enabled_sets` array in your config.

## 📝 Example Workflows

### 1. Interactive Research Session

```bash
neuralvoid
> "Help me analyze this codebase and summarize the project structure"
> [Agent will use file tools to explore and provide insights]
```

### 2. Headless Automation Task

```bash
neuralvoid --deploy "Monitor logs for errors and create a daily summary report" \
  --status-file ./monitor/status.json \
  --pid-file ./monitor/pid.json
```

## 🔄 Extensibility

NeuralVoid is designed to be extended:

### Adding New Tool Sets:

1. Create your tool module in `src/neuralvoid/tools/`
2. Implement the required interface (registry, manager support)
3. Register it in your config under `tools.enabled_sets`

### Adding New LLM Clients:

Modify the `clients` section in your `config.yaml` to add additional models or increase capacity.


---

**Built on [NeuralCore](https://github.com/Abyss-c0re/NeuralCore)**
