#!/bin/bash
# NeuralVoid - Installation script
# Supports standard install and --dev for editable NeuralCore development
# Client-side only, modular, reusable - no framework domain logic

set -euo pipefail

echo "=== NeuralVoid Installation ==="

# Parse dev argument
DEV_MODE=false
if [[ "${1:-}" == "--dev" ]] || [[ "${1:-}" == "-d" ]] || [[ "${1:-}" == "dev" ]]; then
  DEV_MODE=true
fi

# Script directory (NeuralVoid root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ "$DEV_MODE" = true ]; then
  echo "🚀 Dev mode activated - linking editable NeuralCore..."

  # Parent directory = NeuralCore root when inside client/ submodule
  PARENT_DIR="$(dirname "$SCRIPT_DIR")"

  # 1. Original: parent itself is NeuralCore (e.g. NeuralVoid inside NeuralCore/client layout)
  if [ -f "$PARENT_DIR/pyproject.toml" ] && [ -d "$PARENT_DIR/src/neuralcore" ]; then
    NEURALCORE_PATH="$PARENT_DIR"
    echo "✅ Detected NeuralCore project at parent (client submodule)"

  # 2. NEW: sibling NeuralCore folder (common dev layout: ProjectNexus/NeuralCore + ProjectNexus/NeuralVoid)
  elif [ -d "$PARENT_DIR/NeuralCore" ] && \
       [ -f "$PARENT_DIR/NeuralCore/pyproject.toml" ] && \
       [ -d "$PARENT_DIR/NeuralCore/src/neuralcore" ]; then
    NEURALCORE_PATH="$PARENT_DIR/NeuralCore"
    echo "✅ Detected sibling NeuralCore project at $NEURALCORE_PATH"

  else
    echo "ℹ️  Not inside NeuralCore client. Cloning using pyproject.toml link..."
    # Extract exact git URL from our own pyproject.toml
    PYPROJECT="$SCRIPT_DIR/pyproject.toml"
    GIT_URL=$(grep -oP 'neuralcore @ \Kgit\+https?://[^ "]+' "$PYPROJECT" 2>/dev/null || echo "https://github.com/Abyss-c0re/NeuralCore.git")
    
    CLONE_DIR="$PARENT_DIR/NeuralCore"
    if [ ! -d "$CLONE_DIR" ]; then
      git clone "$GIT_URL" "$CLONE_DIR"
      echo "✅ Cloned NeuralCore to $CLONE_DIR"
    else
      echo "✅ Using existing clone at $CLONE_DIR"
    fi
    NEURALCORE_PATH="$CLONE_DIR"
  fi

  # Remove and re-add as editable (exactly as requested)
  echo "🔄 Removing existing neuralcore dependency..."
  uv remove neuralcore 2>/dev/null || true

  echo "🔗 Adding NeuralCore as editable dependency from $NEURALCORE_PATH..."
  uv add --editable "$NEURALCORE_PATH"
fi

# Standard Void installation (exactly as in README)
echo "📦 Syncing dependencies..."
uv sync

echo "🛠 Installing NeuralVoid as editable tool..."
uv tool install -e .

echo ""
echo "✅ NeuralVoid installation completed successfully!"
if [ "$DEV_MODE" = true ]; then
  echo "🔧 Dev mode active — edits to NeuralCore will be live immediately."
fi

echo ""
echo "Next steps:"
echo "  neuralvoid --help"
echo "  (or uv run your-entrypoint)"