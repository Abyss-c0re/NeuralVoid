#!/bin/bash
# NeuralVoid - Installation script
# Supports standard install and --dev for editable NeuralCore + NeuralHub development
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
  echo "🚀 Dev mode activated — linking editable NeuralCore + NeuralHub..."

  PARENT_DIR="$(dirname "$SCRIPT_DIR")"
  PYPROJECT="$SCRIPT_DIR/pyproject.toml"

  # ──────────────────────────────────────────────────────────────
  # IMPORTANT: Remove both git versions FIRST to avoid resolution hell
  # ──────────────────────────────────────────────────────────────
  echo "🔄 Removing git versions of neuralhub + neuralcore (prevents git resolution conflicts)..."
  uv remove neuralhub neuralcore 2>/dev/null || true

  # ──────────────────────────────────────────────────────────────
  # NeuralHub first (because it depends on neuralcore)
  # ──────────────────────────────────────────────────────────────
  if [ -d "$PARENT_DIR/NeuralHub" ] && \
     [ -f "$PARENT_DIR/NeuralHub/pyproject.toml" ] && \
     [ -d "$PARENT_DIR/NeuralHub/src/neuralhub" ]; then
    NEURALHUB_PATH="$PARENT_DIR/NeuralHub"
    echo "✅ Detected sibling NeuralHub at $NEURALHUB_PATH"
  else
    echo "ℹ️  No local NeuralHub found — cloning using pyproject.toml link..."
    HUB_GIT_URL=$(grep -oP 'neuralhub @ \Kgit\+https?://[^ "]+' "$PYPROJECT" 2>/dev/null | sed 's|^git+||' || echo "https://github.com/Abyss-c0re/NeuralHub.git")
    CLONE_DIR="$PARENT_DIR/NeuralHub"
    if [ ! -d "$CLONE_DIR" ]; then
      git clone "$HUB_GIT_URL" "$CLONE_DIR"
      echo "✅ Cloned NeuralHub to $CLONE_DIR"
    else
      echo "✅ Using existing NeuralHub clone at $CLONE_DIR"
    fi
    NEURALHUB_PATH="$CLONE_DIR"
  fi

  echo "🔗 Adding local NeuralHub (this brings its own local neuralcore source)..."
  uv add --editable "$NEURALHUB_PATH"

  # ──────────────────────────────────────────────────────────────
  # NeuralCore second
  # ──────────────────────────────────────────────────────────────
  if [ -d "$PARENT_DIR/NeuralCore" ] && \
     [ -f "$PARENT_DIR/NeuralCore/pyproject.toml" ] && \
     [ -d "$PARENT_DIR/NeuralCore/src/neuralcore" ]; then
    NEURALCORE_PATH="$PARENT_DIR/NeuralCore"
    echo "✅ Detected sibling NeuralCore at $NEURALCORE_PATH"
  else
    echo "ℹ️  No local NeuralCore found — cloning using pyproject.toml link..."
    CORE_GIT_URL=$(grep -oP 'neuralcore @ \Kgit\+https?://[^ "]+' "$PYPROJECT" 2>/dev/null | sed 's|^git+||' || echo "https://github.com/Abyss-c0re/NeuralCore.git")
    CLONE_DIR="$PARENT_DIR/NeuralCore"
    if [ ! -d "$CLONE_DIR" ]; then
      git clone "$CORE_GIT_URL" "$CLONE_DIR"
      echo "✅ Cloned NeuralCore to $CLONE_DIR"
    else
      echo "✅ Using existing NeuralCore clone at $CLONE_DIR"
    fi
    NEURALCORE_PATH="$CLONE_DIR"
  fi

  echo "🔗 Adding local NeuralCore..."
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
  echo "🔧 Dev mode active — edits to NeuralCore and NeuralHub will be live immediately."
fi

echo ""
echo "Next steps:"
echo "  neuralvoid --help"
echo "  (or uv run your-entrypoint)"