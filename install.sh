#!/bin/bash
# NeuralVoid - Installation script
# Supports standard install, --dev (parent siblings) and --bundle (self-contained ./core + ./hub subfolders)
# --bundle avoids polluting parent dir with Core/Hub clones; everything lives inside NeuralVoid/ (the client) for portable editable dev
#   - core/  = NeuralCore (editable)
#   - hub/   = NeuralHub (editable)
# Client-side only, modular, reusable - no framework domain logic

set -euo pipefail

echo "=== NeuralVoid Installation ==="

# Parse arguments: support --dev (parent siblings) and --bundle (self-contained core/hub subfolders inside NeuralVoid)
DEV_MODE=false
BUNDLE_MODE=false
for arg in "$@"; do
  case "$arg" in
    --dev|-d|dev) DEV_MODE=true ;;
    --bundle|-b|bundle) BUNDLE_MODE=true ;;
  esac
done

# Script directory (NeuralVoid root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ "$DEV_MODE" = true ] || [ "$BUNDLE_MODE" = true ]; then
  if [ "$BUNDLE_MODE" = true ]; then
    echo "📦 Bundle mode activated — self-contained editable NeuralCore (./core) + NeuralHub (./hub)..."
  else
    echo "🚀 Dev mode activated — linking editable NeuralCore + NeuralHub from parent..."
  fi

  PARENT_DIR="$(dirname "$SCRIPT_DIR")"
  PYPROJECT="$SCRIPT_DIR/pyproject.toml"

  # Determine target base and folder names based on mode
  if [ "$BUNDLE_MODE" = true ]; then
    TARGET_BASE="$SCRIPT_DIR"
    CORE_DIR_NAME="core"
    HUB_DIR_NAME="hub"
  else
    TARGET_BASE="$PARENT_DIR"
    CORE_DIR_NAME="NeuralCore"
    HUB_DIR_NAME="NeuralHub"
  fi

  # ──────────────────────────────────────────────────────────────
  # IMPORTANT: Remove both git versions FIRST to avoid resolution hell
  # ──────────────────────────────────────────────────────────────
  echo "🔄 Removing git versions of neuralhub + neuralcore (prevents git resolution conflicts)..."
  uv remove neuralhub neuralcore 2>/dev/null || true

  # ──────────────────────────────────────────────────────────────
  # NeuralHub first (because it depends on neuralcore)
  # ──────────────────────────────────────────────────────────────
  HUB_CANDIDATE="$TARGET_BASE/$HUB_DIR_NAME"
  if [ -d "$HUB_CANDIDATE" ] && \
     [ -f "$HUB_CANDIDATE/pyproject.toml" ] && \
     [ -d "$HUB_CANDIDATE/src/neuralhub" ]; then
    NEURALHUB_PATH="$HUB_CANDIDATE"
    echo "✅ Detected ${BUNDLE_MODE:+bundled }NeuralHub at $NEURALHUB_PATH"
  else
    echo "ℹ️  No local NeuralHub found — cloning using pyproject.toml link..."
    HUB_GIT_URL=$(grep -oP 'neuralhub @ \Kgit\+https?://[^ "]+' "$PYPROJECT" 2>/dev/null | sed 's|^git+||' || echo "https://github.com/Abyss-c0re/NeuralHub.git")
    CLONE_DIR="$HUB_CANDIDATE"
    if [ ! -d "$CLONE_DIR" ]; then
      git clone "$HUB_GIT_URL" "$CLONE_DIR"
      echo "✅ Cloned NeuralHub to $CLONE_DIR"
    else
      echo "✅ Using existing NeuralHub clone at $CLONE_DIR"
    fi
    NEURALHUB_PATH="$CLONE_DIR"
  fi

  # For bundle mode, patch the cloned hub's pyproject so its neuralcore source points to ../core instead of ../NeuralCore
  if [ "$BUNDLE_MODE" = true ]; then
    HUB_PYPROJECT="$NEURALHUB_PATH/pyproject.toml"
    if [ -f "$HUB_PYPROJECT" ]; then
      echo "🔧 Patching NeuralHub pyproject.toml for bundle (neuralcore path -> ../core)..."
      sed -i 's|path = "../NeuralCore"|path = "../core"|' "$HUB_PYPROJECT" 2>/dev/null || true
      sed -i 's|extraPaths = \["../NeuralCore"\]|extraPaths = ["../core"]|' "$HUB_PYPROJECT" 2>/dev/null || true
    fi
  fi

  echo "🔗 Adding local NeuralHub (this brings its own local neuralcore source)..."
  uv add --editable "$NEURALHUB_PATH"

  # ──────────────────────────────────────────────────────────────
  # NeuralCore second
  # ──────────────────────────────────────────────────────────────
  CORE_CANDIDATE="$TARGET_BASE/$CORE_DIR_NAME"
  if [ -d "$CORE_CANDIDATE" ] && \
     [ -f "$CORE_CANDIDATE/pyproject.toml" ] && \
     [ -d "$CORE_CANDIDATE/src/neuralcore" ]; then
    NEURALCORE_PATH="$CORE_CANDIDATE"
    echo "✅ Detected ${BUNDLE_MODE:+bundled }NeuralCore at $NEURALCORE_PATH"
  else
    echo "ℹ️  No local NeuralCore found — cloning using pyproject.toml link..."
    CORE_GIT_URL=$(grep -oP 'neuralcore @ \Kgit\+https?://[^ "]+' "$PYPROJECT" 2>/dev/null | sed 's|^git+||' || echo "https://github.com/Abyss-c0re/NeuralCore.git")
    CLONE_DIR="$CORE_CANDIDATE"
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
if [ "$BUNDLE_MODE" = true ]; then
  echo "📦 Bundle mode active — NeuralCore in ./core/ and NeuralHub in ./hub/ (editable, self-contained)."
  echo "   Edits inside core/ and hub/ will be live immediately. No parent-dir traces."
elif [ "$DEV_MODE" = true ]; then
  echo "🔧 Dev mode active — edits to NeuralCore and NeuralHub will be live immediately."
fi

echo ""
echo "Next steps:"
echo "  neuralvoid --help"
echo "  (or uv run your-entrypoint)"