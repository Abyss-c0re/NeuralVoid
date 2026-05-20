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
    echo "📦 Bundle mode activated — will use ./core + ./hub (or parent siblings if present)..."
  else
    echo "🚀 Dev mode activated — linking editable NeuralCore + NeuralHub from parent..."
  fi

  PARENT_DIR="$(dirname "$SCRIPT_DIR")"
  PYPROJECT="$SCRIPT_DIR/pyproject.toml"

  # ──────────────────────────────────────────────────────────────
  # IMPORTANT: Remove both git versions FIRST to avoid resolution hell
  # ──────────────────────────────────────────────────────────────
  echo "🔄 Removing git versions of neuralhub + neuralcore (prevents git resolution conflicts)..."
  uv remove neuralhub neuralcore 2>/dev/null || true

  # ──────────────────────────────────────────────────────────────
  # Resolve sources for NeuralHub + NeuralCore
  #
  # Rules:
  #   • --bundle still looks in the *parent* first (the monorepo siblings).
  #     Only when nothing is found "there first" do we download into
  #     ./core and ./hub **inside the directory that contains install.sh**.
  #   • Both directories are fully resolved (and cloned if needed) *before*
  #     any `uv add`, so that NeuralHub's [tool.uv.sources] reference to
  #     the core location is valid on disk when uv resolves it.
  #   • We only patch hub's pyproject when we are actually using a local
  #     ./hub copy (the one that needs "../core" instead of "../NeuralCore").
  # ──────────────────────────────────────────────────────────────

  DO_PATCH_HUB=false

  # ----- NeuralHub -----
  if [ "$BUNDLE_MODE" = true ]; then
    # Prefer real sibling in parent even when --bundle was requested
    if [ -d "$PARENT_DIR/NeuralHub" ] && \
       [ -f "$PARENT_DIR/NeuralHub/pyproject.toml" ] && \
       [ -d "$PARENT_DIR/NeuralHub/src/neuralhub" ]; then
      NEURALHUB_PATH="$PARENT_DIR/NeuralHub"
      echo "✅ Detected sibling NeuralHub in parent (preferred over local bundle copy)"
      DO_PATCH_HUB=false
    else
      HUB_CANDIDATE="$SCRIPT_DIR/hub"
      if [ -d "$HUB_CANDIDATE" ] && \
         [ -f "$HUB_CANDIDATE/pyproject.toml" ] && \
         [ -d "$HUB_CANDIDATE/src/neuralhub" ]; then
        NEURALHUB_PATH="$HUB_CANDIDATE"
        echo "✅ Detected local ./hub NeuralHub at $NEURALHUB_PATH"
      else
        echo "ℹ️  No NeuralHub in parent — cloning into ./hub (next to install.sh)..."
        HUB_GIT_URL=$(grep -oP 'neuralhub @ \Kgit\+https?://[^ "]+' "$PYPROJECT" 2>/dev/null | sed 's|^git+||' || echo "https://github.com/Abyss-c0re/NeuralHub.git")
        if [ ! -d "$HUB_CANDIDATE" ]; then
          git clone "$HUB_GIT_URL" "$HUB_CANDIDATE"
          echo "✅ Cloned NeuralHub to $HUB_CANDIDATE"
        else
          echo "✅ Using existing clone at $HUB_CANDIDATE"
        fi
        NEURALHUB_PATH="$HUB_CANDIDATE"
      fi
      DO_PATCH_HUB=true
    fi
  else
    # Classic --dev behavior (parent only)
    HUB_CANDIDATE="$PARENT_DIR/NeuralHub"
    if [ -d "$HUB_CANDIDATE" ] && \
       [ -f "$HUB_CANDIDATE/pyproject.toml" ] && \
       [ -d "$HUB_CANDIDATE/src/neuralhub" ]; then
      NEURALHUB_PATH="$HUB_CANDIDATE"
      echo "✅ Detected sibling NeuralHub at $NEURALHUB_PATH"
    else
      echo "ℹ️  No local NeuralHub found — cloning using pyproject.toml link..."
      HUB_GIT_URL=$(grep -oP 'neuralhub @ \Kgit\+https?://[^ "]+' "$PYPROJECT" 2>/dev/null | sed 's|^git+||' || echo "https://github.com/Abyss-c0re/NeuralHub.git")
      if [ ! -d "$HUB_CANDIDATE" ]; then
        git clone "$HUB_GIT_URL" "$HUB_CANDIDATE"
        echo "✅ Cloned NeuralHub to $HUB_CANDIDATE"
      else
        echo "✅ Using existing NeuralHub clone at $HUB_CANDIDATE"
      fi
      NEURALHUB_PATH="$HUB_CANDIDATE"
    fi
    DO_PATCH_HUB=false
  fi

  # Patch only the internal bundle copy.
  # We change its dependency declaration to use the workspace form so that
  # uv is happy when the root already registered `core` as a workspace member.
  if [ "$DO_PATCH_HUB" = true ]; then
    HUB_PYPROJECT="$NEURALHUB_PATH/pyproject.toml"
    if [ -f "$HUB_PYPROJECT" ]; then
      echo "🔧 Patching NeuralHub pyproject.toml for bundle (using workspace = true for neuralcore)..."
      # Force the workspace form right after clone/detect (uv is strict about this once the root declares a workspace)
      sed -i '/^\s*neuralcore\s*=/c\neuralcore = { workspace = true }' "$HUB_PYPROJECT" 2>/dev/null || true
      # Keep pyright happy (it still benefits from a concrete path)
      sed -i 's|extraPaths = \["../NeuralCore"\]|extraPaths = ["../core"]|' "$HUB_PYPROJECT" 2>/dev/null || true
      sed -i 's|extraPaths = \["../core"\]|extraPaths = ["../core"]|' "$HUB_PYPROJECT" 2>/dev/null || true
    fi
  fi

  # ----- NeuralCore (same parent-first preference when in bundle mode) -----
  if [ "$BUNDLE_MODE" = true ]; then
    if [ -d "$PARENT_DIR/NeuralCore" ] && \
       [ -f "$PARENT_DIR/NeuralCore/pyproject.toml" ] && \
       [ -d "$PARENT_DIR/NeuralCore/src/neuralcore" ]; then
      NEURALCORE_PATH="$PARENT_DIR/NeuralCore"
      echo "✅ Detected sibling NeuralCore in parent (preferred)"
    else
      CORE_CANDIDATE="$SCRIPT_DIR/core"
      if [ -d "$CORE_CANDIDATE" ] && \
         [ -f "$CORE_CANDIDATE/pyproject.toml" ] && \
         [ -d "$CORE_CANDIDATE/src/neuralcore" ]; then
        NEURALCORE_PATH="$CORE_CANDIDATE"
        echo "✅ Detected local ./core NeuralCore at $NEURALCORE_PATH"
      else
        echo "ℹ️  No NeuralCore in parent — cloning into ./core (next to install.sh)..."
        CORE_GIT_URL=$(grep -oP 'neuralcore @ \Kgit\+https?://[^ "]+' "$PYPROJECT" 2>/dev/null | sed 's|^git+||' || echo "https://github.com/Abyss-c0re/NeuralCore.git")
        if [ ! -d "$CORE_CANDIDATE" ]; then
          git clone "$CORE_GIT_URL" "$CORE_CANDIDATE"
          echo "✅ Cloned NeuralCore to $CORE_CANDIDATE"
        else
          echo "✅ Using existing clone at $CORE_CANDIDATE"
        fi
        NEURALCORE_PATH="$CORE_CANDIDATE"
      fi
    fi
  else
    # Classic --dev
    CORE_CANDIDATE="$PARENT_DIR/NeuralCore"
    if [ -d "$CORE_CANDIDATE" ] && \
       [ -f "$CORE_CANDIDATE/pyproject.toml" ] && \
       [ -d "$CORE_CANDIDATE/src/neuralcore" ]; then
      NEURALCORE_PATH="$CORE_CANDIDATE"
      echo "✅ Detected sibling NeuralCore at $NEURALCORE_PATH"
    else
      echo "ℹ️  No local NeuralCore found — cloning using pyproject.toml link..."
      CORE_GIT_URL=$(grep -oP 'neuralcore @ \Kgit\+https?://[^ "]+' "$PYPROJECT" 2>/dev/null | sed 's|^git+||' || echo "https://github.com/Abyss-c0re/NeuralCore.git")
      if [ ! -d "$CORE_CANDIDATE" ]; then
        git clone "$CORE_GIT_URL" "$CORE_CANDIDATE"
        echo "✅ Cloned NeuralCore to $CORE_CANDIDATE"
      else
        echo "✅ Using existing NeuralCore clone at $CORE_CANDIDATE"
      fi
      NEURALCORE_PATH="$CORE_CANDIDATE"
    fi
  fi

  # ──────────────────────────────────────────────────────────────
  # Add the packages as editable workspace members.
  # We deliberately add NeuralCore *first* (the leaf), then NeuralHub.
  # Adding the dependent (hub) first can cause uv workspace resolution
  # errors because hub's [tool.uv.sources] references neuralcore before
  # neuralcore itself has been registered as a workspace source.
  # Core-first order makes both registrations clean.
  # ──────────────────────────────────────────────────────────────
  echo "🔗 Adding local NeuralCore..."
  uv add --editable "$NEURALCORE_PATH"

  # When using local ./core + ./hub copies, make sure *inside the hub package itself*
  # the dependency on neuralcore is properly switched from the git version to the
  # local editable one. Manual toml patching often leaves the `dependencies = [...]`
  # list still pointing at the git URL. The only reliable way is to let `uv` itself
  # rewrite hub's pyproject.toml and lockfile.
  if [ "$BUNDLE_MODE" = true ] && [ "$DO_PATCH_HUB" = true ]; then
    echo "🔧 Fixing NeuralHub's own dependency on NeuralCore (uv remove + uv add --editable from inside hub)..."
    (
      cd "$SCRIPT_DIR/hub"
      uv remove neuralcore 2>/dev/null || true
      uv add --editable ../core
    )
    echo "✅ NeuralHub now correctly depends on the local ./core"

    # The `uv add` above writes a path-based entry. We must switch it to the
    # workspace form so it doesn't conflict with the root's [tool.uv.workspace] declaration.
    HUB_PYPROJECT="$SCRIPT_DIR/hub/pyproject.toml"
    if [ -f "$HUB_PYPROJECT" ]; then
      echo "🔧 Normalizing NeuralHub's sources to use { workspace = true }..."
      sed -i '/^\s*neuralcore\s*=/c\neuralcore = { workspace = true }' "$HUB_PYPROJECT" 2>/dev/null || true
    fi
  fi

  # Prepare the root pyproject.toml for a proper uv workspace when using local copies.
  # We declare the two packages via [tool.uv.workspace] members (the modern/recommended way).
  if [ "$BUNDLE_MODE" = true ] && [ "$DO_PATCH_HUB" = true ]; then
    echo "🔧 Preparing root pyproject.toml as a workspace (members + sources)..."
    python3 - <<'PYEOF' "$SCRIPT_DIR/pyproject.toml"
import re
import sys

p = sys.argv[1]
with open(p, 'r', encoding='utf-8') as f:
    txt = f.read()

# Pre-clean any old path-based declarations for our workspace members
# (uv really dislikes seeing `neuralcore = { path = ... }` once members are declared)
txt = re.sub(r'^\s*neuralcore\s*=.*path.*$', 'neuralcore = { workspace = true }', txt, flags=re.MULTILINE)
txt = re.sub(r'^\s*neuralhub\s*=.*path.*$',  'neuralhub  = { workspace = true }', txt, flags=re.MULTILINE)

# --- 1. Ensure [tool.uv.workspace] members = ["core", "hub"] ---
workspace_block = '[tool.uv.workspace]\nmembers = ["core", "hub"]\n'

if "[tool.uv.workspace]" not in txt:
    if not txt.endswith("\n"):
        txt += "\n"
    txt += "\n" + workspace_block
else:
    if 'members = ["core", "hub"]' not in txt:
        # Try to inject or fix the members line
        txt = re.sub(
            r'(\[tool\.uv\.workspace\][^\[]*)',
            r'\1members = ["core", "hub"]\n',
            txt, flags=re.DOTALL
        )

# --- 2. Ensure [tool.uv.sources] declares the workspace members using the
#     `workspace = true` form (required once [tool.uv.workspace] members are declared).
#     Using a direct path here is what triggers the exact error the user is seeing.
desired_sources = {
    "neuralcore": 'neuralcore = { workspace = true }',
    "neuralhub":  'neuralhub  = { workspace = true }',
}

if "[tool.uv.sources]" not in txt:
    if not txt.endswith("\n"):
        txt += "\n"
    txt += "\n[tool.uv.sources]\n" + "\n".join(desired_sources.values()) + "\n"
else:
    lines = txt.splitlines(keepends=True)
    out = []
    in_src = False
    seen = {"neuralcore": False, "neuralhub": False}
    for ln in lines:
        if ln.strip().startswith("[") and "[tool.uv.sources]" not in ln:
            if in_src:
                for k, v in desired_sources.items():
                    if not seen[k]:
                        out.append(v + "\n")
            in_src = False
        if "[tool.uv.sources]" in ln:
            in_src = True
            out.append(ln)
            continue
        if in_src:
            m = re.match(r'\s*(neuralcore|neuralhub)\s*=', ln)
            if m:
                k = m.group(1)
                out.append(desired_sources[k] + "\n")
                seen[k] = True
                continue
        out.append(ln)
    if in_src:
        for k, v in desired_sources.items():
            if not seen[k]:
                out.append(v + "\n")
    txt = "".join(out)

with open(p, 'w', encoding='utf-8') as f:
    f.write(txt)
print("Root workspace + sources prepared for core/hub.")
PYEOF
  fi

  echo "🔗 Adding local NeuralHub (this brings its own local neuralcore source)..."
  uv add --editable "$NEURALHUB_PATH"
fi

# Standard Void installation (exactly as in README)
echo "📦 Syncing dependencies..."
uv sync

echo "🛠 Installing NeuralVoid as editable tool..."
uv tool install -e .

echo ""
echo "✅ NeuralVoid installation completed successfully!"
if [ "$BUNDLE_MODE" = true ]; then
  echo "📦 Bundle mode active — editable NeuralCore + NeuralHub linked (local ./core+./hub or parent siblings)."
  echo "   No stray clones left in parent directory unless that is where the sources were found."
elif [ "$DEV_MODE" = true ]; then
  echo "🔧 Dev mode active — edits to NeuralCore and NeuralHub will be live immediately."
fi

echo ""
echo "Next steps:"
echo "  neuralvoid --help"
echo "  (or uv run your-entrypoint)"