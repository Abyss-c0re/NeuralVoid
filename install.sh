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
      echo "🔧 Fixing old NeuralCore reference in bundled NeuralHub pyproject (pyright paths)..."
      # Only touch pyright extraPaths (harmless). The real dependency fix happens via
      # `cd hub && uv add --editable ../core` later.
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
  # Register the local packages using uv (each folder is its own project).
  # Order: register core from root first, then fix hub's dependency on core
  # using uv inside the hub project, then register hub from root.
  # uv manages all [tool.uv.sources] entries automatically — no manual TOML editing.
  # ──────────────────────────────────────────────────────────────
  echo "🔗 Adding local NeuralCore..."
  uv add --editable "$NEURALCORE_PATH"

  # When using local ./core + ./hub copies, make sure *inside the hub package itself*
  # the dependency on neuralcore is properly switched from the git version to the
  # local editable one. Manual toml patching often leaves the `dependencies = [...]`
  # list still pointing at the git URL. The only reliable way is to let `uv` itself
  # rewrite hub's pyproject.toml and lockfile.
  if [ "$BUNDLE_MODE" = true ] && [ "$DO_PATCH_HUB" = true ]; then
    # Repair any broken [tool.uv.workspace] left from earlier script versions.
    # We only touch it if it contains the known-bad pattern ("members = members =").
    # Otherwise we leave it alone so uv can manage the workspace + sources properly.
    ROOT_PYPROJECT="$SCRIPT_DIR/pyproject.toml"
    if grep -q 'members = members =' "$ROOT_PYPROJECT" 2>/dev/null; then
      echo "🔧 Repairing broken [tool.uv.workspace] section in root..."
      python3 - <<'PYEOF' "$ROOT_PYPROJECT"
import sys, re
p = sys.argv[1]
with open(p, 'r', encoding='utf-8') as f:
    txt = f.read()
# Remove the broken workspace block
txt = re.sub(r'\n?\[tool\.uv\.workspace\][^\[]*?(?=\n\[|\Z)', '', txt, flags=re.DOTALL)
# Also remove a following sources block if it was part of the mess
txt = re.sub(r'\n?\[tool\.uv\.sources\][^\[]*?(?=\n\[|\Z)', '', txt, flags=re.DOTALL)
with open(p, 'w', encoding='utf-8') as f:
    f.write(txt)
print("Repaired broken workspace section.")
PYEOF
    fi

    # After the core add, uv may have created workspace members.
    # Ensure the root has the correct sources declarations using the workspace form
    # (this is what uv itself recommends in the error messages).
    # We do a minimal, safe "append or fix the two lines" only for these entries.
    python3 - <<'PYEOF' "$ROOT_PYPROJECT"
import sys, re
p = sys.argv[1]
with open(p, 'r', encoding='utf-8') as f:
    txt = f.read()

desired = {
    "neuralcore": "neuralcore = { workspace = true }",
    "neuralhub":  "neuralhub  = { workspace = true }",
}

if "[tool.uv.sources]" not in txt:
    if not txt.endswith("\n"):
        txt += "\n"
    txt += "\n[tool.uv.sources]\n" + "\n".join(desired.values()) + "\n"
else:
    # Replace or add the two keys inside the existing sources section
    for key, line in desired.items():
        pattern = rf'^\s*{key}\s*=.*$'
        if re.search(pattern, txt, flags=re.MULTILINE):
            txt = re.sub(pattern, line, txt, flags=re.MULTILINE)
        else:
            # Insert right after the [tool.uv.sources] line
            txt = re.sub(r'(\[tool\.uv\.sources\])', r'\1\n' + line, txt, count=1)
with open(p, 'w', encoding='utf-8') as f:
    f.write(txt)
print("Ensured correct workspace sources in root.")
PYEOF

    echo "🔧 Fixing NeuralHub's own dependency on NeuralCore (uv remove + uv add --editable from inside hub)..."
    (
      cd "$SCRIPT_DIR/hub"
      uv remove neuralcore 2>/dev/null || true
      uv add --editable ../core
    )
    echo "✅ NeuralHub now correctly depends on the local ./core"

    # uv's "add --editable ../core" inside hub writes a path reference.
    # Once the root treats core/hub as workspace members (which uv does automatically
    # when it sees subdirectories with pyproject.toml), hub must use the workspace form.
    # We do a minimal, safe normalization of just that one line.
    sed -i '/^\s*neuralcore\s*=/c\neuralcore = { workspace = true }' "$SCRIPT_DIR/hub/pyproject.toml" 2>/dev/null || true
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