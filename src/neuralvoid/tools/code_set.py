import os
import asyncio
from typing import List, Dict, Any, Optional
from collections import defaultdict

from neuralcore.utils.file_helpers import _collect_text_files, _is_text_file
from neuralcore.actions.registry import tool
from neuralcore.utils.logger import Logger

logger = Logger.get_logger()

IGNORE_DIRS = {
    ".git",
    "node_modules",
    "venv",
    "env",
    "__pycache__",
    "build",
    "dist",
    ".venv",
    ".idea",
    ".vscode",
}
IGNORE_FILES = {".DS_Store", "Thumbs.db", ".gitignore"}


# ─────────────────────────────────────────────────────────────
# CODING TOOLS 
# ─────────────────────────────────────────────────────────────


@tool(
    "CodingTools",
    tags=["code", "index", "codebase", "kb"],
    name="index_codebase",
    description="Index all text files in a folder into the agent's knowledge base using read_file + GetContext.",
)
async def index_codebase(
    agent, folder_path: str = ".", recursive: bool = True, max_files: int = 150
) -> str:
    """Index → triggers internal embedding/summarization via read_file."""
    files = await _collect_text_files(folder_path, recursive, max_files)

    if not files:
        return f"✅ Indexed 0 text files from '{folder_path}' (nothing readable found)"

    indexed = 0
    skipped = 0

    for file_path in files:
        try:
            await agent.manager.execute_direct("read_file", file_path=file_path)
            indexed += 1
        except Exception as e:
            logger.debug(f"index skip {file_path}: {e}")
            skipped += 1

    return f"✅ Indexed {indexed} text files from '{folder_path}' (skipped {skipped})"


@tool(
    "CodingTools",
    tags=["code", "read", "codebase"],
    name="read_codebase",
    description="Read + preview all text files in a folder. Returns tree with first 400 chars of each",
)
async def read_codebase(
    agent, folder_path: str = ".", recursive: bool = True, max_files: int = 80
) -> str:
    """Read codebase with smart previews — uses universal read_file."""
    files = await _collect_text_files(folder_path, recursive, max_files)

    if not files:
        return f"❌ No readable text files found in '{folder_path}'."

    lines = [f"📂 Codebase: {os.path.abspath(folder_path)} ({len(files)} text files)"]

    for file_path in files:
        rel = os.path.relpath(file_path, folder_path)
        depth = rel.count(os.sep)
        indent = "  " * (depth + 1)

        lines.append(f"{indent}📄 {rel}")

        try:
            content_result = await agent.manager.execute_direct(
                "read_file", file_path=file_path
            )

            preview = ""
            if isinstance(content_result, str):
                preview = content_result[:450].strip()
            else:
                # Streaming case
                async for chunk in content_result:
                    preview = chunk[:450].strip()
                    break

            if len(preview) > 400:
                preview = preview[:400] + " …"
            lines.append(f"{indent}   └─ {preview or '(empty)'}")

        except Exception as e:
            lines.append(f"{indent}   └─ ⚠️ {str(e)[:60]}")

    return "\n".join(lines)


@tool(
    "CodingTools",
    tags=["code", "list", "structure"],
    name="list_code_files",
    description="List all text files in a folder (async content detection, no extensions).",
)
async def list_code_files(folder_path: str = ".", recursive: bool = True) -> str:
    """Pure listing — fast and extension-free."""
    files = await _collect_text_files(folder_path, recursive, max_files=1000)
    if not files:
        return "(no text files found)"
    return "\n".join(sorted(files))


@tool(
    "CodingTools",
    tags=["code", "search", "codebase"],
    name="search_code",
    description="Search text inside all text files (case-insensitive). Returns file:line:match.",
)
async def search_code(
    agent, pattern: str, folder_path: str = ".", recursive: bool = True
) -> str:
    """Grep-like search across entire codebase using read_file."""
    files = await _collect_text_files(folder_path, recursive, max_files=600)
    if not files:
        return "(no text files to search)"

    results: List[str] = []
    pattern_lower = pattern.lower()

    for file_path in files:
        try:
            content = await agent.manager.execute_direct(
                "read_file", file_path=file_path
            )
            if isinstance(content, str):
                for i, line in enumerate(content.splitlines(), 1):
                    if pattern_lower in line.lower():
                        results.append(f"{file_path}:{i}: {line.strip()[:120]}")
        except Exception:
            continue

    return "\n".join(results) if results else f"(no matches for '{pattern}')"


@tool(
    "CodingTools",
    tags=["code", "structure", "tree"],
    name="get_project_structure",
    description="Beautiful folder tree of only text files (max_depth limit). Context overview.",
)
async def get_project_structure(folder_path: str = ".", max_depth: int = 4) -> str:
    """Async text-aware project tree."""
    if not os.path.isdir(folder_path):
        return f"❌ Folder not found: {folder_path}"

    lines = [f"📂 {os.path.basename(os.path.abspath(folder_path)) or 'root'}"]

    for root, dirs, files in os.walk(folder_path):
        depth = root.count(os.sep) - os.path.abspath(folder_path).count(os.sep)
        if depth > max_depth:
            continue

        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        indent = "  " * depth

        for d in sorted(dirs):
            lines.append(f"{indent}📁 {d}/")

        for f in sorted(files):
            if f in IGNORE_FILES:
                continue
            file_path = os.path.join(root, f)
            if await _is_text_file(file_path):
                lines.append(f"{indent}📄 {f}")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# GIT TOOLS 
# ─────────────────────────────────────────────────────────────


@tool(
    "CodingTools",
    tags=["git", "vcs"],
    name="git_status",
    description="git status --short -b",
)
async def git_status(repo_path: str = ".") -> str:
    try:
        proc = await asyncio.create_subprocess_exec(
            "git",
            "-C",
            repo_path,
            "status",
            "--short",
            "-b",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        return stdout.decode().strip() or "✅ Clean working tree."
    except Exception as e:
        return f"git_status error: {e}"


@tool(
    "CodingTools",
    tags=["git", "diff"],
    name="git_diff",
    description="git diff (staged or unstaged)",
)
async def git_diff(repo_path: str = ".", staged: bool = False) -> str:
    args = ["git", "-C", repo_path, "diff", "--no-color", "--unified=3"]
    if staged:
        args.insert(-1, "--cached")
    try:
        proc = await asyncio.create_subprocess_exec(
            *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, _ = await proc.communicate()
        return stdout.decode() or "(no changes)"
    except Exception as e:
        return f"git_diff error: {e}"


@tool(
    "CodingTools",
    tags=["git", "commit"],
    name="git_commit",
    description="Stage + commit (safe, no push). require_confirmation=True",
    require_confirmation=True,
)
async def git_commit(repo_path: str = ".", message: str = "AI-assisted changes") -> str:
    try:
        await asyncio.create_subprocess_exec("git", "-C", repo_path, "add", ".")
        proc = await asyncio.create_subprocess_exec(
            "git",
            "-C",
            repo_path,
            "commit",
            "-m",
            message,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        return f"✅ Committed: {stdout.decode().strip() or stderr.decode().strip()}"
    except Exception as e:
        return f"git_commit error: {e}"


@tool(
    "CodingTools",
    tags=["git", "branch"],
    name="git_branch",
    description="List or create/switch branch",
)
async def git_branch(repo_path: str = ".", new_branch: str = "") -> str:
    cmd = (
        ["git", "-C", repo_path, "checkout", "-b", new_branch]
        if new_branch
        else ["git", "-C", repo_path, "branch", "--show-current"]
    )
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, _ = await proc.communicate()
        return stdout.decode().strip()
    except Exception as e:
        return f"git_branch error: {e}"


# ─────────────────────────────────────────────────────────────
# TREE-SITTER SECTION (LLM-friendly structural parsing)
# ─────────────────────────────────────────────────────────────

# Tree-sitter setup (lazy + graceful fallback)
TREE_SITTER_LANGS: Dict[str, Any] = {}
TREE_SITTER_AVAILABLE = False


def _init_tree_sitter():
    global TREE_SITTER_AVAILABLE, TREE_SITTER_LANGS
    if TREE_SITTER_AVAILABLE:
        return

    errors = []
    try:
        from tree_sitter import Language, # type: ignore
    except Exception as e:
        errors.append(f"tree_sitter core: {e}")
        TREE_SITTER_AVAILABLE = False
        logger.warning("tree-sitter core not importable: " + "; ".join(errors))
        return

    langs_to_try = [
        ("python", "tree_sitter_python"),
        ("lua", "tree_sitter_lua"),
        ("c", "tree_sitter_c"),
        ("cpp", "tree_sitter_cpp"),
    ]

    success = {}
    for key, mod_name in langs_to_try:
        try:
            mod = __import__(mod_name)
            # Modern API: mod.language() returns a capsule / pointer
            if hasattr(mod, "language"):
                lang = Language(mod.language())
            elif hasattr(mod, "LANGUAGE"):  # some older wheels
                lang = Language(mod.LANGUAGE)
            else:
                # last resort - sometimes the module itself is the language
                lang = Language(mod)
            success[key] = lang
        except Exception as e:
            errors.append(f"{mod_name}: {e}")

    if len(success) == 4:
        TREE_SITTER_LANGS.update(success)
        TREE_SITTER_AVAILABLE = True
        logger.info("✅ tree-sitter initialized successfully (python, lua, c, cpp)")
    else:
        TREE_SITTER_AVAILABLE = False
        logger.warning(
            f"tree-sitter partially available ({len(success)}/4 languages). "
            f"Missing: {', '.join(k for k, _ in langs_to_try if k not in success)}. "
            f"Errors: {'; '.join(errors)}"
        )
        # Still store what we have
        TREE_SITTER_LANGS.update(success)


def _get_language_and_name(file_path: str) -> Optional[tuple[Any, str]]:
    """Return (Language object, lang_name) or None if unsupported / not available."""
    _init_tree_sitter()
    if not TREE_SITTER_AVAILABLE:
        return None

    ext = os.path.splitext(file_path)[1].lower()
    mapping = {
        ".py": ("python", "python"),
        ".lua": ("lua", "lua"),
        ".c": ("c", "c"),
        ".h": ("c", "c"),
        ".cpp": ("cpp", "cpp"),
        ".cc": ("cpp", "cpp"),
        ".cxx": ("cpp", "cpp"),
        ".hpp": ("cpp", "cpp"),
        ".hh": ("cpp", "cpp"),
        ".hxx": ("cpp", "cpp"),
    }
    if ext in mapping:
        lang_key, name = mapping[ext]
        lang = TREE_SITTER_LANGS.get(lang_key)
        if lang is not None:
            return lang, name
    return None


# ─────────────────────────────────────────────────────────────
# ASYNC FILE READER via agent
# ─────────────────────────────────────────────────────────────


async def _read_file_content(agent, file_path: str) -> bytes:
    """
    Read file content using the agent's read_file tool (respects permissions, caching, etc.).
    Returns UTF-8 bytes suitable for tree-sitter.
    Handles both str return and streaming return.
    """
    try:
        content_result = await agent.manager.execute_direct(
            "read_file", file_path=file_path
        )

        if isinstance(content_result, str):
            return content_result.encode("utf-8", errors="replace")

        # Streaming case - collect all chunks
        full_parts: List[str] = []
        async for chunk in content_result:
            if isinstance(chunk, bytes):
                full_parts.append(chunk.decode("utf-8", errors="replace"))
            elif isinstance(chunk, str):
                full_parts.append(chunk)
            else:
                full_parts.append(str(chunk))
        return "".join(full_parts).encode("utf-8", errors="replace")

    except Exception as e:
        logger.debug(f"_read_file_content failed for {file_path}: {e}")
        return b""


# ─────────────────────────────────────────────────────────────
# SYMBOL EXTRACTION (LLM-friendly, robust across languages)
# ─────────────────────────────────────────────────────────────


def _extract_symbols(tree, source: bytes, lang_name: str) -> List[Dict[str, Any]]:
    """Traverse tree-sitter tree and extract classes/functions/methods with signatures."""
    symbols: List[Dict[str, Any]] = []

    def get_text(node) -> str:
        return (
            source[node.start_byte : node.end_byte]
            .decode("utf-8", errors="ignore")
            .strip()
        )

    def find_child(node, *types) -> Optional[Any]:
        for child in node.children:
            if child.type in types:
                return child
        return None

    def traverse(node, parent_type: str = ""):
        node_type = node.type

        # Language-agnostic definition detection
        is_definition = any(
            kw in node_type
            for kw in (
                "function_definition",
                "method_definition",
                "function_declaration",
                "class_definition",
                "class_declaration",
                "class_specifier",
                "struct_specifier",
                "local_function",
                "function",
            )
        )

        if is_definition:
            name_node = find_child(node, "identifier", "name", "type_identifier")
            name = get_text(name_node) if name_node else "<anonymous>"

            # Parameters (works for most languages)
            params_node = find_child(
                node,
                "parameters",
                "parameter_list",
                "formal_parameters",
                "argument_list",
                "parameters",
            )
            params = get_text(params_node) if params_node else "()"

            # Try to get return type / declarator info for C/C++
            return_type = ""
            if lang_name in ("c", "cpp"):
                type_node = find_child(
                    node, "type", "primitive_type", "type_identifier"
                )
                if type_node:
                    return_type = get_text(type_node) + " "

            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1

            # Python docstring extraction (first string in body)
            docstring = ""
            if lang_name == "python" and node_type in (
                "function_definition",
                "class_definition",
            ):
                block = find_child(node, "block", "suite")
                if block:
                    for child in block.children:
                        if child.type == "expression_statement":
                            string_node = find_child(child, "string")
                            if string_node:
                                raw = get_text(string_node)
                                if raw.startswith(('"""', "'''")) and len(raw) > 6:
                                    docstring = raw[3:-3].strip()[:180]
                                elif raw.startswith(('"', "'")) and len(raw) > 2:
                                    docstring = raw[1:-1].strip()[:180]
                                break
                        if docstring:
                            break

            symbols.append(
                {
                    "type": node_type,
                    "name": name,
                    "params": params,
                    "return_type": return_type,
                    "start_line": start_line,
                    "end_line": end_line,
                    "docstring": docstring,
                    "language": lang_name,
                    "parent_type": parent_type,
                }
            )

        # Recurse (track parent for context)
        for child in node.children:
            traverse(child, node_type)

    traverse(tree.root_node)
    return symbols


# ─────────────────────────────────────────────────────────────
# TREE-SITTER TOOLS (async + use agent.read_file)
# ─────────────────────────────────────────────────────────────


@tool(
    "CodingTools",
    tags=["code", "parse", "treesitter", "symbols", "structure", "llm-friendly"],
    name="parse_codebase_with_treesitter",
    description="Parse entire folder with tree-sitter (Python/Lua/C/C++). Returns beautiful LLM-friendly symbol map: file → classes/functions/methods + signatures + docstrings + line ranges. Uses agent's read_file (no direct FS access).",
)
async def parse_codebase_with_treesitter(
    agent, folder_path: str = ".", recursive: bool = True, max_files: int = 120
) -> str:
    """Main tree-sitter powered codebase parser → structured symbol overview (via agent.read_file)."""
    _init_tree_sitter()
    if not TREE_SITTER_AVAILABLE:
        return (
            "❌ tree-sitter not available.\n"
            "Install: pip install tree-sitter tree-sitter-python tree-sitter-lua tree-sitter-c tree-sitter-cpp"
        )

    # Collect only text files, then filter to supported extensions
    all_files = await _collect_text_files(folder_path, recursive, max_files)
    supported_exts = {
        ".py",
        ".lua",
        ".c",
        ".h",
        ".cpp",
        ".cc",
        ".cxx",
        ".hpp",
        ".hh",
        ".hxx",
    }
    files = [f for f in all_files if os.path.splitext(f)[1].lower() in supported_exts]

    if not files:
        return f"❌ No supported code files (py/lua/c/cpp) found in '{folder_path}'."

    file_summaries: List[str] = []
    all_symbols: List[Dict[str, Any]] = []

    for file_path in files:
        result = _get_language_and_name(file_path)
        if result is None:
            continue
        lang, lang_name = result

        try:
            source = await _read_file_content(agent, file_path)
            if not source:
                continue

            from tree_sitter import Parser  # type: ignore

            parser = Parser(lang)
            tree = parser.parse(source)

            symbols = _extract_symbols(tree, source, lang_name)
            rel = os.path.relpath(file_path, folder_path)

            file_summaries.append(f"📄 {rel} ({lang_name}, {len(symbols)} symbols)")

            for sym in symbols:
                sym["file"] = rel
                all_symbols.append(sym)

        except Exception as e:
            logger.debug(f"treesitter parse error {file_path}: {e}")
            file_summaries.append(
                f"⚠️ {os.path.relpath(file_path, folder_path)}: {str(e)[:60]}"
            )

    if not all_symbols:
        return (
            "✅ Scanned files but found no parseable symbols (check language support)."
        )

    # Build beautiful LLM-friendly output
    lines: List[str] = [
        "🗺️  CODEBASE SYMBOL MAP (tree-sitter)",
        f"   Folder: {os.path.abspath(folder_path)}",
        f"   Files parsed: {len(files)} | Symbols found: {len(all_symbols)}",
        "",
    ]

    by_file: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for sym in all_symbols:
        by_file[sym["file"]].append(sym)

    for file_path in sorted(by_file.keys()):
        syms = sorted(by_file[file_path], key=lambda s: s["start_line"])
        lines.append(f"\n📁 {file_path}")

        for sym in syms:
            indent = (
                "    "
                if "method" in sym["type"]
                or sym.get("parent_type", "").endswith("class")
                else "  "
            )
            typ = sym["type"].replace("_", " ").title()
            ret = sym.get("return_type", "")
            doc = f"  📝 {sym['docstring']}" if sym.get("docstring") else ""
            lines.append(
                f"{indent}{sym['start_line']:4d}-{sym['end_line']:4d}  {typ}: {ret}{sym['name']}{sym['params']}{doc}"
            )

    lines.append(
        "\n✅ Use this map to navigate, plan changes, or feed into other agents."
    )
    return "\n".join(lines)


@tool(
    "CodingTools",
    tags=["code", "parse", "treesitter", "single-file", "structure"],
    name="analyze_file_with_treesitter",
    description="Parse ONE file with tree-sitter and return detailed LLM-friendly structure (symbols + context). Uses agent's read_file.",
)
async def analyze_file_with_treesitter(agent, file_path: str) -> str:
    """Detailed single-file parse (tree-sitter via agent.read_file)."""
    _init_tree_sitter()
    if not TREE_SITTER_AVAILABLE:
        return "❌ tree-sitter not available. See parse_codebase_with_treesitter for install instructions."

    if not os.path.isfile(file_path):
        return f"❌ File not found: {file_path}"

    result = _get_language_and_name(file_path)
    if result is None:
        return f"❌ Unsupported extension for tree-sitter: {file_path} (supported: .py .lua .c .cpp etc.)"
    lang, lang_name = result

    try:
        source = await _read_file_content(agent, file_path)
        if not source:
            return f"❌ Could not read content via agent: {file_path}"

        from tree_sitter import Parser  # type: ignore

        parser = Parser(lang)
        tree = parser.parse(source)

        symbols = _extract_symbols(tree, source, lang_name)
        rel = os.path.basename(file_path)

        lines = [
            f"🔍 TREE-SITTER ANALYSIS: {rel} ({lang_name})",
            f"   Total symbols: {len(symbols)} | Lines: {len(source.splitlines())}",
            "",
        ]

        if not symbols:
            lines.append("(No top-level definitions found — may be data/config file)")
            return "\n".join(lines)

        for sym in sorted(symbols, key=lambda s: s["start_line"]):
            typ = sym["type"].replace("_", " ").title()
            ret = sym.get("return_type", "")
            doc = f"\n      📝 {sym['docstring']}" if sym.get("docstring") else ""
            lines.append(
                f"  {sym['start_line']:4d}-{sym['end_line']:4d}  {typ}: {ret}{sym['name']}{sym['params']}{doc}"
            )

        lines.append("\n✅ Ready for LLM consumption or further agent actions.")
        return "\n".join(lines)

    except Exception as e:
        return f"❌ Parse error: {e}"


@tool(
    "CodingTools",
    tags=["code", "search", "treesitter", "symbols"],
    name="search_symbols_with_treesitter",
    description="Search for function/class/method names across the codebase using tree-sitter (exact or fuzzy). Returns file:line + signature. Uses agent's read_file.",
)
async def search_symbols_with_treesitter(
    agent,
    pattern: str,
    folder_path: str = ".",
    recursive: bool = True,
    max_files: int = 150,
) -> str:
    """Symbol search powered by tree-sitter (much more precise than text grep)."""
    _init_tree_sitter()
    if not TREE_SITTER_AVAILABLE:
        return "❌ tree-sitter not available."

    all_files = await _collect_text_files(folder_path, recursive, max_files)
    supported_exts = {
        ".py",
        ".lua",
        ".c",
        ".h",
        ".cpp",
        ".cc",
        ".cxx",
        ".hpp",
        ".hh",
        ".hxx",
    }
    files = [f for f in all_files if os.path.splitext(f)[1].lower() in supported_exts]

    if not files:
        return "(no supported code files)"

    pattern_lower = pattern.lower()
    results: List[str] = []

    for file_path in files:
        result = _get_language_and_name(file_path)
        if result is None:
            continue
        lang, lang_name = result
        try:
            source = await _read_file_content(agent, file_path)
            if not source:
                continue

            from tree_sitter import Parser  # type: ignore

            parser = Parser(lang)
            tree = parser.parse(source)
            symbols = _extract_symbols(tree, source, lang_name)

            for sym in symbols:
                if pattern_lower in sym["name"].lower():
                    rel = os.path.relpath(file_path, folder_path)
                    ret = sym.get("return_type", "")
                    results.append(
                        f"{rel}:{sym['start_line']:4d}  {sym['type'].replace('_', ' ')}: {ret}{sym['name']}{sym['params']}"
                    )
        except Exception:
            continue

    if not results:
        return f"(no symbols matching '{pattern}')"

    return "\n".join(sorted(results))


@tool(
    "CodingTools",
    tags=["code", "treesitter", "advanced"],
    name="get_treesitter_tree",
    description="Return raw tree-sitter tree for a file (for advanced agents that want to walk the AST themselves). Uses agent's read_file.",
)
async def get_treesitter_tree(agent, file_path: str) -> str:
    """Advanced: returns string representation of the full parse tree (use sparingly — can be large)."""
    _init_tree_sitter()
    if not TREE_SITTER_AVAILABLE:
        return "❌ tree-sitter not available."

    result = _get_language_and_name(file_path)
    if result is None:
        return f"Unsupported: {file_path}"
    lang, _ = result

    try:
        source = await _read_file_content(agent, file_path)
        if not source:
            return f"❌ Could not read via agent: {file_path}"

        from tree_sitter import Parser  # type: ignore

        parser = Parser(lang)
        tree = parser.parse(source)
        sexp = tree.root_node.sexp()
        return sexp[:8000] + (" … (truncated)" if len(sexp) > 8000 else "")
    except Exception as e:
        return f"Error: {e}"


# ─────────────────────────────────────────────────────────────
# STANDALONE TEST / DIAGNOSTIC (run with: uv run python src/neuralvoid/tools/code_set.py --test)
# ─────────────────────────────────────────────────────────────


def test_tree_sitter():
    """Run this to diagnose import / initialization issues."""
    print("=== Tree-sitter Diagnostic ===")
    print(f"Python version: {__import__('sys').version.split()[0]}")

    try:
        import tree_sitter

        print(
            f"✅ tree_sitter core: {tree_sitter.__version__} @ {tree_sitter.__file__}"
        )
    except Exception as e:
        print(f"❌ tree_sitter core import failed: {e}")
        return

    for name in [
        "tree_sitter_python",
        "tree_sitter_lua",
        "tree_sitter_c",
        "tree_sitter_cpp",
    ]:
        try:
            mod = __import__(name)
            has_lang = hasattr(mod, "language")
            print(f"✅ {name}: {mod.__file__} | has .language(): {has_lang}")
            if has_lang:
                try:
                    from tree_sitter import Language

                    _ = Language(mod.language())
                    print(f"   → Language() created successfully")
                except Exception as e:
                    print(f"   → Language() failed: {e}")
        except Exception as e:
            print(f"❌ {name} import failed: {e}")

    print("\n--- Full init test ---")
    _init_tree_sitter()
    print(f"TREE_SITTER_AVAILABLE = {TREE_SITTER_AVAILABLE}")
    print(f"Languages loaded: {list(TREE_SITTER_LANGS.keys())}")


if __name__ == "__main__":
    import sys

    if "--test" in sys.argv or "test" in sys.argv:
        test_tree_sitter()
    else:
        print("NeuralCore CodingTools (merged) loaded. Use --test for diagnostics.")
