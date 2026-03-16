from neuralcore.actions.actions import Action, ActionSet

from neuralcore.utils.terminal_utils import *


# ─────────────────────────────────────────────────────────────
#  Action definitions (parameter names must match executors)
# ─────────────────────────────────────────────────────────────

ls_action = Action(
    name="ls",
    description="List files in a directory",
    tags=[
        "filesystem","file","directory","list","browse","navigation",
        "shell","terminal","system","developer","inspect","structure"
    ],
    parameters={
        "path": {"type": "string", "description": "Directory path, default '.'"}
    },
    executor=exec_ls,
    required=[],
)

pwd_action = Action(
    name="pwd",
    description="Print working directory",
    tags=[
        "filesystem","directory","path","location","current","navigation",
        "shell","terminal","system","developer"
    ],
    parameters={},
    executor=exec_pwd,
    required=[],
)

cd_action = Action(
    name="cd",
    description="Change current working directory",
    tags=[
        "filesystem","directory","navigation","path","change","cwd",
        "shell","terminal","system","developer"
    ],
    parameters={"path": {"type": "string", "description": "Directory to change to"}},
    executor=exec_cd,
    required=["path"],
)

cp_action = Action(
    name="cp",
    description="Copy a file or directory",
    tags=[
        "filesystem","file","directory","copy","duplicate",
        "backup","replicate","developer","shell","terminal"
    ],
    parameters={
        "source": {"type": "string"},
        "destination": {"type": "string"},
    },
    executor=exec_cp,
    required=["source","destination"],
)

mv_action = Action(
    name="mv",
    description="Move or rename a file/directory",
    tags=[
        "filesystem","file","directory","move","rename","transfer",
        "organize","developer","shell","terminal"
    ],
    parameters={
        "source": {"type": "string"},
        "destination": {"type": "string"},
    },
    executor=exec_mv,
    required=["source","destination"],
)

mkdir_action = Action(
    name="mkdir",
    description="Create a new directory",
    tags=[
        "filesystem","directory","create","folder",
        "structure","project","developer","shell","terminal"
    ],
    parameters={"path": {"type": "string"}},
    executor=exec_mkdir,
    required=["path"],
)

cat_action = Action(
    name="cat",
    description="Display file contents",
    tags=[
        "filesystem","file","read","view","display","content",
        "text","code","inspect","developer","shell","terminal"
    ],
    parameters={"file_path": {"type": "string", "description": "Path to the file"}},
    executor=exec_cat,
    required=["file_path"],
)

delete_file_action = Action(
    name="delete_file",
    description="Delete a file (requires human confirmation)",
    tags=[
        "filesystem","file","delete","remove","dangerous",
        "cleanup","developer","shell","terminal"
    ],
    parameters={"file_path": {"type": "string"}},
    executor=exec_delete_file,
    required=["file_path"],
    require_confirmation=True,
)

delete_dir_action = Action(
    name="delete_dir",
    description="Delete directory recursively",
    tags=[
        "filesystem","directory","delete","remove","recursive",
        "dangerous","cleanup","developer","shell","terminal"
    ],
    parameters={"dir_path": {"type": "string"}},
    executor=exec_delete_dir,
    required=["dir_path"],
    require_confirmation=True,
)

find_action = Action(
    name="find",
    description="Find files in directory optionally by name",
    tags=[
        "filesystem","file","directory","search","lookup","locate",
        "pattern","discover","developer","shell","terminal"
    ],
    parameters={
        "path": {"type": "string"},
        "name": {"type": "string"},
    },
    executor=exec_find,
    required=[],
)



touch_action = Action(
    name="touch",
    description="Create empty file or update its timestamp",
    tags=[
        "filesystem","file","create","empty","timestamp","update",
        "developer","shell","terminal"
    ],
    parameters={"file_path": {"type": "string", "description": "Path to the file"}},
    executor=exec_touch,
    required=["file_path"],
)

head_action = Action(
    name="head",
    description="Show first lines of a file",
    tags=[
        "filesystem","file","read","preview","text","lines",
        "log","inspect","developer","shell","terminal"
    ],
    parameters={
        "file_path": {"type": "string", "description": "Path to the file"},
        "lines": {"type": "integer", "description": "Number of lines", "default": 10},
    },
    executor=exec_head,
    required=["file_path"],
)

tail_action = Action(
    name="tail",
    description="Show last lines of a file",
    tags=[
        "filesystem","file","read","logs","text","monitor",
        "stream","inspect","developer","shell","terminal"
    ],
    parameters={
        "file_path": {"type": "string", "description": "Path to the file"},
        "lines": {"type": "integer", "description": "Number of lines", "default": 10},
    },
    executor=exec_tail,
    required=["file_path"],
)

wc_action = Action(
    name="wc",
    description="Count lines, words, characters in a file",
    tags=[
        "filesystem","file","analysis","count","lines","words",
        "characters","text","statistics","developer","shell"
    ],
    parameters={
        "file_path": {"type": "string"},
        "lines": {"type": "boolean","default": True},
        "words": {"type": "boolean","default": True},
        "chars": {"type": "boolean","default": False},
    },
    executor=exec_wc,
    required=["file_path"],
)


grep_action = Action(
    name="grep",
    description="Search for pattern in file or directory",
    tags=[
        "filesystem","file","search","pattern","text","regex",
        "code","log","analysis","developer","shell","terminal"
    ],
    parameters={
        "pattern": {"type": "string"},
        "file_path": {"type": "string"},
        "recursive": {"type": "boolean","default": False},
        "case_sensitive": {"type": "boolean","default": True},
    },
    executor=exec_grep,
    required=["pattern","file_path"],
)


tree_action = Action(
    name="tree",
    description="Display directory tree structure",
    tags=[
        "filesystem","directory","structure","hierarchy",
        "visualize","navigation","project","developer","shell"
    ],
    parameters={
        "path": {"type": "string"},
        "max_depth": {"type": "integer","default": 3},
    },
    executor=exec_tree,
    required=[],
)


file_action = Action(
    name="file",
    description="Determine file type",
    tags=[
        "filesystem","file","type","format","metadata",
        "inspect","analysis","developer","shell"
    ],
    parameters={"path": {"type": "string"}},
    executor=exec_file,
    required=["path"],
)


stat_action = Action(
    name="stat",
    description="Show detailed file status",
    tags=[
        "filesystem","file","metadata","permissions","size",
        "timestamps","inspect","analysis","developer","system"
    ],
    parameters={"path": {"type": "string"}},
    executor=exec_stat,
    required=["path"],
)


realpath_action = Action(
    name="realpath",
    description="Resolve absolute path",
    tags=[
        "filesystem","path","absolute","resolve",
        "navigation","location","developer","shell"
    ],
    parameters={"path": {"type": "string"}},
    executor=exec_realpath,
    required=["path"],
)


which_action = Action(
    name="which",
    description="Locate command in PATH",
    tags=[
        "system","command","binary","executable","path",
        "lookup","environment","developer","shell","terminal"
    ],
    parameters={"command": {"type": "string"}},
    executor=exec_which,
    required=["command"],
)


awk_action = Action(
    name="awk",
    description="Process file with awk script",
    tags=[
        "filesystem","file","text","processing","transform",
        "script","pattern","data","analysis","developer","shell"
    ],
    parameters={
        "file_path": {"type": "string"},
        "script": {"type": "string"},
    },
    executor=exec_awk,
    required=["file_path","script"],
)



# ─────────────────────────────────────────────────────────────
# Putting it all together
# ─────────────────────────────────────────────────────────────
def get_terminal_actions():
    terminal_tools = ActionSet(
        name="TerminalCommands",
        description=(
            "Standard shell-style commands for working directly in the terminal environment: "
            "navigate directories (cd, pwd, ls), read/write files (cat, head, tail, write_file), "
            "create/delete/move/copy (touch, mkdir, mv, cp, rm), search and analyze content "
            "(grep, find, wc, awk, sed), view structure (tree), and get file metadata (stat, file, du)."
        )
    )

    for act in [
        pwd_action,
        cd_action,
        ls_action,
        cat_action,
        head_action,
        tail_action,
        touch_action,
        mkdir_action,
        mv_action,
        cp_action,
        delete_file_action,
        delete_dir_action,
        find_action,
        wc_action,
        grep_action,
        tree_action,
        file_action,
        stat_action,
        realpath_action,
        which_action,
        awk_action,
    ]:
        terminal_tools.add(act)
    return terminal_tools
