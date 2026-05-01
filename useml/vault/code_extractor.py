import inspect
import os
import ast
import types as _types
from typing import Dict, Any, Set

from ..session.component import Component


def _get_project_root() -> str:
    return os.getcwd()


def _is_local_file(path: str, project_root: str) -> bool:
    if path is None:
        return False
    path = os.path.abspath(path)
    return path.startswith(project_root)


def _resolve_import(module_name: str, project_root: str) -> str | None:
    try:
        module = __import__(module_name, fromlist=["*"])
        return inspect.getsourcefile(module)
    except Exception:
        return None


def _extract_local_imports(file_path: str, project_root: str) -> Set[str]:
    imports = set()

    try:
        with open(file_path, "r") as f:
            tree = ast.parse(f.read())
    except Exception:
        return imports

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                imports.add(n.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module)

    local_files = set()
    for mod in imports:
        path = _resolve_import(mod, project_root)
        if path and _is_local_file(path, project_root):
            local_files.add(os.path.abspath(path))

    return local_files


def _extract_relevant_notebook_cells(target_names: Set[str]) -> str:
    try:
        from IPython import get_ipython
        shell = get_ipython()
        if not shell:
            return ""

        history = shell.user_ns.get("In", [])
        if not history:
            return ""

        selected_cells = []
        needed = set(target_names)

        for cell in reversed(history):
            if not cell.strip():
                continue
            try:
                tree = ast.parse(cell)
            except Exception:
                continue

            defined_names = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    defined_names.add(node.name)
                elif isinstance(node, ast.FunctionDef):
                    defined_names.add(node.name)
                elif isinstance(node, ast.Assign):
                    for t in node.targets:
                        if isinstance(t, ast.Name):
                            defined_names.add(t.id)

            if defined_names & needed:
                selected_cells.append(cell)
                needed |= defined_names

        selected_cells.reverse()
        return "\n\n# --- Cell ---\n".join(selected_cells)

    except Exception:
        return ""


def _collect_recursive(file_path: str, project_root: str, visited: Set[str]) -> Set[str]:
    file_path = os.path.abspath(file_path)
    if file_path in visited:
        return visited
    visited.add(file_path)
    for dep in _extract_local_imports(file_path, project_root):
        _collect_recursive(dep, project_root, visited)
    return visited


def _get_notebook_history() -> str:
    try:
        from IPython import get_ipython
        shell = get_ipython()
        if shell:
            return "\n\n# --- Cell ---\n".join(shell.user_ns.get("In", []))
    except Exception:
        pass
    return ""


def _extract_class_source(cls: type, project_root: str, assets: Dict[str, str]) -> None:
    """Archive the source of a single class into assets.

    Follows the same two-path logic as model extraction:
    - local project file  → read the whole file at its relative path
    - external / in-memory → store under losses/<ClassName>.py
    """
    if cls is None:
        return

    try:
        src_file = inspect.getsourcefile(cls)
    except Exception:
        src_file = None

    if src_file:
        src_file = os.path.abspath(src_file)
        if _is_local_file(src_file, project_root):
            rel = os.path.relpath(src_file, project_root)
            if rel not in assets:
                try:
                    with open(src_file, "r") as f:
                        assets[rel] = f.read()
                except Exception:
                    pass
            return

        # External file (e.g. pytest tmp dir)
        if os.path.isfile(src_file) and "site-packages" not in src_file:
            key = f"losses/{cls.__name__}.py"
            if key not in assets:
                try:
                    with open(src_file, "r") as f:
                        assets[key] = f.read()
                except Exception:
                    pass
            return

    # Pure in-memory class
    try:
        src = inspect.getsource(cls)
        if src.strip():
            assets[f"losses/{cls.__name__}.py"] = src
    except Exception:
        pass


def _get_source_assets(components: Dict[str, Component]) -> Dict[str, str]:
    project_root = _get_project_root()
    all_files: Set[str] = set()
    assets: Dict[str, str] = {}

    pure_notebook_detected = False

    # --- 1. Resolve model sources ---
    for name, obj in components.items():
        model = obj.model

        try:
            file_path = inspect.getsourcefile(model.__class__)
        except Exception:
            file_path = None

        # Case 1: local project file
        if file_path:
            file_path = os.path.abspath(file_path)
            if _is_local_file(file_path, project_root):
                _collect_recursive(file_path, project_root, all_files)
                continue

        # Case 2: external or in-memory
        try:
            src_file = inspect.getsourcefile(model.__class__)
            if src_file and os.path.isfile(src_file) and "site-packages" not in src_file:
                actual_filename = os.path.basename(src_file)
                with open(src_file, "r") as f:
                    assets[actual_filename] = f.read()
            else:
                pure_notebook_detected = True
                src = inspect.getsource(model.__class__)
                if src.strip():
                    assets[f"notebook/{model.__class__.__name__}.py"] = src
        except Exception:
            pure_notebook_detected = True

    # --- 2. Load file-based model sources ---
    for file_path in all_files:
        rel_path = os.path.relpath(file_path, project_root)
        try:
            with open(file_path, "r") as f:
                assets[rel_path] = f.read()
        except Exception:
            continue

    # --- 3. Extract custom loss sources (one per unique class, deduped) ---
    seen_loss_classes: Set[type] = set()

    for name, obj in components.items():
        if obj.useml_config is None:
            continue

        loss_obj = obj.useml_config.loss_object()
        if loss_obj is None:
            continue   # built-in string loss — no source to extract

        # Plain function: extract source directly, not via type()
        if isinstance(loss_obj, _types.FunctionType):
            key = f"losses/{loss_obj.__name__}.py"
            if key not in assets:
                try:
                    assets[key] = inspect.getsource(loss_obj)
                except (OSError, TypeError):
                    pass
            continue

        cls = loss_obj if isinstance(loss_obj, type) else type(loss_obj)
        if cls in seen_loss_classes:
            continue
        seen_loss_classes.add(cls)

        _extract_class_source(cls, project_root, assets)

    # --- 4. Notebook history fallback (pure in-memory only) ---
    if pure_notebook_detected:
        target_names = set()
        for obj in components.values():
            try:
                target_names.add(obj.model.__class__.__name__)
            except Exception:
                pass

        focused_history = _extract_relevant_notebook_cells(target_names)
        if focused_history.strip():
            assets["notebook/session.py"] = focused_history
        else:
            full_history = _get_notebook_history()
            assets["notebook/session_full.py"] = full_history

    return assets
