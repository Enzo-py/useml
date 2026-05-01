import ast
import inspect
import os
import types as _types
import warnings
from typing import Dict, Optional, Set

from ..session.component import Component


def _get_project_root() -> str:
    return os.getcwd()


def _is_local_file(path: str, project_root: str) -> bool:
    if path is None:
        return False
    return os.path.abspath(path).startswith(project_root)


def _resolve_import(module_name: str, project_root: str) -> Optional[str]:
    """Returns the source file for a module if it lives inside the project.

    Args:
        module_name: Dotted module name to resolve (e.g. ``"models.net"``).
        project_root: Absolute path to the project root directory.

    Returns:
        Absolute path to the source file, or None if unavailable.
    """
    try:
        module = __import__(module_name, fromlist=["*"])
        return inspect.getsourcefile(module)
    except Exception:
        return None


def _extract_local_imports(file_path: str, project_root: str) -> Set[str]:
    """Collects absolute paths of local files imported by the given file.

    Args:
        file_path: Absolute path to the Python file to analyse.
        project_root: Absolute path to the project root directory.

    Returns:
        Set of absolute paths to locally imported source files.
    """
    imports: Set[str] = set()
    try:
        with open(file_path, "r") as f:
            tree = ast.parse(f.read())
    except (OSError, SyntaxError):
        return imports

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module)

    local_files: Set[str] = set()
    for mod in imports:
        path = _resolve_import(mod, project_root)
        if path and _is_local_file(path, project_root):
            local_files.add(os.path.abspath(path))
    return local_files


def _collect_recursive(
    file_path: str, project_root: str, visited: Set[str]
) -> Set[str]:
    """Recursively collects all local source files reachable from file_path.

    Args:
        file_path: Starting file (absolute path).
        project_root: Absolute path to the project root directory.
        visited: Set of already-visited paths (mutated in place).

    Returns:
        The updated visited set.
    """
    file_path = os.path.abspath(file_path)
    if file_path in visited:
        return visited
    visited.add(file_path)
    for dep in _extract_local_imports(file_path, project_root):
        _collect_recursive(dep, project_root, visited)
    return visited


def _get_notebook_history() -> str:
    """Returns the full cell history of the active IPython/Jupyter session.

    Returns:
        All input cells joined by cell separators, or an empty string.
    """
    try:
        from IPython import get_ipython
        shell = get_ipython()
        if shell:
            return "\n\n# --- Cell ---\n".join(shell.user_ns.get("In", []))
    except Exception:
        pass
    return ""


def _extract_relevant_notebook_cells(target_names: Set[str]) -> str:
    """Extracts the notebook cells that define the given names.

    Args:
        target_names: Set of symbol names to search for.

    Returns:
        Selected cells joined by separators, or an empty string.
    """
    try:
        from IPython import get_ipython
        shell = get_ipython()
        if not shell:
            return ""
        history = shell.user_ns.get("In", [])
        if not history:
            return ""

        selected: list = []
        needed = set(target_names)

        for cell in reversed(history):
            if not cell.strip():
                continue
            try:
                tree = ast.parse(cell)
            except SyntaxError:
                continue

            defined: Set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                    defined.add(node.name)
                elif isinstance(node, ast.Assign):
                    for t in node.targets:
                        if isinstance(t, ast.Name):
                            defined.add(t.id)

            if defined & needed:
                selected.append(cell)
                needed |= defined

        selected.reverse()
        return "\n\n# --- Cell ---\n".join(selected)
    except Exception:
        return ""


def _extract_class_source(
    cls: type, project_root: str, assets: Dict[str, str]
) -> None:
    """Archives the source file for a single class into assets.

    Follows a two-path logic:
    - Local project file → stored under its relative path.
    - External or in-memory → stored under ``losses/<ClassName>.py``.

    Args:
        cls: The class whose source should be archived.
        project_root: Absolute path to the project root directory.
        assets: Mutable dict mapping relative paths to file contents.
    """
    try:
        src_file = inspect.getsourcefile(cls)
    except TypeError:
        src_file = None

    if src_file:
        src_file = os.path.abspath(src_file)
        if _is_local_file(src_file, project_root):
            rel = os.path.relpath(src_file, project_root)
            if rel not in assets:
                try:
                    with open(src_file, "r") as f:
                        assets[rel] = f.read()
                except OSError as exc:
                    warnings.warn(
                        f"Could not archive source for '{cls.__name__}' "
                        f"from '{src_file}': {exc}",
                        UserWarning,
                        stacklevel=2,
                    )
            return

        if os.path.isfile(src_file) and "site-packages" not in src_file:
            key = f"losses/{cls.__name__}.py"
            if key not in assets:
                try:
                    with open(src_file, "r") as f:
                        assets[key] = f.read()
                except OSError as exc:
                    warnings.warn(
                        f"Could not archive loss source for '{cls.__name__}' "
                        f"from '{src_file}': {exc}",
                        UserWarning,
                        stacklevel=2,
                    )
            return

    try:
        src = inspect.getsource(cls)
        if src.strip():
            assets[f"losses/{cls.__name__}.py"] = src
    except (OSError, TypeError):
        warnings.warn(
            f"Could not retrieve source for '{cls.__name__}'. "
            "The loss implementation will not be archived in this snapshot.",
            UserWarning,
            stacklevel=2,
        )


def _get_source_assets(
    components: Dict[str, Component],
) -> Dict[str, str]:
    """Builds a mapping of relative paths to source file contents.

    Covers model source files (steps 1-2), custom loss classes (step 3), and
    an IPython history fallback for purely in-memory classes (step 4).

    Args:
        components: Named Component objects from the current session.

    Returns:
        Mapping of relative paths to their file contents, ready to be
        written into the snapshot ``source/`` directory.
    """
    project_root = _get_project_root()
    all_files: Set[str] = set()
    assets: Dict[str, str] = {}
    pure_notebook_detected = False

    # 1. Resolve model source files
    for comp in components.values():
        model = comp.model
        try:
            file_path = inspect.getsourcefile(model.__class__)
        except TypeError:
            file_path = None

        if file_path:
            file_path = os.path.abspath(file_path)
            if _is_local_file(file_path, project_root):
                _collect_recursive(file_path, project_root, all_files)
                continue

        # External or in-memory class — best-effort extraction
        try:
            src_file = inspect.getsourcefile(model.__class__)
            if (
                src_file
                and os.path.isfile(src_file)
                and "site-packages" not in src_file
            ):
                key = os.path.basename(src_file)
                with open(src_file, "r") as f:
                    assets[key] = f.read()
            else:
                pure_notebook_detected = True
                src = inspect.getsource(model.__class__)
                if src.strip():
                    assets[f"notebook/{model.__class__.__name__}.py"] = src
        except OSError as exc:
            pure_notebook_detected = True
            warnings.warn(
                f"Could not archive source for model "
                f"'{model.__class__.__name__}': {exc}. "
                "Weights will still be saved.",
                UserWarning,
                stacklevel=2,
            )
        except TypeError:
            pure_notebook_detected = True

    # 2. Read file-based model sources
    for file_path in all_files:
        rel = os.path.relpath(file_path, project_root)
        try:
            with open(file_path, "r") as f:
                assets[rel] = f.read()
        except OSError as exc:
            warnings.warn(
                f"Could not read source file '{file_path}': {exc}. "
                "Weights will still be saved.",
                UserWarning,
                stacklevel=2,
            )

    # 3. Archive custom loss sources (deduplicated by class)
    seen_loss_classes: Set[type] = set()
    for comp in components.values():
        if comp.useml_config is None:
            continue
        loss_obj = comp.useml_config.loss_object()
        if loss_obj is None:
            continue

        if isinstance(loss_obj, _types.FunctionType):
            key = f"losses/{loss_obj.__name__}.py"
            if key not in assets:
                try:
                    assets[key] = inspect.getsource(loss_obj)
                except (OSError, TypeError) as exc:
                    warnings.warn(
                        f"Could not archive source for loss function "
                        f"'{loss_obj.__name__}': {exc}. "
                        "Weights will still be saved.",
                        UserWarning,
                        stacklevel=2,
                    )
            continue

        cls = loss_obj if isinstance(loss_obj, type) else type(loss_obj)
        if cls in seen_loss_classes:
            continue
        seen_loss_classes.add(cls)
        _extract_class_source(cls, project_root, assets)

    # 4. IPython history fallback for purely in-memory classes
    if pure_notebook_detected:
        target_names: Set[str] = set()
        for comp in components.values():
            try:
                target_names.add(comp.model.__class__.__name__)
            except AttributeError:
                pass

        focused = _extract_relevant_notebook_cells(target_names)
        if focused.strip():
            assets["notebook/session.py"] = focused
        else:
            full = _get_notebook_history()
            assets["notebook/session_full.py"] = full

    return assets
