# useml/imports.py

import ast
import sys
import importlib.util
from pathlib import Path
from typing import List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .session.manager import Session


class NothingMountedError(ImportError):
    """Raised when useml.workdir.* is accessed but no snapshot is mounted."""
    pass


def _load_snapshot_module(real_name: str, snapshot_source_path: str):
    """Load a module from a snapshot source directory.

    Temporarily adds the snapshot source to sys.path so that imports within
    the snapshot module (e.g. 'from utils import helper') resolve to files
    inside the snapshot, not the global environment.

    Returns:
        (module, is_package) tuple, or (None, True) for namespace packages.

    Raises:
        ImportError: If the module is not found in the snapshot.
    """
    parts = real_name.split(".")
    base = Path(snapshot_source_path).joinpath(*parts)

    file_candidate = base.with_suffix(".py")
    package_init = base / "__init__.py"

    if file_candidate.exists():
        path = file_candidate
        is_package = False
    elif package_init.exists():
        path = package_init
        is_package = True
    elif base.is_dir():
        return None, True  # namespace package — caller builds the module
    else:
        raise ImportError(
            f"Module {real_name!r} not found in snapshot {snapshot_source_path!r}"
        )

    internal_name = f"_useml_workdir_internal.{real_name}"

    if internal_name in sys.modules:
        return sys.modules[internal_name], is_package

    spec = importlib.util.spec_from_file_location(
        internal_name,
        path,
        submodule_search_locations=[str(base)] if is_package else None,
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[internal_name] = mod

    # Temporarily expose the snapshot source on sys.path so that absolute
    # imports within snapshot modules (e.g. 'import utils') resolve correctly.
    injected = snapshot_source_path not in sys.path
    if injected:
        sys.path.insert(0, snapshot_source_path)
    try:
        spec.loader.exec_module(mod)
    except Exception:
        del sys.modules[internal_name]
        raise
    finally:
        if injected and snapshot_source_path in sys.path:
            sys.path.remove(snapshot_source_path)

    return mod, is_package


class ImportManager:
    """Manages dynamic import resolution for the useml.workdir namespace.

    Provides introspection utilities to show what's importable from a mounted
    snapshot and what imports are declared in __main__ / notebook cells.
    """

    def __init__(self, session: "Session") -> None:
        self._session = session

    @property
    def is_mounted(self) -> bool:
        return self._session._mounted_snapshot is not None

    @property
    def mounted_path(self) -> Optional[Path]:
        if self._session._mounted_sys_path:
            return Path(self._session._mounted_sys_path)
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def available_modules(self) -> List[str]:
        """Returns all module names importable via useml.workdir.* from the snapshot."""
        if not self.is_mounted or not self.mounted_path:
            raise NothingMountedError(
                "No snapshot is mounted. Use useml.mount('\\\\latest') first."
            )
        return self._scan_directory(self.mounted_path)

    def debug(self) -> None:
        """Prints a summary of __main__ imports and available useml.workdir.* imports."""
        print("=== useml Import Debug ===\n")

        print("[ __main__ & associated imports ]")
        sections = self._collect_main_imports()
        if sections:
            for origin, lines in sections:
                print(f"  # {origin}")
                seen: set = set()
                for line in lines:
                    if line not in seen:
                        print(f"    {line}")
                        seen.add(line)
        else:
            print("  (none found)")

        print()
        print("[ useml.workdir.* available imports ]")

        if not self.is_mounted:
            print("  (no snapshot mounted — use useml.mount() to enable)")
        else:
            print(f"  snapshot : {self._session._mounted_snapshot}")
            print(f"  path     : {self.mounted_path}")
            try:
                modules = self.available_modules()
                print()
                if modules:
                    for mod in modules:
                        print(f"  from useml.workdir.{mod} import ...")
                else:
                    print("  (snapshot source is empty)")
            except NothingMountedError:
                print("  (could not list modules)")

        print("\n==========================")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _scan_directory(self, base: Path, prefix: str = "") -> List[str]:
        modules: List[str] = []
        for p in sorted(base.iterdir()):
            if p.name.startswith("_") or p.name.startswith("."):
                continue
            if p.is_file() and p.suffix == ".py":
                modules.append(prefix + p.stem)
            elif p.is_dir() and (p / "__init__.py").exists():
                pkg = prefix + p.name
                modules.append(pkg)
                modules.extend(self._scan_directory(p, pkg + "."))
        return modules

    def _parse_imports(self, source: str) -> List[str]:
        lines: List[str] = []
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return lines

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    as_part = f" as {alias.asname}" if alias.asname else ""
                    lines.append(f"import {alias.name}{as_part}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                dots = "." * (node.level or 0)
                names = ", ".join(
                    alias.name + (f" as {alias.asname}" if alias.asname else "")
                    for alias in node.names
                )
                lines.append(f"from {dots}{module} import {names}")

        return lines

    def _collect_main_imports(self) -> List[Tuple[str, List[str]]]:
        """Returns [(origin_label, [import_line, ...])] for __main__ and notebooks."""
        import __main__

        results: List[Tuple[str, List[str]]] = []

        main_file = getattr(__main__, "__file__", None)
        if main_file:
            try:
                source = Path(main_file).read_text()
                lines = self._parse_imports(source)
                if lines:
                    results.append((main_file, lines))
            except OSError:
                pass

        # IPython / Jupyter notebook cells
        try:
            from IPython import get_ipython
            shell = get_ipython()
            if shell:
                cells = [c for c in shell.user_ns.get("In", []) if c.strip()]
                nb_lines: List[str] = []
                for cell in cells:
                    nb_lines.extend(self._parse_imports(cell))
                if nb_lines:
                    results.append(("<notebook>", nb_lines))
        except Exception:
            pass

        return results
