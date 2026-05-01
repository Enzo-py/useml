import ast
import importlib.util
import sys
from pathlib import Path
from typing import List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .session.manager import Session


class NothingMountedError(ImportError):
    """Raised when ``useml.workdir.*`` is accessed without a mounted snapshot."""


def _load_snapshot_module(real_name: str, snapshot_source_path: str):
    """Loads a module from a snapshot source directory.

    Temporarily inserts *snapshot_source_path* into ``sys.path`` so that
    absolute imports inside the snapshot module resolve to files within the
    snapshot rather than the global environment.

    Args:
        real_name: Dotted module name relative to the snapshot root
            (e.g. ``"models.net"``).
        snapshot_source_path: Absolute path to the snapshot ``source/``
            directory.

    Returns:
        A ``(module, is_package)`` tuple. Returns ``(None, True)`` for
        namespace packages (directory without ``__init__.py``).

    Raises:
        ImportError: If the module is not found inside the snapshot.
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
        return None, True
    else:
        raise ImportError(
            f"Module {real_name!r} not found in snapshot "
            f"{snapshot_source_path!r}."
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
    """Introspection utilities for the ``useml.workdir`` import namespace.

    Provides methods to inspect what modules are importable from the currently
    mounted snapshot and what imports are declared in ``__main__`` or notebook
    cells.
    """

    def __init__(self, session: "Session") -> None:
        self._session = session

    @property
    def is_mounted(self) -> bool:
        """True when a snapshot is currently mounted."""
        return self._session._mounted_snapshot is not None

    @property
    def mounted_path(self) -> Optional[Path]:
        """Filesystem path to the mounted snapshot's ``source/`` directory."""
        if self._session._mounted_sys_path:
            return Path(self._session._mounted_sys_path)
        return None

    def available_modules(self) -> List[str]:
        """Lists all module names importable via ``useml.workdir.*``.

        Returns:
            Sorted list of dotted module names available in the snapshot.

        Raises:
            NothingMountedError: If no snapshot is mounted.
        """
        if not self.is_mounted or not self.mounted_path:
            raise NothingMountedError(
                "No snapshot is mounted. Use useml.mount('\\\\latest') first."
            )
        return self._scan_directory(self.mounted_path)

    def debug(self) -> None:
        """Prints a summary of ``__main__`` imports and available workdir modules."""
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
                for mod in modules:
                    print(f"  from useml.workdir.{mod} import ...")
                if not modules:
                    print("  (snapshot source is empty)")
            except NothingMountedError:
                print("  (could not list modules)")

        print("\n==========================")

    def _scan_directory(self, base: Path, prefix: str = "") -> List[str]:
        modules: List[str] = []
        for p in sorted(base.iterdir()):
            if p.name.startswith(("_", ".")):
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
                    suffix = f" as {alias.asname}" if alias.asname else ""
                    lines.append(f"import {alias.name}{suffix}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                dots = "." * (node.level or 0)
                names = ", ".join(
                    a.name + (f" as {a.asname}" if a.asname else "")
                    for a in node.names
                )
                lines.append(f"from {dots}{module} import {names}")
        return lines

    def _collect_main_imports(self) -> List[Tuple[str, List[str]]]:
        import __main__

        results: List[Tuple[str, List[str]]] = []
        main_file = getattr(__main__, "__file__", None)
        if main_file:
            try:
                lines = self._parse_imports(Path(main_file).read_text())
                if lines:
                    results.append((main_file, lines))
            except OSError:
                pass

        try:
            from IPython import get_ipython
            shell = get_ipython()
            if shell:
                nb_lines: List[str] = []
                for cell in shell.user_ns.get("In", []):
                    if cell.strip():
                        nb_lines.extend(self._parse_imports(cell))
                if nb_lines:
                    results.append(("<notebook>", nb_lines))
        except Exception:
            pass

        return results
