# useml/workdir/__init__.py

import hashlib
import sys
import importlib.util
from pathlib import Path
from importlib.abc import MetaPathFinder, Loader
from importlib.util import spec_from_loader


class WorkdirImportFinder(MetaPathFinder):
    """Intercept ONLY useml.workdir.* imports."""

    def find_spec(self, fullname, path, target=None):
        # Do not touch anything except useml.workdir.*
        if fullname == "useml.workdir":
            return None

        if fullname.startswith("useml.workdir."):
            real_name = fullname[len("useml.workdir."):]
            return spec_from_loader(fullname, WorkdirLoader(real_name))

        return None


class WorkdirLoader(Loader):
    """Load modules from snapshot or current workdir WITHOUT polluting global imports."""

    def __init__(self, real_module_name):
        self.real_module_name = real_module_name

    def create_module(self, spec):
        return None  # default behavior

    def exec_module(self, module):
        from useml.session.manager import _session
        import importlib
        import importlib.util
        import sys
        from pathlib import Path

        # --- SNAPSHOT MODE ---
        if _session._mounted_snapshot and _session._mounted_sys_path:
            parts = self.real_module_name.split(".")
            root = Path(_session._mounted_sys_path)
            base = root.joinpath(*parts)

            file_candidate = base.with_suffix(".py")
            package_init = base / "__init__.py"

            # --- CASE 1: module file ---
            if file_candidate.exists():
                path = file_candidate
                is_package = False

            # --- CASE 2: classic package ---
            elif package_init.exists():
                path = package_init
                is_package = True

            # --- CASE 3: namespace package ---
            elif base.is_dir():
                module.__path__ = [str(base)]
                module.__file__ = None
                module.__loader__ = self
                module.__package__ = module.__name__
                return

            else:
                raise ImportError(f"{self.real_module_name} not found in snapshot {root}")

            # --- LOAD MODULE ---
            internal_name = f"_useml_workdir_internal.{self.real_module_name}"

            if internal_name in sys.modules:
                real_module = sys.modules[internal_name]
            else:
                spec = importlib.util.spec_from_file_location(
                    internal_name,
                    path,
                    submodule_search_locations=[str(base)] if is_package else None
                )
                real_module = importlib.util.module_from_spec(spec)
                sys.modules[internal_name] = real_module
                spec.loader.exec_module(real_module)

            module.__dict__.update(real_module.__dict__)
            module.__file__ = str(path)
            module.__loader__ = self
            module.__package__ = module.__name__.rpartition('.')[0]

            if is_package:
                module.__path__ = [str(base)]

        # --- CURRENT MODE ---
        else:
            import importlib.util

            spec = importlib.util.find_spec(self.real_module_name)
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot find module {self.real_module_name}")

            real_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(real_module)

            module.__dict__.update(real_module.__dict__)
            module.__file__ = getattr(real_module, "__file__", None)
            module.__loader__ = self
            module.__package__ = module.__name__.rpartition('.')[0]

            if hasattr(real_module, "__path__"):
                module.__path__ = real_module.__path__

def install_import_hook():
    """Install the import hook once."""
    for finder in sys.meta_path:
        if isinstance(finder, WorkdirImportFinder):
            return
    sys.meta_path.insert(0, WorkdirImportFinder())


# Auto-install
install_import_hook()


def __getattr__(name):
    """Allow attribute access on useml.workdir itself."""
    from useml.session.manager import _session
    
    # Expose session attributes
    if name == '_mounted_sys_path':
        return _session._mounted_sys_path
    if name == '_mounted_snapshot':
        return _session._mounted_snapshot
    
    if hasattr(_session, name):
        return getattr(_session, name)
    
    raise AttributeError(f"module 'useml.workdir' has no attribute '{name}'")

def find_module_path(module_name):
    parts = module_name.split(".")

    for p in sys.path:
        if not p:  # "" → cwd
            base = Path.cwd()
        else:
            base = Path(p)

        candidate = base.joinpath(*parts)

        file = candidate.with_suffix(".py")
        package = candidate / "__init__.py"

        if file.exists():
            return file, False
        if package.exists():
            return package, True

    return None, None
