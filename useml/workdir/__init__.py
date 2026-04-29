# useml/workdir/__init__.py

import sys
from importlib.abc import MetaPathFinder, Loader
from importlib.util import spec_from_loader


class WorkdirImportFinder(MetaPathFinder):
    """Intercept ONLY useml.workdir.* imports."""

    def find_spec(self, fullname, path, target=None):
        if fullname == "useml.workdir":
            return None
        if fullname.startswith("useml.workdir."):
            real_name = fullname[len("useml.workdir."):]
            return spec_from_loader(fullname, WorkdirLoader(real_name))
        return None


class WorkdirLoader(Loader):
    """Load modules from a mounted snapshot via the useml.workdir.* namespace.

    Raises NothingMountedError when no snapshot is mounted — imports from this
    namespace are only valid inside a useml.mount() context.
    """

    def __init__(self, real_module_name: str):
        self.real_module_name = real_module_name

    def create_module(self, spec):
        return None  # default behaviour

    def exec_module(self, module):
        from useml.session.manager import _session
        from useml.imports import NothingMountedError, _load_snapshot_module
        from pathlib import Path

        if not _session._mounted_snapshot or not _session._mounted_sys_path:
            raise NothingMountedError(
                f"Cannot import 'useml.workdir.{self.real_module_name}': "
                "no snapshot is mounted.\n"
                "Use useml.mount('\\\\latest') (or another tag) first."
            )

        snapshot_source = _session._mounted_sys_path
        real_name = self.real_module_name

        real_module, is_package = _load_snapshot_module(real_name, snapshot_source)

        if real_module is None:
            # namespace package
            base = Path(snapshot_source).joinpath(*real_name.split("."))
            module.__path__ = [str(base)]
            module.__file__ = None
            module.__loader__ = self
            module.__package__ = module.__name__
            return

        module.__dict__.update({
            k: v for k, v in real_module.__dict__.items()
            if not k.startswith("__") or k in ("__all__", "__version__")
        })
        module.__file__ = getattr(real_module, "__file__", None)
        module.__loader__ = self
        module.__package__ = module.__name__.rpartition(".")[0]

        if is_package:
            base = Path(snapshot_source).joinpath(*real_name.split("."))
            module.__path__ = [str(base)]


def install_import_hook():
    """Install the WorkdirImportFinder once."""
    for finder in sys.meta_path:
        if isinstance(finder, WorkdirImportFinder):
            return
    sys.meta_path.insert(0, WorkdirImportFinder())


# Auto-install on import of useml.workdir
install_import_hook()


def __getattr__(name):
    """Expose session mount state as attributes on the module."""
    from useml.session.manager import _session

    if name == "_mounted_sys_path":
        return _session._mounted_sys_path
    if name == "_mounted_snapshot":
        return _session._mounted_snapshot

    if hasattr(_session, name):
        return getattr(_session, name)

    raise AttributeError(f"module 'useml.workdir' has no attribute {name!r}")
