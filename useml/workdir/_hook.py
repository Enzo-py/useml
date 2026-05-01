import sys
from importlib.abc import Loader, MetaPathFinder
from importlib.util import spec_from_loader
from pathlib import Path


class WorkdirImportFinder(MetaPathFinder):
    """Intercepts ``useml.workdir.*`` imports and routes them to the mounted snapshot."""

    def find_spec(self, fullname, path, target=None):
        if fullname == "useml.workdir":
            return None
        if fullname.startswith("useml.workdir."):
            real_name = fullname[len("useml.workdir."):]
            return spec_from_loader(fullname, WorkdirLoader(real_name))
        return None


class WorkdirLoader(Loader):
    """Loads modules from the mounted snapshot into the ``useml.workdir.*`` namespace.

    Raises :exc:`~useml.errors.NothingMountedError` when no snapshot is mounted.
    """

    def __init__(self, real_module_name: str) -> None:
        self.real_module_name = real_module_name

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        from useml.session.manager import _session
        from useml.errors import NothingMountedError
        from useml.imports import _load_snapshot_module

        if not _session._mounted_snapshot or not _session._mounted_sys_path:
            raise NothingMountedError(
                f"Cannot import 'useml.workdir.{self.real_module_name}': "
                "no snapshot is mounted. "
                "Use useml.mount('\\\\latest') first."
            )

        real_module, is_package = _load_snapshot_module(
            self.real_module_name, _session._mounted_sys_path
        )

        if real_module is None:
            base = Path(_session._mounted_sys_path).joinpath(
                *self.real_module_name.split(".")
            )
            module.__path__ = [str(base)]
            module.__file__ = None
            module.__loader__ = self
            module.__package__ = module.__name__
            return

        module.__dict__.update(
            {
                k: v
                for k, v in real_module.__dict__.items()
                if not k.startswith("__") or k in ("__all__", "__version__")
            }
        )
        module.__file__ = getattr(real_module, "__file__", None)
        module.__loader__ = self
        module.__package__ = module.__name__.rpartition(".")[0]

        if is_package:
            base = Path(_session._mounted_sys_path).joinpath(
                *self.real_module_name.split(".")
            )
            module.__path__ = [str(base)]


def _install_import_hook() -> None:
    for finder in sys.meta_path:
        if isinstance(finder, WorkdirImportFinder):
            return
    sys.meta_path.insert(0, WorkdirImportFinder())
