from ._hook import _install_import_hook

_install_import_hook()


def __getattr__(name):
    """Exposes session mount state as module-level attributes."""
    from useml.session.manager import _session

    if name in ("_mounted_sys_path", "_mounted_snapshot"):
        return getattr(_session, name)
    if hasattr(_session, name):
        return getattr(_session, name)
    raise AttributeError(f"module 'useml.workdir' has no attribute {name!r}")
