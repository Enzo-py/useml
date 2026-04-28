# tests/conftest.py
"""Pytest configuration and shared fixtures."""

import sys
import importlib
import importlib.util
import shutil
from pathlib import Path
import pytest


# ============================================================================
# MODULE UTILITIES
# ============================================================================

class ModuleManager:
    """Manage module caching and reloading for tests."""

    @staticmethod
    def clear(*module_names):
        """Clear modules and their __pycache__ from sys.modules."""
        to_delete = [
            key for key in list(sys.modules.keys())
            for mod_name in module_names
            if key == mod_name or key.startswith(mod_name + ".")
        ]
        for key in to_delete:
            del sys.modules[key]

    @staticmethod
    def clear_pycache(directory: Path, *module_names):
        """Remove __pycache__ entries to prevent stale bytecode."""
        pycache = directory / "__pycache__"
        if not pycache.exists():
            return
        for mod_name in module_names:
            for pyc in pycache.glob(f"{mod_name}*.pyc"):
                pyc.unlink(missing_ok=True)

    @staticmethod
    def reload_fresh(module_name: str, search_path: Path = None):
        """
        Reload a module bypassing all caches (sys.modules + bytecode).

        Must be called AFTER the source file has been modified on disk.

        Args:
            module_name: Name of the module to reload.
            search_path: Directory containing the module (for pycache clearing).

        Returns:
            The freshly loaded module.
        """
        # 1. Clear bytecode cache to force recompilation from source
        if search_path:
            ModuleManager.clear_pycache(search_path, module_name)

        # 2. Clear sys.modules
        ModuleManager.clear(module_name)

        # 3. Re-find the spec (now picks up source, not bytecode)
        spec = importlib.util.find_spec(module_name)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot find module '{module_name}' in sys.path")

        # 4. Create and execute module fresh from source
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        return module


# ============================================================================
# FILE UTILITIES
# ============================================================================

class FileManager:
    """Manage test file creation."""

    @staticmethod
    def write_model(model_file: Path, version: str, class_name: str = "MyModel"):
        """Write a standard MyModel test file."""
        model_file.parent.mkdir(parents=True, exist_ok=True)
        model_file.write_text(f'''\
import torch

class {class_name}(torch.nn.Module):
    VERSION = "{version}"

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.ones(1))

    def forward(self, x):
        return x * self.w
''')

    @staticmethod
    def write_simple_model(model_file: Path, version: str):
        """Write a minimal Model test file."""
        model_file.parent.mkdir(parents=True, exist_ok=True)
        model_file.write_text(f'''\
import torch

class Model(torch.nn.Module):
    VERSION = '{version}'

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor([0.0]))

    def forward(self, x):
        return x * self.w
''')


# ============================================================================
# PYTEST FIXTURES
# ============================================================================

@pytest.fixture(autouse=True)
def reset_useml_session():
    """
    Reset the useml _session singleton before and after every test.

    This is CRITICAL: _session is a module-level singleton that persists
    across tests and causes state leakage between them.
    """
    from useml.session.manager import _session
    _session.__init__()
    yield
    _session.__init__()


@pytest.fixture
def project_env(tmp_path):
    """
    Create a temporary project directory added to sys.path.

    Yields:
        Path to the project directory.
    """
    project_dir = tmp_path / "proj"
    project_dir.mkdir()

    sys.path.insert(0, str(project_dir))
    yield project_dir

    if str(project_dir) in sys.path:
        sys.path.remove(str(project_dir))


@pytest.fixture
def isolated_project(tmp_path):
    """
    Create an isolated project with separate app and vault directories.

    Yields:
        dict with 'project_dir' and 'vault_dir' Path objects.
    """
    import os
    project_dir = tmp_path / "app"
    project_dir.mkdir()
    vault_dir = tmp_path / "vault"

    original_cwd = Path.cwd()
    os.chdir(project_dir)
    sys.path.insert(0, str(project_dir))

    yield {
        "project_dir": project_dir,
        "vault_dir": vault_dir,
    }

    os.chdir(original_cwd)
    if str(project_dir) in sys.path:
        sys.path.remove(str(project_dir))
