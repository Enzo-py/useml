# tests/test_vault.py
"""Tests for vault snapshots, source archiving, and mount isolation."""

import sys
import os
import torch
import pytest
import useml
from pathlib import Path
from conftest import ModuleManager, FileManager


# ============================================================================
# STANDALONE TESTS
# ============================================================================

def test_session_mount_isolation(isolated_project):
    """Test that mount() isolates code versions without polluting global imports."""
    project_dir = isolated_project["project_dir"]
    vault_dir = isolated_project["vault_dir"]

    useml.init(vault_path=vault_dir)
    useml.new("iso_test")

    module_path = project_dir / "net.py"

    # --- VERSION 1 ---
    FileManager.write_simple_model(module_path, "v1")
    net = ModuleManager.reload_fresh("net", project_dir)

    assert net.Model.VERSION == "v1"

    model = net.Model()
    useml.track("m", model)
    model.w.data.fill_(1.0)
    useml.commit("v1")

    # --- VERSION 2 ---
    FileManager.write_simple_model(module_path, "v2")
    net = ModuleManager.reload_fresh("net", project_dir)

    assert net.Model.VERSION == "v2", \
        f"Expected v2 after reload, got {net.Model.VERSION}"

    model = net.Model()
    model.w.data.fill_(2.0)
    useml.track("m", model)
    useml.commit("v2")

    # TEST: Load current (latest = v2)
    m_current = useml.load("m")
    assert m_current.w.item() == 2.0
    assert net.Model.VERSION == "v2"

    # TEST: Mount v1 snapshot via useml.workdir
    useml.mount("\\latest")

    from useml.workdir import net as net_mounted
    assert net_mounted.Model.VERSION == "v1", \
        f"Expected v1 from mounted snapshot, got {net_mounted.Model.VERSION}"

    # Regular import should still be v2 (unaffected by mount)
    assert net.Model.VERSION == "v2"

    # Cleanup
    ModuleManager.clear("net", "useml.workdir.net")


# ============================================================================
# VAULT RUNTIME TESTS
# ============================================================================

class TestVaultRuntime:
    """Test vault snapshot creation, source archiving, and code reloading."""

    def test_snapshot_vs_current_code(self, project_env):
        """Test that code changes are reflected after reload."""
        model_file = project_env / "mymodel.py"

        # V1
        FileManager.write_model(model_file, "v1")
        mymodel = ModuleManager.reload_fresh("mymodel", project_env)
        assert mymodel.MyModel.VERSION == "v1"

        useml.init(project_env.parent / "vault")
        useml.new("p")
        useml.track("m", mymodel.MyModel())
        snap_v1 = useml.commit("v1")

        # V2
        FileManager.write_model(model_file, "v2")
        mymodel = ModuleManager.reload_fresh("mymodel", project_env)

        assert mymodel.MyModel.VERSION == "v2", \
            f"Expected v2 after file change + reload, got {mymodel.MyModel.VERSION}"

        # Snapshot v1 metadata should exist
        assert snap_v1.components["m"]["code_hash"] is not None

        # Cleanup
        ModuleManager.clear("mymodel")

    def test_mount_switching_source(self, project_env):
        """Test that mount() switches the code loaded via useml.workdir."""
        model_file = project_env / "mymodel.py"

        # V1: commit
        FileManager.write_model(model_file, "v1")
        mymodel = ModuleManager.reload_fresh("mymodel", project_env)

        useml.init(project_env.parent / "vault")
        useml.new("p", auto_focus=True)
        useml.track("m", mymodel.MyModel())
        snap = useml.commit("v1")

        # V2: update code
        FileManager.write_model(model_file, "v2")
        mymodel = ModuleManager.reload_fresh("mymodel", project_env)

        # Mount and verify source was archived in snapshot
        useml.mount("\\latest")

        source_dir = snap.path / "source"
        assert source_dir.exists(), f"Source dir not created: {source_dir}"

        files = list(source_dir.rglob("mymodel.py"))
        assert len(files) >= 1, \
            f"mymodel.py not found in snapshot source.\n" \
            f"Contents: {[f.relative_to(source_dir) for f in source_dir.rglob('*')]}"

        # Verify mounted code is v1
        from useml.workdir import mymodel as mymodel_mounted
        assert mymodel_mounted.MyModel.VERSION == "v1", \
            f"Expected v1 from snapshot, got {mymodel_mounted.MyModel.VERSION}"

        # Cleanup
        ModuleManager.clear("mymodel", "useml.workdir.mymodel")

    def test_source_code_archived(self, project_env):
        """Test that source code is properly archived in the snapshot."""
        model_file = project_env / "mymodel.py"

        FileManager.write_model(model_file, "v1")
        mymodel = ModuleManager.reload_fresh("mymodel", project_env)

        useml.init(project_env.parent / "vault")
        useml.new("p", auto_focus=True)
        useml.track("m", mymodel.MyModel())
        snap = useml.commit("archive")

        source_dir = snap.path / "source"
        assert source_dir.exists(), \
            f"Source directory was not created: {source_dir}"

        files = list(source_dir.rglob("mymodel.py"))
        all_files = [f.relative_to(source_dir) for f in source_dir.rglob("*")]

        assert len(files) >= 1, \
            f"Expected mymodel.py in snapshot source.\n" \
            f"Contents: {all_files}"

        # Cleanup
        ModuleManager.clear("mymodel")

    def test_load_weights_with_code_change(self, project_env):
        """Test loading weights when source code has changed."""
        model_file = project_env / "mymodel.py"

        # V1: commit
        FileManager.write_model(model_file, "v1")
        mymodel = ModuleManager.reload_fresh("mymodel", project_env)

        useml.init(project_env.parent / "vault")
        useml.new("p")
        useml.track("m", mymodel.MyModel())
        useml.commit("v1")

        # V2: change code
        FileManager.write_model(model_file, "v2")
        mymodel = ModuleManager.reload_fresh("mymodel", project_env)

        # Load with code change warning (should not raise)
        m_loaded = useml.load("m", _from="\\latest")
        assert isinstance(m_loaded, mymodel.MyModel)

        # Cleanup
        ModuleManager.clear("mymodel")

    def test_no_sys_modules_pollution(self, project_env):
        """Test that mount() does not pollute global sys.modules."""
        model_file = project_env / "mymodel.py"

        FileManager.write_model(model_file, "v1")
        mymodel = ModuleManager.reload_fresh("mymodel", project_env)

        useml.init(project_env.parent / "vault")
        useml.new("p", auto_focus=True)
        useml.track("m", mymodel.MyModel())
        useml.commit("v1")

        modules_before = set(sys.modules.keys())
        useml.mount("\\latest")
        modules_after = set(sys.modules.keys())

        # Only useml.workdir.* modules should have been added
        new_modules = modules_after - modules_before
        polluting = [m for m in new_modules if not m.startswith("useml.workdir")]

        assert not polluting, \
            f"mount() added unexpected modules to sys.modules: {polluting}"

        # Cleanup
        ModuleManager.clear("mymodel")


# ============================================================================
# IMPORT HOOK TESTS
# ============================================================================

class TestImportHook:
    """Test the useml.workdir import hook behavior."""

    def test_import_hook_scope(self, project_env):
        """Test that normal imports are not affected by the workdir hook."""
        import importlib.util

        model_file = project_env / "mymodel.py"
        FileManager.write_model(model_file, "v1")

        # Module must be findable via sys.path
        spec = importlib.util.find_spec("mymodel")
        assert spec is not None, \
            "mymodel should be findable via sys.path"

        mymodel = ModuleManager.reload_fresh("mymodel", project_env)
        assert mymodel.MyModel.VERSION == "v1"

        # useml.workdir hook should not affect direct imports
        assert "useml.workdir.mymodel" not in sys.modules

        # Cleanup
        ModuleManager.clear("mymodel")
