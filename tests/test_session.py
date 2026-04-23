import importlib
import sys
import pytest
import torch
import useml
from useml.session.manager import _session, UncommittedSessionError

class DummyModel(torch.nn.Module):
    def __init__(self, val=1.0):
        super().__init__()
        self.param = torch.nn.Parameter(torch.ones(1) * val)

@pytest.fixture(autouse=True)
def reset_session():
    """Wipes the global singleton, including the stash, for each test."""
    _session.vault = None
    _session._project = None
    _session.components = {}
    _session._is_dirty = False
    _session._stash = {} # Ensure stash is cleared
    yield

# --- DIRTY STATE & PROTECTION TESTS ---

def test_dirty_state_tracking(tmp_path):
    """Verifies that tracking a component marks the session as dirty."""
    useml.init(vault_path=tmp_path)
    useml.new("proj_a")
    assert _session._is_dirty is False
    
    useml.track("model", DummyModel())
    assert _session._is_dirty is True
    
    useml.commit("First save")
    assert _session._is_dirty is False

def test_error_uncommitted_switch(tmp_path):
    """Verifies that switching focus without committing or stashing raises an error."""
    useml.init(vault_path=tmp_path)
    useml.new("proj_a")
    useml.track("m1", DummyModel())
    
    # Attempting to switch should fail
    with pytest.raises(UncommittedSessionError, match="unsaved changes"):
        useml.focus("proj_b")
    
    # Verification: we are still on proj_a and dirty
    assert _session.project.path.name == "proj_a"
    assert "m1" in _session.components

def test_force_focus_discards_changes(tmp_path):
    """Verifies that focus(force=True) bypasses the dirty check and wipes RAM."""
    useml.init(vault_path=tmp_path)
    useml.new("proj_a")
    useml.track("m1", DummyModel())
    
    # Forced switch
    useml.focus("proj_b", force=True)
    
    assert _session.project.path.name == "proj_b"
    assert len(_session.components) == 0  # RAM wiped
    assert _session._is_dirty is False

# --- STASH LOGIC TESTS ---

def test_stash_and_restore_flow(tmp_path):
    """Tests the full stash cycle: track A -> stash -> work B -> focus A (restore)."""
    useml.init(vault_path=tmp_path)
    
    # 1. Start Project A
    useml.new("proj_a")
    model_a = DummyModel(1.0)
    useml.track("net_a", model_a)
    assert _session._is_dirty is True
    
    # 2. Stash A
    useml.stash()
    assert _session._project is None
    assert len(_session.components) == 0
    assert "proj_a" in _session._stash
    
    # 3. Work on Project B
    useml.new("proj_b")
    useml.track("net_b", DummyModel(2.0))
    useml.commit("Save B")
    
    # 4. Return to Project A (should auto-restore from stash)
    useml.focus("proj_a")
    assert _session.project.path.name == "proj_a"
    assert "net_a" in _session.components
    assert _session._is_dirty is True  # Still dirty because we never committed A
    assert _session.components["net_a"].model.param.item() == 1.0

def test_stash_overwrites_on_re_stash(tmp_path):
    """Verifies that stashing the same project again updates the stashed state."""
    useml.init(vault_path=tmp_path)
    useml.new("proj_a")
    
    useml.track("m", DummyModel(1.0))
    useml.stash()
    
    useml.focus("proj_a")
    useml.track("m_new", DummyModel(2.0)) # Added a new component
    useml.stash()
    
    assert len(_session._stash["proj_a"].components) == 2

# --- DASHBOARD UPDATES ---

def test_show_dirty_warning(tmp_path, capsys):
    """Ensures show() displays a warning when the session is dirty."""
    useml.init(vault_path=tmp_path)
    useml.new("dirty_proj")
    useml.track("m", DummyModel())
    
    useml.show()
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "unsaved components" in out

# --- REGRESSION TESTS ---

def test_focus_same_project_is_no_op(tmp_path):
    """Focusing on the current project should not trigger dirty errors or wipe RAM."""
    useml.init(vault_path=tmp_path)
    useml.new("proj_a")
    useml.track("m", DummyModel())
    
    # This should NOT raise UncommittedSessionError
    useml.focus("proj_a") 
    assert len(_session.components) == 1

def test_session_mount_isolation(tmp_path):
    """Verifies that mount() correctly isolates the snapshot's code using high-level API."""
    import os
    import sys
    
    # setup workdir
    project_dir = tmp_path / "my_working_project"
    project_dir.mkdir()
    vault_dir = tmp_path / "vault_storage"
    
    old_cwd = os.getcwd()
    os.chdir(project_dir)
    
    try:
        useml.init(vault_path=vault_dir)
        useml.new("mount_test_proj")
        
        # creation of dummy module V1
        module_path = project_dir / "ghost_module.py"
        module_path.write_text("VERSION = 'v1'")
        
        # for tracking we need a model
        useml.track("dummy", DummyModel())
        snap = useml.commit("Commit V1")
        tag_v1 = snap.id

        local_var = "test_local_var"
        
        # update dummy module to V2
        module_path.write_text("VERSION = 'v2'")
        
        # cleaning for forcing reimport
        if "ghost_module" in sys.modules:
            del sys.modules["ghost_module"]
        
        sys.path.insert(0, str(project_dir))
        import ghost_module # type: ignore
        assert ghost_module.VERSION == 'v2'
        
        with useml.mount(tag_v1):
            import ghost_module # type: ignore
            assert ghost_module.VERSION == 'v1', "Should have loaded V1 from snapshot code"
            
        assert ghost_module.VERSION == 'v2', "Should revert to current working code"
        assert local_var == "test_local_var", "Local var was erased or corrupted"

    finally:
        os.chdir(old_cwd)
        # cleaning every thing
        if str(project_dir) in sys.path:
            sys.path.remove(str(project_dir))
        if "ghost_module" in sys.modules:
            del sys.modules["ghost_module"]
