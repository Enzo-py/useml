import pytest
import torch
from pathlib import Path

from useml.vault import Vault
from useml.vault.snapshot import SnapshotOverwriteError
from useml.session import Component

# --- HELPERS ---

def create_dummy_component(name="model", model=None):
    """Helper to create a valid Component for testing."""
    if model is None:
        model = torch.nn.Linear(10, 1)
    return Component(name=name, model=model, config={"lr": 0.01})

# --- TEST CLASSES ---

class TestVaultHierarchy:
    """Tests the navigation and relationship between Vault and Project levels."""

    def test_project_initialization(self, tmp_path):
        """Verifies that projects are correctly instantiated and tracked."""
        vault = Vault(path=tmp_path)
        project = vault.get_project("mnist_v1")
        
        assert project.path.name == "mnist_v1"
        assert project.path.exists()
        assert "mnist_v1" in vault.projects()

    def test_multi_project_isolation(self, tmp_path):
        """Ensures that multiple projects coexist independently."""
        vault = Vault(path=tmp_path)
        vault.get_project("cv_task")
        vault.get_project("nlp_task")
        
        projects = vault.projects()
        assert len(projects) == 2
        assert "cv_task" in projects
        assert "nlp_task" in projects


class TestProjectOperations:
    """Tests the lifecycle of snapshots including persistence and constraints."""

    def test_commit_persistence_lifecycle(self, tmp_path):
        """Validates data persistence across Vault re-initializations."""
        v_path = tmp_path / "storage"
        
        # Session 1: Commit
        v1 = Vault(path=v_path)
        proj1 = v1.get_project("p1")
        comp = create_dummy_component("main")
        
        proj1.commit(message="Initial", components={"main": comp}, accuracy=0.95)

        # Session 2: Re-load
        v2 = Vault(path=v_path)
        proj2 = v2.get_project("p1")
        
        assert len(proj2) == 1
        snapshot = proj2[0]
        # On suppose que Snapshot.__getitem__ accède au manifest/metadata
        assert snapshot.manifest["message"] == "Initial"
        assert snapshot.user_metadata["accuracy"] == 0.95

    def test_chronological_ordering(self, tmp_path):
        """Ensures log() returns snapshots in descending chronological order."""
        vault = Vault(path=tmp_path)
        proj = vault.get_project("chrono")
        comp = create_dummy_component()
        
        proj.commit("First", {"m": comp}, step=1)
        proj.commit("Second", {"m": comp}, step=2)
        
        history = proj.log()
        assert history[0].user_metadata["step"] == 2
        assert history[1].user_metadata["step"] == 1


class TestSnapshotIntegrity:
    """Tests technical validity and security of stored artifacts."""

    def test_multi_component_fidelity(self, tmp_path):
        """Verifies restoration for multiple named components."""
        vault = Vault(path=tmp_path)
        proj = vault.get_project("pipeline")
        
        m1, m2 = torch.nn.Linear(2, 2), torch.nn.Linear(3, 3)
        with torch.no_grad():
            m1.weight.fill_(1.0)
            m2.weight.fill_(2.0)
            
        comps = {
            "enc": create_dummy_component("enc", m1),
            "dec": create_dummy_component("dec", m2)
        }
        
        proj.commit("Multi-save", comps)
        
        # Restoration
        new_m1, new_m2 = torch.nn.Linear(2, 2), torch.nn.Linear(3, 3)
        snap = proj[0]
        
        snap.load_component(Component("enc", new_m1))
        snap.load_component(Component("dec", new_m2))
        
        assert torch.all(new_m1.weight == 1.0)
        assert torch.all(new_m2.weight == 2.0)

    def test_overwrite_protection(self, tmp_path):
        """Checks that SnapshotOverwriteError is raised when saving to a non-empty dir."""
        vault = Vault(path=tmp_path)
        proj = vault.get_project("security")
        comp = create_dummy_component()
        
        # On crée un snapshot
        snap = proj.commit("First", {"m": comp})
        
        # On tente de re-save manuellement sur le même chemin
        with pytest.raises(SnapshotOverwriteError):
            snap.save(components={"m": comp}, manifest={}, metadata={})

    def test_source_code_archiving(self, tmp_path):
        """Verifies that the project source is archived and accessible."""
        vault = Vault(path=tmp_path)
        proj = vault.get_project("audit")
        
        model = torch.nn.Linear(1, 1) # <- to findback
        comp = Component("m", model) 
        
        snap = proj.commit("Code check", {"m": comp})
        
        source_dir = snap.path / "source"
        assert source_dir.exists()
        assert source_dir.is_dir()

        current_file_name = Path(__file__).name
        archived_file = source_dir / "tests" / current_file_name 
        assert archived_file.exists()
        assert "model = torch.nn.Linear(1, 1) # <- to findback" in open(archived_file).read()
