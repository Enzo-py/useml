import pytest
import torch
import shutil
from pathlib import Path
from useml import Vault

# --- FIXTURES ---

@pytest.fixture
def clean_vault_path():
    """
    Ensures a clean testing environment by creating a temporary 
    directory before each test and removing it afterward.
    """
    path = Path("test_vault_storage")
    if path.exists():
        shutil.rmtree(path)
    yield path
    if path.exists():
        shutil.rmtree(path)

# --- TEST CLASSES ---

class TestVaultHierarchy:
    """Tests the navigation and relationship between Vault and Project levels."""

    def test_project_initialization(self, clean_vault_path):
        """Verifies that projects are correctly instantiated and tracked by the Vault."""
        vault = Vault(path=clean_vault_path)
        project = vault.get_project("mnist_v1")
        
        assert project.path.name == "mnist_v1"
        assert project.path.exists()
        assert "mnist_v1" in vault.list_projects()

    def test_multi_project_isolation(self, clean_vault_path):
        """Ensures that multiple projects can coexist independently within the same Vault."""
        vault = Vault(path=clean_vault_path)
        vault.get_project("computer_vision")
        vault.get_project("nlp_task")
        
        projects = vault.list_projects()
        assert len(projects) == 2
        assert "computer_vision" in projects
        assert "nlp_task" in projects


class TestProjectOperations:
    """Tests the lifecycle of snapshots within a project, including persistence and ordering."""

    def test_commit_persistence_lifecycle(self, clean_vault_path):
        """
        Validates that data remains persistent after the Vault object is destroyed 
        and re-initialized.
        """
        # Session 1: Commit data
        v1 = Vault(path=clean_vault_path)
        proj1 = v1.get_project("persistence_study")
        model = torch.nn.Linear(10, 1)
        
        proj1.commit(model, "Stable baseline", accuracy=0.98, stage="production")
        del v1, proj1 # Simulate program exit

        # Session 2: Re-load and Verify
        v2 = Vault(path=clean_vault_path)
        proj2 = v2.get_project("persistence_study")
        
        assert len(proj2) == 1
        snapshot = proj2[0]
        assert snapshot["message"] == "Stable baseline"
        assert snapshot["accuracy"] == 0.98
        assert snapshot["stage"] == "production"

    def test_chronological_ordering(self, clean_vault_path):
        """Ensures that .log() returns snapshots in descending chronological order (newest first)."""
        vault = Vault(path=clean_vault_path)
        proj = vault.get_project("chronology_test")
        model = torch.nn.Linear(2, 2)
        
        # Commit sequence
        proj.commit(model, "First Commit", version=1)
        proj.commit(model, "Second Commit", version=2)
        proj.commit(model, "Third Commit", version=3)
        
        history = proj.log()
        assert len(history) == 3
        assert history[0]["version"] == 3  # Latest
        assert history[-1]["version"] == 1 # Oldest

    def test_empty_project_constraints(self, clean_vault_path):
        """Checks expected failure modes when interacting with an empty project."""
        vault = Vault(path=clean_vault_path)
        proj = vault.get_project("empty_repo")
        
        assert len(proj) == 0
        with pytest.raises(IndexError):
            _ = proj[0]


class TestSnapshotIntegrity:
    """Tests the technical validity of stored artifacts (PyTorch weights and JSON metadata)."""

    def test_weight_restoration_fidelity(self, clean_vault_path):
        """Verifies that loaded weights are numerically identical to saved weights."""
        vault = Vault(path=clean_vault_path)
        proj = vault.get_project("integrity_check")
        model = torch.nn.Linear(5, 5)
        
        # Inject known values
        with torch.no_grad():
            model.weight.fill_(42.0)
            model.bias.fill_(0.0)
        
        proj.commit(model, "Checkpoint 42")
        
        # Load into a fresh model instance
        new_model = torch.nn.Linear(5, 5)
        proj[0].load_weights(new_model)
        
        assert torch.all(new_model.weight == 42.0)
        assert torch.all(new_model.bias == 0.0)

    def test_manifest_integrity(self, clean_vault_path):
        """Validates that commit messages are stored in the manifest, not the folder name."""
        vault = Vault(path=clean_vault_path)
        proj = vault.get_project("formatting")
        model = torch.nn.Linear(2, 2)
        
        msg = "Initial Research Prototype"
        snap = proj.commit(model, msg)
        
        # 1. The folder name should be a clean timestamped ID
        assert "snap_" in snap.path.name
        # 2. The REAL message must be retrieved via __getitem__ (from manifest.yaml)
        assert snap["message"] == msg
