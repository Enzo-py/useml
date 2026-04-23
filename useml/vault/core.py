from pathlib import Path
from typing import List, Union
from .project import Project

class Vault:
    """
    The central management system for machine learning storage.
    
    The Vault acts as the root directory for all experiments, organizing data into 
    individual Projects. It handles the high-level filesystem structure and 
    provides access to project-specific tracking.
    """

    def __init__(self, path: Union[str, Path] = "vault"):
        """
        Initializes the Vault at the specified filesystem location.

        Args:
            path (Union[str, Path]): The root directory for the vault. 
                Defaults to "vault".
        """
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

    def get_project(self, name: str) -> Project:
        """
        Retrieves an existing project or initializes a new one.

        Args:
            name (str): The unique name of the project. This will correspond 
                to a subdirectory within the vault.

        Returns:
            Project: An initialized Project instance tied to the specified name.
        """
        project_path = self.path / name
        return Project(project_path)

    def list_projects(self) -> List[str]:
        """
        Scans the vault directory and lists all identified projects.

        A directory is considered a project if it exists directly within 
        the vault's root path.

        Returns:
            List[str]: A list of project names (directory names).
        """
        return [d.name for d in self.path.iterdir() if d.is_dir()]

    def __repr__(self) -> str:
        """Returns a string representation of the Vault instance."""
        return f"<useml.Vault path='{self.path.absolute()}' projects={len(self.list_projects())}>"
