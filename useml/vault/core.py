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
        (Path(path) / ".useml_vault").touch(exist_ok=True) # creation of a vault file gate

    def exists(self, project_name: str) -> bool:
        """Checks if a project directory exists within the vault."""
        project_path = self.path / project_name
        return project_path.is_dir()

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
    
    def projects(self) -> List[Project]:
        """Lists all projects available in the vault directory.
        
        Returns:
            List[Project]: A list of Project instances.
        """
        if not self.path.exists():
            return []
            
        return [
            self.get_project(d.name) 
            for d in self.path.iterdir() 
            if d.is_dir() and not d.name.startswith(".")
        ]

    def __repr__(self) -> str:
        """Returns a string representation of the Vault instance."""
        return f"<useml.Vault path='{self.path.absolute()}' projects={len(self.projects())}>"

