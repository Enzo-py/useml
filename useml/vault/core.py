from pathlib import Path
from typing import List, Union

from .project import Project


class Vault:
    """Root storage directory that organises experiments into projects.

    Each project corresponds to a subdirectory inside the vault. A hidden
    `.useml_vault` marker file is created at initialisation so the codebase
    can identify vault roots and avoid accidentally archiving them.
    """

    def __init__(self, path: Union[str, Path] = "vault") -> None:
        """Initialises the vault at the given filesystem location.

        Args:
            path: Root directory for the vault. Created if absent.
        """
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / ".useml_vault").touch(exist_ok=True)

    def exists(self, project_name: str) -> bool:
        """Returns True if a project directory already exists in the vault.

        Args:
            project_name: Name of the project to look up.

        Returns:
            True if the project directory is present, False otherwise.
        """
        return (self.path / project_name).is_dir()

    def get_project(self, name: str) -> Project:
        """Returns a Project instance for the given name, creating it if needed.

        Args:
            name: Unique project identifier, used as the subdirectory name.

        Returns:
            An initialised Project bound to ``vault/<name>``.
        """
        return Project(self.path / name)

    def projects(self) -> List[Project]:
        """Lists all projects found inside the vault.

        Returns:
            All Project instances whose directory does not start with ``"."``.
        """
        if not self.path.exists():
            return []
        return [
            self.get_project(d.name)
            for d in self.path.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ]

    def __repr__(self) -> str:
        return (
            f"<useml.Vault path='{self.path.absolute()}' "
            f"projects={len(self.projects())}>"
        )
