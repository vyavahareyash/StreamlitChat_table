import os
from pathlib import Path
from typing import List
import pathspec

class LocalFileIngestor:
    def __init__(self, base_path: str, extensions: List[str] = None):
        self.base_path = Path(base_path).resolve()
        self.extensions = extensions or [".py", ".js", ".ts", ".java"]
        self.ignore_spec = self._load_gitignore()

    def _load_gitignore(self):
        gitignore_path = self.base_path / ".gitignore"
        if gitignore_path.exists():
            with open(gitignore_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            return pathspec.PathSpec.from_lines("gitwildmatch", lines)
        return None

    def list_code_files(self) -> List[Path]:
        all_files = self.base_path.rglob("*")
        valid_files = []
        for f in all_files:
            if f.suffix in self.extensions and f.is_file():
                relative_path = f.relative_to(self.base_path)
                if self.ignore_spec and self.ignore_spec.match_file(str(relative_path)):
                    continue  # Skip ignored files
                valid_files.append(f)
        return valid_files

    def read_files(self) -> List[dict]:
        file_data = []
        for file_path in self.list_code_files():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    file_data.append({
                        "path": str(file_path.relative_to(self.base_path)),
                        "content": content
                    })
            except Exception as e:
                print(f"Failed to read {file_path}: {e}")
        return file_data
