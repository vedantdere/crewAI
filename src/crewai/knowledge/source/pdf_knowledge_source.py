from pathlib import Path
from typing import Dict, List, Union
from crewai.knowledge.source.base_file_knowledge_source import BaseFileKnowledgeSource


class PDFKnowledgeSource(BaseFileKnowledgeSource):
    """A knowledge source that stores and queries PDF file content using embeddings."""

    def load_content(self) -> Dict[Path, str]:
        """Load and preprocess PDF file content."""
        pdfplumber = self._import_pdfplumber()

        content = {}

        for path in self.safe_file_paths:
            text = ""
            path = self.convert_to_path(path)
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            content[path] = text

        self.content = content  # Store loaded content for use in `add`
        return content

    def _import_pdfplumber(self):
        """Dynamically import pdfplumber."""
        try:
            import pdfplumber
            return pdfplumber
        except ImportError:
            raise ImportError(
                "pdfplumber is not installed. Please install it with: pip install pdfplumber"
            )

    def add(self) -> None:
        """
        Add PDF file content to the knowledge source, chunk it, compute embeddings,
        and save the embeddings.
        """
        if not hasattr(self, "content"):
            self.load_content()  # Load content if not already done

        for _, text in self.content.items():
            new_chunks = self._chunk_text(text)
            self.chunks.extend(new_chunks)

        self._save_documents()

    def _chunk_text(self, text: str) -> List[str]:
        """Utility method to split text into chunks."""
        return [
            text[i: i + self.chunk_size]
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap)
        ]

    def set_files(self, files: Union[List[Union[str, Path]], str, Path]) -> None:
        """
        Sets multiple PDF files from a list of paths or a single folder path.
        """
        paths = []
        if isinstance(files, (str, Path)):
            path_obj = Path(files)
            if path_obj.is_dir():
                paths = list(path_obj.glob("*.pdf"))
            else:
                paths = [path_obj]
        elif isinstance(files, list):
            paths = [self.convert_to_path(f) for f in files]
        else:
            raise ValueError("Invalid input type. Provide a list of paths or a folder path.")

        self.safe_file_paths = paths
