from pathlib import Path
from typing import Dict, Any, List
from docling.document_converter import DocumentConverter


class DoclingDocumentParser:
    """
    Local layout-aware document parser using Docling.
    Converts PDFs into structured representations while preserving table markdown.
    """

    def __init__(self):
        # Initializes local Docling converter
        self.converter = DocumentConverter()

    def parse_pdf(self, file_path: str | Path) -> Dict[str, Any]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Source file not found at: {path}")

        print(f"[Parser] Processing {path.name} with Docling...")
        conversion_result = self.converter.convert(path)
        doc = conversion_result.document

        parsed_data = {
            "filename": path.name,
            "markdown_full": doc.export_to_markdown(),
            "tables": [],
            "elements": []
        }

        # Extract structured tables as Markdown
        for table_ix, table in enumerate(doc.tables):
            table_md = table.export_to_markdown()
            parsed_data["tables"].append({
                "table_index": table_ix,
                "markdown": table_md
            })

        print(f"[Parser] Successfully parsed {path.name} with {len(doc.tables)} tables extracted.")
        return parsed_data