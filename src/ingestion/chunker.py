import re
from typing import List, Tuple
from src.models.schemas import (
    ParentDocument,
    ChildChunk,
    ChunkType,
    DocumentMetadata
)


class HierarchicalChunker:
    """
    Transforms parsed Markdown documents into linked ParentDocuments and ChildChunks.
    """

    def __init__(
        self,
        child_word_limit: int = 200,
        child_word_overlap: int = 30,
        parent_word_limit: int = 800
    ):
        self.child_word_limit = child_word_limit
        self.child_word_overlap = child_word_overlap
        self.parent_word_limit = parent_word_limit

    def _split_into_child_chunks(
        self, 
        text: str, 
        parent_id: str, 
        chunk_type: ChunkType
    ) -> List[ChildChunk]:
        words = text.split()
        if not words:
            return []

        chunks: List[ChildChunk] = []
        step = self.child_word_limit - self.child_word_overlap
        if step <= 0:
            step = self.child_word_limit

        for i in range(0, len(words), step):
            slice_words = words[i:i + self.child_word_limit]
            chunk_text = " ".join(slice_words)
            chunks.append(
                ChildChunk(
                    parent_id=parent_id,
                    content=chunk_text,
                    chunk_type=chunk_type,
                    token_count=len(slice_words)
                )
            )
        return chunks

    def chunk_document(
        self,
        parsed_data: dict,
        company_name: str,
        fiscal_year: str = None
    ) -> Tuple[List[ParentDocument], List[ChildChunk]]:
        parents: List[ParentDocument] = []
        children: List[ChildChunk] = []

        filename = parsed_data.get("filename", "unknown.pdf")
        metadata = DocumentMetadata(
            source_filename=filename,
            company_name=company_name,
            fiscal_year=fiscal_year
        )

        # 1. Process Extracted Tables as discrete, unbroken Parent units
        for tbl in parsed_data.get("tables", []):
            table_md = tbl.get("markdown", "").strip()
            if not table_md:
                continue

            parent_table = ParentDocument(
                content=table_md,
                metadata=metadata,
                section_title=f"Financial Table {tbl.get('table_index', 0) + 1}",
                token_count=len(table_md.split())
            )
            parents.append(parent_table)

            # Child chunk from table for vector matching
            table_children = self._split_into_child_chunks(
                text=table_md,
                parent_id=parent_table.id,
                chunk_type=ChunkType.TABLE
            )
            children.extend(table_children)

        # 2. Process Narrative Prose by Markdown Headings
        full_md = parsed_data.get("markdown_full", "")
        sections = re.split(r'(^#{1,3}\s+.+$)', full_md, flags=re.MULTILINE)

        current_heading = "Executive Summary / Overview"
        prose_accumulator: List[str] = []

        for part in sections:
            part_stripped = part.strip()
            if not part_stripped:
                continue

            if part_stripped.startswith("#"):
                # Flush previous accumulated section
                if prose_accumulator:
                    section_content = "\n\n".join(prose_accumulator).strip()
                    if section_content:
                        parent_prose = ParentDocument(
                            content=section_content,
                            metadata=metadata,
                            section_title=current_heading,
                            token_count=len(section_content.split())
                        )
                        parents.append(parent_prose)
                        children.extend(
                            self._split_into_child_chunks(
                                text=section_content,
                                parent_id=parent_prose.id,
                                chunk_type=ChunkType.PROSE
                            )
                        )
                    prose_accumulator = []
                current_heading = part_stripped.lstrip("#").strip()
            else:
                prose_accumulator.append(part_stripped)

        # Flush any trailing section
        if prose_accumulator:
            section_content = "\n\n".join(prose_accumulator).strip()
            if section_content:
                parent_prose = ParentDocument(
                    content=section_content,
                    metadata=metadata,
                    section_title=current_heading,
                    token_count=len(section_content.split())
                )
                parents.append(parent_prose)
                children.extend(
                    self._split_into_child_chunks(
                        text=section_content,
                        parent_id=parent_prose.id,
                        chunk_type=ChunkType.PROSE
                    )
                )

        return parents, children