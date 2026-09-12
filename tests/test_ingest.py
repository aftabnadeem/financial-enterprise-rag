from pathlib import Path
from src.ingestion.parser import DoclingDocumentParser
from src.ingestion.chunker import HierarchicalChunker


def run_smoke_test():
    sample_path = Path("data/raw/sample.pdf")
    if not sample_path.exists():
        print(f"Please place a sample PDF at {sample_path} first.")
        return

    parser = DoclingDocumentParser()
    chunker = HierarchicalChunker()

    parsed = parser.parse_pdf(sample_path)
    parents, children = chunker.chunk_document(
        parsed_data=parsed,
        company_name="Alphabet Inc.",
        fiscal_year="2024"
    )

    print("\n--- INGESTION RESULTS ---")
    print(f"Total Parent Documents created: {len(parents)}")
    print(f"Total Child Chunks generated:    {len(children)}")
    
    if parents:
        print("\n--- SAMPLE PARENT (First item) ---")
        print(f"Section: {parents[0].section_title}")
        print(f"Type: {type(parents[0])}")
        print(f"Content Preview:\n{parents[0].content[:200]}...\n")

    if children:
        print("--- SAMPLE CHILD (First item) ---")
        print(f"Parent ID: {children[0].parent_id}")
        print(f"Chunk Type: {children[0].chunk_type}")
        print(f"Content Preview:\n{children[0].content[:150]}...\n")


if __name__ == "__main__":
    run_smoke_test()