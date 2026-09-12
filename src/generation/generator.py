import os
from typing import List, Generator
from google import genai
from google.genai import types
from dotenv import load_dotenv

from src.models.retrieval import RetrievedChunk
from src.retrieval.hybrid_retriever import HybridRetriever
from src.storage.cache import SemanticCache

load_dotenv()

SYSTEM_INSTRUCTION = """
You are a senior financial analyst and enterprise search assistant.
Your task is to answer the user's question using ONLY the provided reference documents.

Strict Rules:
1. Grounding: Rely strictly on the context provided. Do NOT extrapolate or assume figures not explicitly stated.
2. If the context does not contain sufficient information to answer the question with certainty, state:
   "The available financial documents do not contain sufficient information to answer this question."
3. Citations: Every statement containing numbers, dates, or factual claims MUST include an inline citation
   referencing the document source title or section, e.g., [Source: Consolidated Statements of Income].
4. Format: Structure your response cleanly using Markdown headings, bullet points, or comparison tables when reporting figures.
"""


class GroundedGenerator:
    """
    Coordinates semantic caching, hybrid retrieval, and grounded streaming generation.
    """

    def __init__(self, cache_threshold: float = 0.95):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set in environment.")

        self.client = genai.Client(api_key=api_key)
        self.model_name = "gemini-3.5-flash"
        self.retriever = HybridRetriever()
        self.cache = SemanticCache(similarity_threshold=cache_threshold)

    def _build_context_prompt(self, query: str, chunks: List[RetrievedChunk]) -> str:
        """Constructs the prompt containing resolved parent document contexts."""
        context_blocks = []
        for i, chunk in enumerate(chunks, start=1):
            source_title = chunk.parent_section_title or f"Document Section {i}"
            content = chunk.parent_content or chunk.content
            context_blocks.append(
                f"### [Reference {i}: {source_title}]\n{content}\n"
            )

        joined_context = "\n---\n".join(context_blocks)
        prompt = f"""Use the following reference documents to answer the user query.

{joined_context}

---
User Query: {query}
Answer:"""
        return prompt

    def generate_stream(self, query: str, top_k: int = 3) -> Generator[str, None, None]:
        """
        Streams answer tokens. Checks semantic cache first; on miss, retrieves
        parent documents and streams from Gemini while populating the cache.
        """
        # 1. Embed query to check cache
        query_vector = self.retriever.vector_store.embed_query(query)

        # 2. Check Semantic Cache
        cached_result = self.cache.get(query_vector)
        if cached_result:
            cached_text, sim_score = cached_result
            yield f"[CACHE HIT: Cosine Similarity {sim_score:.4f}]\n\n"
            yield cached_text
            return

        # 3. Cache Miss: Execute Hybrid Retrieval & Re-ranking
        retrieved_chunks = self.retriever.retrieve(query=query, top_k=top_k)
        if not retrieved_chunks:
            yield "No relevant financial documents found matching your query."
            return

        # 4. Construct Prompt
        full_prompt = self._build_context_prompt(query, retrieved_chunks)

        # 5. Stream from Gemini 2.5 Flash
        response_stream = self.client.models.generate_content_stream(
            model=self.model_name,
            contents=full_prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.1,  # Low temperature for deterministic factual extraction
            )
        )

        accumulated_text = []
        for chunk in response_stream:
            if chunk.text:
                accumulated_text.append(chunk.text)
                yield chunk.text

        # 6. Store completed response in Semantic Cache
        complete_response = "".join(accumulated_text)
        if complete_response.strip():
            self.cache.set(query, query_vector, complete_response)