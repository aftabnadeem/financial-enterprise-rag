import time
from src.generation.generator import GroundedGenerator


def test_generation_and_cache():
    generator = GroundedGenerator(cache_threshold=0.95)

    # First execution: CACHE MISS (runs retrieval + LLM synthesis)
    query_1 = "Summarize the revenues and operating income for Sample Company"
    print("=" * 60)
    print(f"TEST 1 (Fresh Query): '{query_1}'")
    print("=" * 60)

    start_time = time.time()
    for token in generator.generate_stream(query_1):
        print(token, end="", flush=True)
    duration_1 = time.time() - start_time
    print(f"\n\n[Total Execution Time: {duration_1:.2f}s]\n")

    # Second execution: Semantically equivalent query -> EXPECT CACHE HIT
    query_2 = "Give me a summary of Sample Company's revenue and operating income"
    print("=" * 60)
    print(f"TEST 2 (Semantically Equivalent): '{query_2}'")
    print("=" * 60)

    start_time = time.time()
    for token in generator.generate_stream(query_2):
        print(token, end="", flush=True)
    duration_2 = time.time() - start_time
    print(f"\n\n[Total Execution Time: {duration_2:.2f}s]\n")


if __name__ == "__main__":
    test_generation_and_cache()