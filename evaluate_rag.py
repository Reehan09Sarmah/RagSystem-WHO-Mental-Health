import os
import pickle
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from app import HybridRetriever

CHROMA_PATH = "./Database"
CHUNKS_FILE = "./Database/chunks.pkl"

# --- TEST DATASET ---
# Format: (Question, [List of Keywords that MUST be found in the retrieved text])
TEST_QUESTIONS = [
    (
        "What is the definition of mental health?",
        ["state of mental well-being", "cope with the stresses of life"],
    ),
    (
        "What are the three types of individual interventions?",
        [
            "psychosocial",
            "leisure-based physical activity",
            "healthy lifestyle promotion",
        ],
    ),
    ("Which region is Shuang Li from?", ["Western Pacific"]),
    ("What percentage of working-age adults have a mental disorder?", ["15%"]),
    (
        "Does the guideline recommend screening programmes?",
        ["no recommendation", "unclear whether the potential benefits"],
    ),
    ("What is Recommendation 4?", ["Manager training", "support their workers"]),
    ("How many workers are in the informal economy?", ["2 billion", "two billion"]),
]


def main():
    print("🧪 Starting RAG Evaluation (Hit Rate Calculation)...")

    # 1. Setup Retrieval (Copying logic from app.py)
    embedding_function = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=embedding_function
    )
    chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": 7})

    if not os.path.exists(CHUNKS_FILE):
        print("❌ Error: chunks.pkl not found. Run build_chroma_db.py first.")
        return

    with open(CHUNKS_FILE, "rb") as f:
        chunks = pickle.load(f)

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 7

    hybrid_retriever = HybridRetriever(
        vector_ret=chroma_retriever,
        keyword_ret=bm25_retriever,
        vector_weight=0.6,
        keyword_weight=0.4,
    )

    # 2. Run Evaluation
    total_questions = len(TEST_QUESTIONS)
    hits = 0

    print(f"\nEvaluating {total_questions} test cases...\n")

    for q_idx, (question, expected_keywords) in enumerate(TEST_QUESTIONS):
        print(f"❓ Q{q_idx + 1}: {question}")

        # Retrieve docs
        results = hybrid_retriever.invoke(question)
        combined_text = " ".join([doc.page_content.lower() for doc in results])

        # Check for keywords
        # We define a "Hit" if ANY of the expected keywords are found
        # (For stricter testing, change logic to require ALL keywords)
        found_any = False
        found_keywords = []

        for keyword in expected_keywords:
            if keyword.lower() in combined_text:
                found_any = True
                found_keywords.append(keyword)

        if found_any:
            print(f"   ✅ HIT! Found: {found_keywords}")
            hits += 1
        else:
            print(f"   ❌ MISS. Expected: {expected_keywords}")
            print(f"      (Retrieved {len(results)} chunks, but keywords were missing)")

    # 3. Final Score
    score = (hits / total_questions) * 100
    print("\n" + "=" * 30)
    print(f"📊 FINAL SCORE: {score:.1f}% ({hits}/{total_questions})")
    print("=" * 30)

    if score < 80:
        print("⚠️  Suggestion: Check the 'MISS' questions. You may need to:")
        print("    1. Adjust chunk overlap.")
        print("    2. Add those specific terms to your Hybrid/BM25 index.")
        print("    3. Increase 'k' in the retriever.")


if __name__ == "__main__":
    main()
