import os
import shutil
import argparse
import pickle
import re
import hashlib
from typing import List
from tqdm import tqdm

from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
from langchain_experimental.text_splitter import SemanticChunker  # type: ignore
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHROMA_PATH = "./Datab  ase"
DATA_PATH = "./Knowledge"
CHUNKS_FILE = "./Database/chunks.pkl"


def main():
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: Data directory '{DATA_PATH}' not found.")
        return

    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()

    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    print("🧠 Initializing Embedding Model (Singleton)...")
    embedding_function = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    documents = load_documents()
    if not documents:
        print("❌ No documents loaded. Exiting.")
        return

    # Hierarchical Strategy
    chunks = split_documents_hierarchically(documents, embedding_function)

    chunks = enrich_metadata(chunks)
    chunks = calculate_deterministic_ids(chunks)

    save_chunks_for_bm25(chunks)
    add_to_chroma(chunks, embedding_function)


def load_documents() -> List[Document]:
    print(f"📂 Loading PDFs from '{DATA_PATH}'...")
    loader = DirectoryLoader(DATA_PATH, glob="**/*.pdf", loader_cls=PyMuPDFLoader)
    try:
        documents = loader.load()
        print(f"✅ Loaded {len(documents)} pages.")
        return documents
    except Exception as e:
        print(f"❌ Error loading documents: {e}")
        return []


def split_documents_hierarchically(
    documents: List[Document], embedding_function
) -> List[Document]:
    print("✂️  Starting Hierarchical Chunking...")

    # LEVEL 1: Structural Splitting (The "Parent" Splitter) using the highly accurate pattern
    priority_pattern = (
        r"^(?:RECOMMENDATION|KEY QUESTION)\s+\d+[A-Z]?|"
        r"^(?:FOREWORD|ACKNOWLEDGEMENTS|ABBREVIATIONS)|"
        r"^(?:EXECUTIVE SUMMARY|EXECUTIVE summary)|"
        r"^INTRODUCTION|"
        r"^BACKGROUND|"
        r"^(?:Scope of the guidelines|Guideline objectives)|"
        r"^Method:\s+How the guidelines were developed|"
        r"^RECOMMENDATIONS(?:\s+for)?|"
        r"^Recommendations\s+for\s+(?:organizational|training|individual|returning|gaining|screening)|"
        r"^(?:Evidence and rationale|Evidence-to-decision)|"
        r"^(?:Key remarks|Common implementation remarks|Implementation remarks|Subgroup remarks|Monitoring and evaluation remarks|Additional remarks)|"
        r"^(?:Research gaps|Dissemination|References|Glossary|Annex \d+)|"
        r"^(?:APPENDIX|APPENDICES)"
    )

    structural_splitter = RecursiveCharacterTextSplitter(
        separators=[priority_pattern, "\n\n", "\n", " ", ""],
        chunk_size=4000,
        chunk_overlap=0,
        length_function=len,
        is_separator_regex=True,
    )

    print("   Level 1: Identifiying major sections...")
    structural_sections = structural_splitter.split_documents(documents)
    print(f"   Found {len(structural_sections)} major structural sections.")

    # Content-Specific Splitting

    semantic_splitter = SemanticChunker(
        embedding_function,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=90,
    )

    narrative_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    )

    final_chunks = []

    print("   Level 2: Applying hybrid splitting logic...")
    for doc in tqdm(structural_sections, desc="Processing Sections", unit="section"):
        content = doc.page_content.strip()

        # Heuristic: If the section starts with Recommendation, Key Question, or major Remarks, use Semantic.
        if re.search(
            r"^(RECOMMENDATION|KEY QUESTION|Key remarks|Implementation remarks|Evidence and rationale)",
            content,
            re.IGNORECASE,
        ):
            chunks = semantic_splitter.split_documents([doc])
        else:
            chunks = narrative_splitter.split_documents([doc])

        final_chunks.extend(chunks)

    print(f"✅ Final Result: {len(final_chunks)} optimized chunks.")
    return final_chunks


def enrich_metadata(chunks: List[Document]) -> List[Document]:
    print("🏷️  Enriching Metadata (Hierarchical Header Detection)...")

    # Track hierarchy: major section > subsection > subsubsection
    current_major_section = "General Introduction"
    current_subsection = None
    current_level = None

    # Hierarchical pattern with priority levels
    major_section_pattern = r"^(FOREWORD|ACKNOWLEDGEMENTS|INTRODUCTION|BACKGROUND|RECOMMENDATIONS|EXECUTIVE SUMMARY|Research gaps|References|Glossary)"

    intervention_pattern = r"^Recommendations\s+for\s+(organizational interventions|training managers|training workers|individual interventions|returning to work|gaining employment|screening)"

    recommendation_pattern = r"^(?:RECOMMENDATION|KEY QUESTION)\s+(\d+[A-Z]?)"

    subsection_pattern = r"^(Key remarks:|Common implementation remarks:|Implementation remarks:|Evidence and rationale|Evidence-to-decision|Subgroup remarks:|Monitoring and evaluation remarks:|Additional remarks:)"

    annex_pattern = r"^Annex\s+(\d+|[A-Z]+)"

    for chunk in chunks:
        content = chunk.page_content.strip()

        # Major sections (highest priority)
        if match := re.search(
            major_section_pattern, content, re.MULTILINE | re.IGNORECASE
        ):
            current_major_section = match.group(1).strip()
            current_subsection = None
            current_level = "major"

        # Recommendations for X type (e.g., organizational interventions)
        elif match := re.search(
            intervention_pattern, content, re.MULTILINE | re.IGNORECASE
        ):
            intervention_type = match.group(1).strip()
            current_major_section = f"Recommendations for {intervention_type}"
            current_subsection = None
            current_level = "intervention_type"

        # Individual recommendation (1, 2, 3A, etc.)
        elif match := re.search(
            recommendation_pattern, content, re.MULTILINE | re.IGNORECASE
        ):
            rec_number = match.group(1).strip()
            # We don't change current_major_section here, it inherits from Level 2
            current_subsection = f"Recommendation {rec_number}"
            current_level = "recommendation"

        # Subsections within a recommendation
        elif match := re.search(
            subsection_pattern, content, re.MULTILINE | re.IGNORECASE
        ):
            subsection_title = match.group(1).strip().replace(":", "")
            current_subsection = subsection_title
            current_level = "subsection"

        # Level 5: Annexes
        elif match := re.search(annex_pattern, content, re.MULTILINE | re.IGNORECASE):
            annex_number = match.group(1).strip()
            current_major_section = f"Annex {annex_number}"
            current_subsection = None
            current_level = "annex"

        # Build full hierarchical section string
        if current_subsection:
            full_section = f"{current_major_section} > {current_subsection}"
        else:
            full_section = current_major_section

        # Enrich metadata with hierarchy info
        chunk.metadata["section"] = full_section
        chunk.metadata["major_section"] = current_major_section
        chunk.metadata["subsection"] = current_subsection or "N/A"
        chunk.metadata["level"] = current_level or "content"

        # Prepend section context to content
        chunk.page_content = (
            f"SECTION: {full_section}\n\nLEVEL: {current_level}\n\n{content}"
        )

    return chunks


def calculate_deterministic_ids(chunks: List[Document]) -> List[Document]:
    print("🆔 Calculating Deterministic IDs...")
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", -1)
        content = chunk.page_content

        chunk_signature = f"{source}:{page}:{content}"
        chunk_id = hashlib.sha256(chunk_signature.encode()).hexdigest()

        chunk.metadata["id"] = chunk_id

    return chunks


def save_chunks_for_bm25(chunks: List[Document]):
    print("💾 Saving chunks for Hybrid Search (BM25)...")
    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(chunks, f)
    print(f"✅ Chunks saved to {CHUNKS_FILE}")


def add_to_chroma(chunks: List[Document], embedding_function):
    print("📥 Indexing Vectors...")

    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"   Existing documents in DB: {len(existing_ids)}")

    new_chunks = [c for c in chunks if c.metadata["id"] not in existing_ids]

    if len(new_chunks):
        print(f"👉 Adding {len(new_chunks)} new documents...")

        batch_size = 100
        for i in tqdm(
            range(0, len(new_chunks), batch_size), desc="Indexing", unit="batch"
        ):
            batch = new_chunks[i : i + batch_size]
            batch_ids = [c.metadata["id"] for c in batch]
            db.add_documents(batch, ids=batch_ids)

        print("✅ Success! Database updated.")
    else:
        print("✅ Database is already up to date. No new chunks added.")


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    if os.path.exists(CHUNKS_FILE):
        os.remove(CHUNKS_FILE)
    print("   Database cleared.")


if __name__ == "__main__":
    main()
