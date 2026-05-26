import os
import sys
import json
import hashlib
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

PERSIST_DIR = "./chroma_db"
METADATA_FILE = os.path.join(PERSIST_DIR, "source_info.json")


def file_hash(path):
    """Generate a short fingerprint of a file's contents."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def get_pdf_path():
    """Get PDF path from command line, or default to test.pdf."""
    if len(sys.argv) > 1:
        return sys.argv[1]
    return "test.pdf"


def needs_rebuild(pdf_path):
    """Return True if the vector store is missing OR the PDF has changed."""
    if not os.path.exists(METADATA_FILE):
        return True
    with open(METADATA_FILE, "r") as f:
        stored = json.load(f)
    return stored.get("pdf_hash") != file_hash(pdf_path)


def build_vectorstore(pdf_path, embeddings):
    """Load PDF, chunk it, embed, and persist to ChromaDB."""
    print(f"\nBuilding vector store from: {pdf_path}")

    # Delete any existing store to start clean
    if os.path.exists(PERSIST_DIR):
        print("Removing old vector store...")
        shutil.rmtree(PERSIST_DIR)

    # Load
    print("Loading PDF...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages")

    # Chunk
    print("Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")

    # Embed and store
    print("Embedding chunks... (this takes 1-2 min)")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )
    print(f"Stored {vectorstore._collection.count()} embeddings")

    # Save metadata so we know which PDF was indexed
    with open(METADATA_FILE, "w") as f:
        json.dump({
            "pdf_path": pdf_path,
            "pdf_hash": file_hash(pdf_path),
            "chunks": len(chunks)
        }, f, indent=2)

    return vectorstore


def load_vectorstore(embeddings):
    """Load an existing vector store from disk."""
    print("Loading existing vector store...")
    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )
    print(f"Loaded {vectorstore._collection.count()} embeddings")
    return vectorstore


def main():
    pdf_path = get_pdf_path()

    if not os.path.exists(pdf_path):
        print(f"ERROR: PDF not found at '{pdf_path}'")
        print("Usage: python build_vectorstore.py [path_to_pdf]")
        sys.exit(1)

    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    if needs_rebuild(pdf_path):
        print("PDF is new or has changed → rebuilding vector store")
        vectorstore = build_vectorstore(pdf_path, embeddings)
    else:
        print("PDF unchanged since last build → loading existing store")
        vectorstore = load_vectorstore(embeddings)

    # Quick smoke test
    print("\n--- TEST RETRIEVAL ---")
    test_query = "What is the main topic of this document?"
    print(f"Query: {test_query}\n")
    results = vectorstore.similarity_search(test_query, k=3)
    for i, doc in enumerate(results, 1):
        page = doc.metadata.get("page", "?")
        print(f"--- Result {i} (page {page}) ---")
        print(doc.page_content[:250])
        print()

    print("✅ Vector store ready.")


if __name__ == "__main__":
    main()