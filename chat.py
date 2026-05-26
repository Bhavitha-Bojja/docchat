import os
import sys
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

PERSIST_DIR = "./chroma_db"

# Load env vars
load_dotenv()

# --- Setup the components ---
print("Initializing DocChat...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

if not os.path.exists(PERSIST_DIR):
    print("ERROR: No vector store found. Run build_vectorstore.py first.")
    sys.exit(1)

vectorstore = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings
)
print(f"Loaded vector store with {vectorstore._collection.count()} chunks")

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)


# --- Prompt template: answer ONLY from context ---
QA_PROMPT = PromptTemplate(
    template="""You are a helpful assistant answering questions about a specific document.

Use ONLY the following context from the document to answer the question.
If the context does not contain the answer, say "I couldn't find that in the document."
Do not invent information. Be concise.

Context:
{context}

Question: {question}

Answer:""",
    input_variables=["context", "question"]
)


def answer_question(question, k=4):
    """Retrieve relevant chunks and ask the LLM to answer based on them."""
    # 1. Retrieve relevant chunks
    docs = vectorstore.similarity_search(question, k=k)
    context = "\n\n---\n\n".join(doc.page_content for doc in docs)

    # 2. Build the prompt
    prompt = QA_PROMPT.format(context=context, question=question)

    # 3. Ask the LLM
    response = llm.invoke(prompt)
    return response.content, docs


def generate_summary():
    """Generate a high-level summary using the first several chunks."""
    # Use the first ~5 chunks (usually the beginning of the doc)
    all_docs = vectorstore.similarity_search("introduction overview summary main topic", k=8)
    context = "\n\n---\n\n".join(doc.page_content for doc in all_docs)

    prompt = f"""Based on the following excerpts from a document, write a concise 3-5 sentence summary describing what the document is about.

Excerpts:
{context}

Summary:"""

    response = llm.invoke(prompt)
    return response.content


# --- Main interactive loop ---
def main():
    print("\n" + "=" * 60)
    print("📄 DocChat — Ask questions about your document")
    print("=" * 60)

    # Show initial summary
    print("\nGenerating document summary...\n")
    summary = generate_summary()
    print("📌 SUMMARY:")
    print(summary)
    print("\n" + "-" * 60)

    print("\nNow you can ask questions about the document.")
    print("Type 'quit' or 'exit' to leave.\n")

    while True:
        question = input("\n❓ Your question: ").strip()
        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("\nBye! 👋")
            break

        print("\n🤔 Thinking...")
        answer, sources = answer_question(question)
        print(f"\n💡 Answer:\n{answer}")

        # Show source pages
        pages = sorted(set(doc.metadata.get("page", "?") for doc in sources))
        print(f"\n📖 Sources: pages {pages}")


if __name__ == "__main__":
    main()