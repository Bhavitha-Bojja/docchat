import os
import sys
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory

PERSIST_DIR = "./chroma_db"

load_dotenv()

# --- Setup ---
print("Initializing DocChat with memory...")

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

# --- Memory: keep last 5 turns ---
memory = ConversationBufferWindowMemory(
    k=5,  # last 5 turns
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

# --- Conversational chain ---
# This automatically handles:
# - Query rewriting (uses history to make follow-ups standalone)
# - Retrieval (using rewritten query)
# - Answer generation (with context + question)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
    memory=memory,
    return_source_documents=True,
    verbose=False
)


def main():
    print("\n" + "=" * 60)
    print("📄 DocChat (with memory) — Ask questions about your document")
    print("=" * 60)
    print("\nFollow-up questions work — try things like 'how does it work?' after asking about a topic.")
    print("Type 'quit' or 'exit' to leave.\n")

    while True:
        question = input("\n❓ Your question: ").strip()
        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("\nBye! 👋")
            break

        print("\n🤔 Thinking...")
        result = qa_chain.invoke({"question": question})

        print(f"\n💡 Answer:\n{result['answer']}")

        # Show source pages
        sources = result.get("source_documents", [])
        pages = sorted(set(doc.metadata.get("page", "?") for doc in sources))
        print(f"\n📖 Sources: pages {pages}")


if __name__ == "__main__":
    main()