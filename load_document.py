from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Step 1: Load the PDF
print("Loading PDF...")
loader = PyPDFLoader("test.pdf")
documents = loader.load()

# documents is a list — one item per page
print(f"Loaded {len(documents)} pages from the PDF")
print(f"First 200 characters of page 1:\n{documents[0].page_content[:200]}")
print("---")

# Step 2: Split into smaller chunks
print("Splitting into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,       # Each chunk = ~1000 characters
    chunk_overlap=200,     # Overlap so context isn't lost between chunks
    separators=["\n\n", "\n", " ", ""]
)

chunks = text_splitter.split_documents(documents)

print(f"Split into {len(chunks)} chunks")
print(f"First chunk preview:\n{chunks[0].page_content[:300]}")