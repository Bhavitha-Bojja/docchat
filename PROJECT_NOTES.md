# DocChat — Build Journal

A personal build log capturing decisions, bugs hit, and lessons learned while building DocChat.

---

## Motivation

I wanted to build a document Q&A tool from scratch to deepen my hands-on RAG implementation skills. At Yeshiva I contributed to a RAG chatbot for the career center — I understood the architecture and was involved in prompt design, embedding workflows, and knowledge-base structuring, but I wasn't writing the core code. This project closes that gap.

Use case: a tool where the user drops in any PDF and gets a quick summary plus the ability to ask follow-up questions. The goal is to read long documents faster — get the gist quickly, then dig into specifics.

---

## Phase 1 — Setup

- Python venv, VS Code, `.env` + `.gitignore` from the start
- Installed LangChain, ChromaDB, sentence-transformers, langchain-groq, pypdf, python-dotenv
- Free Groq API key (Llama 3.3 70B) — no cost barrier
- Small `test_setup.py` to verify LLM call works end-to-end

---

## Phase 2 — Document Loading and Chunking

- `PyPDFLoader` to load PDFs
- `RecursiveCharacterTextSplitter` with chunk_size=1000, overlap=200
- The "recursive" name refers to fallback split strategy: try paragraphs → lines → spaces → arbitrary cuts
- Chunk size is a tradeoff: smaller = lose context, bigger = lose retrieval precision

---

## Phase 3 — Embeddings + Vector Database

### What I built
- `HuggingFaceEmbeddings` with `sentence-transformers/all-MiniLM-L6-v2` (local, free)
- ChromaDB persisting to `./chroma_db`

### Why local embeddings?
Every chunk is embedded once at index time. Every query is embedded on the fly. Running locally avoids per-call API costs and is plenty fast for a personal tool.

### 🐛 Bug 1: Duplicate embeddings on re-runs
Re-running the build script kept appending to the persistent store instead of replacing. After 3 runs, the same 52 chunks existed 3 times → retrieval returned the same chunk three times because it was in the DB three times.

**Fix:** Detect if the store already exists; load it instead of rebuilding.
**Lesson:** Persistent vector stores need explicit "build vs. load" logic. Default behavior is append, not replace.

### 🐛 Bug 2: Stale embeddings when the PDF changes
The fix to Bug 1 introduced a new failure mode: if a user swaps the PDF without deleting `chroma_db`, the script silently loads OLD vectors. New document is never indexed — no error, just wrong results.

**Fix:** MD5 hash the source PDF, store the hash with the vector DB. On each run, compare current hash to stored. Match → load. Mismatch → rebuild.
**Bonus:** Made the PDF path a command-line argument so the same script works on any document.
**Lesson:** Persistence + change detection is a real concern. For production, named collections per document would be cleaner.

---

## Phase 4 — Summary + Q&A Loop

### What I built
- `chat.py` with two functions:
  - `generate_summary()` — pulls broad chunks and asks LLM for a 3-5 sentence summary
  - `answer_question()` — embeds question, retrieves top-4 chunks, formats grounded prompt, sends to Groq
- Strict `PromptTemplate`: "Use ONLY the following context. If not found, say 'I couldn't find that.'"
- Source page tracking on every answer
- Simple CLI loop

### Why temperature=0?
Deterministic factual Q&A. Higher temperature is for creative writing. For RAG, low temperature reduces hallucination.

### Observation
Specific technical questions (e.g., "self-attention mechanism") retrieved well. Generic queries ("main topic") didn't — the Transformer paper has lots of repetitive tokenized text (attention diagrams) that cluster densely in embedding space, so generic queries grabbed those instead of the abstract.

---

## Phase 5 — Conversational Memory

### What I built
- New file `chat_with_memory.py` (kept original `chat.py` for comparison)
- `ConversationBufferWindowMemory(k=5)` — keeps last 5 turns
- `ConversationalRetrievalChain` — pre-built LangChain pipeline that handles memory + query rewriting

### The key insight: memory alone isn't enough
A follow-up like "how does it work?" is too vague to retrieve well. The chain handles this by:
1. Looking at history + new question
2. Asking the LLM to **rewrite the new question as a standalone version** ("how does the Transformer architecture work?")
3. Embedding the rewritten question and retrieving
4. Generating the final answer with both context and history

So each turn = 2 LLM calls (rewrite + answer). Tradeoff: latency and cost for coherent multi-turn conversations.

### Why a separate file instead of editing `chat.py`?
- Easier before/after comparison
- If memory breaks, original still works
- Cleaner Git history — two distinct features as two commits

### Tradeoff I noticed
The default chain prompts are less constraining than my strict prompt in `chat.py`. Answers occasionally drift to general knowledge with a caveat. For production, I'd customize the chain's prompts to enforce strict grounding.

---

## Phase 5b — Real-world testing (memory)

Ran a 10-question test on the Transformer paper. Mixed pronouns, topic switches, off-topic, and meta-questions. Three real limitations surfaced:

### ✅ What worked
- **Pronoun resolution.** "Who created it?", "How is it different from RNNs?", "Can you explain that more simply?" — the chain rewrote each correctly using history.
- **Topic switches.** "What datasets were used?" cleanly retrieved new sources without polluting from prior context.
- **Off-topic refusal.** "What's the weather today?" — correctly declined and noted the document is about a research paper.

### ❌ What broke

**1. Generic conceptual questions missed the abstract.**
"What problem does it solve?" returned "I don't know" — but the answer is in the paper's abstract. The embedding for "what problem does it solve" doesn't match the abstract's academic phrasing ("we propose..."). Embeddings don't bridge that linguistic gap.

**2. Numeric/tabular comparisons broke.**
"Which had the highest score?" gave a garbled answer listing scores in random order. The paper's score table flattens awkwardly when PDF-parsed — the LLM sees scattered numbers and can't reconstruct the comparison.

**3. Meta-questions about the conversation failed.**
"Summarize what we've discussed so far" returned "we haven't talked yet." The chain is hardwired to retrieve from the document, not from memory. There's no mode for "answer using memory only, skip retrieval."

### What I'd fix next
- **Generic question issue** → hybrid search (keyword + semantic) or boosting abstract chunks via metadata
- **Table issue** → preprocessing layer that extracts tables separately, or sends raw images of tables to a vision model
- **Meta-question issue** → routing layer that detects meta vs. document questions and skips retrieval for the former

### Overall takeaway from this test
RAG is harder than it looks once you actually test it. Pronoun resolution works elegantly. But generic queries, structured data, and meta-questions each need their own engineering. The interesting work in RAG isn't the architecture diagram — it's the cases where the architecture breaks.

---

## What's next

Planned future sessions:
1. **Streamlit UI** — turn the CLI into a web interface, makes it demo-able
2. **Multi-document support** — named ChromaDB collections per document
3. **Local LLM (Ollama)** — for actual privacy, swap Groq with a local model
4. **Hybrid search** — combine keyword and semantic to handle the generic-query problem
5. **Conversation memory improvements** — custom prompts to enforce strict grounding, routing for meta-questions

---

## Overall reflection

The thing this project gave me that I couldn't have gotten any other way: **specific things to talk about when asked about RAG.** Not architectural overviews — concrete bugs I hit, decisions I made, tradeoffs I understood after the fact. Building it was less about the final tool and more about earning the right to discuss the details.

Things I now know firsthand:
- Why local embeddings vs. API embeddings is a real cost decision
- The duplicate-embedding-on-re-run bug and how to prevent it
- The stale-embedding bug when the source changes, and the hash-check fix
- Why query rewriting matters for memory + RAG
- Where RAG breaks: generic queries, tables, meta-questions
- The 2-LLM-call cost of conversational chains
