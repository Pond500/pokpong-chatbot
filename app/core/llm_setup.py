# app/core/llm_setup.py

import os
import json
import glob
from llama_index.core import (
    StorageContext,
    load_index_from_storage,
    Settings,
    Document,
    VectorStoreIndex,
    SimpleDirectoryReader,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
import tiktoken

# --- [v8] CONFIGURATION ---
# !!! สำคัญ: แก้ไข Path เหล่านี้ให้เป็น Path ที่ถูกต้องในเครื่องของคุณ !!!
PROJECT_ROOT_PATH = "/home/ai-intern02/Pond_Rag/Dopa_Indexing_Project"
FAISS_PATH = f"{PROJECT_ROOT_PATH}/New_dopa_url/index_VWurl/DOPA9_INDEX"
DATA_PATH = f"{PROJECT_ROOT_PATH}/Dopa"
FAQ_JSONL_PATH = f"{PROJECT_ROOT_PATH}/New_dopa_url/Link_out_scope"
# ---------------------------------------------------------------------

EMBED_MODEL_NAME = "BAAI/bge-m3"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
LLM_MODEL_NAME = "google/gemma-3-27b-it"
LLM_API_BASE = "http://203.156.3.45/vllm/v1"
LLM_API_KEY = "dopa"
RAG_CHUNK_SIZE = 800
RAG_CHUNK_OVERLAP = 200

# --- Token Counter Setup ---
try:
    Settings.tokenizer = tiktoken.get_encoding("cl100k_base")
except:
    Settings.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
token_counter = TokenCountingHandler()
Settings.callback_manager = CallbackManager([token_counter])
print("Token counter initialized.")

# --- Path Validation ---
if not os.path.exists(FAISS_PATH): raise FileNotFoundError(f"ไม่พบไดเรกทอรี FAISS index ที่: {FAISS_PATH}")
if not os.path.exists(DATA_PATH): raise FileNotFoundError(f"ไม่พบไดเรกทอรีข้อมูลสำหรับ BM25 ที่: {DATA_PATH}")

# --- AI Models Setup ---
Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
llm = OpenAILike(model=LLM_MODEL_NAME, api_base=LLM_API_BASE, api_key=LLM_API_KEY, temperature=0.5, context_window=8000, max_new_tokens=1500, is_chat_model=True, timeout=120)
Settings.llm = llm
reranker = SentenceTransformerRerank(model=RERANKER_MODEL_NAME, top_n=3)

# --- [v8] Function to load FAQ data ---
def load_faq_retriever(jsonl_path: str, embed_model):
    print(f"INFO: Loading FAQ data from path: {jsonl_path}")
    faq_documents = []
    all_jsonl_files = glob.glob(os.path.join(jsonl_path, "**", "*.jsonl"), recursive=True)

    if not all_jsonl_files:
        print(f"WARNING: No .jsonl files found in {jsonl_path}. FAQ feature will be disabled.")
        return None

    for file_path in all_jsonl_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    doc = Document(
                        text=data.get("title", ""),
                        metadata={"title": data.get("title", "N/A"), "url": data.get("url", "N/A")}
                    )
                    faq_documents.append(doc)
                except json.JSONDecodeError:
                    print(f"WARNING: Skipping malformed JSON on line {line_num} in file {file_path}")
                    continue

    if not faq_documents:
        print("!!! WARNING: Loaded files but found no valid FAQ entries. FAQ feature will be disabled.")
        return None

    print(f"INFO: Found {len(faq_documents)} FAQ entries. Creating in-memory FAQ index...")
    faq_index = VectorStoreIndex.from_documents(faq_documents, embed_model=embed_model, show_progress=False)
    print("INFO: FAQ Retriever created successfully.")
    return faq_index.as_retriever(similarity_top_k=5)

# --- System 1: FAQ Link Retriever ---
faq_retriever = load_faq_retriever(FAQ_JSONL_PATH, Settings.embed_model)

# --- System 2: Hybrid Search Retriever for RAG ---
print("Initializing Hybrid Search Engine for RAG...")
documents = SimpleDirectoryReader(DATA_PATH, recursive=True).load_data()
node_parser = SentenceSplitter(chunk_size=RAG_CHUNK_SIZE, chunk_overlap=RAG_CHUNK_OVERLAP)
nodes = node_parser.get_nodes_from_documents(documents)
print(f"Loaded {len(nodes)} nodes for BM25.")

rag_index = load_index_from_storage(StorageContext.from_defaults(vector_store=FaissVectorStore.from_persist_dir(FAISS_PATH), persist_dir=FAISS_PATH))
print("FAISS index loaded.")

vector_retriever = rag_index.as_retriever(similarity_top_k=10)
bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=10)
print("RAG Vector & BM25 Retrievers created.")

retriever = QueryFusionRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    similarity_top_k=10,
    mode="reciprocal_rerank",
    use_async=True,
)
print("Hybrid Search Engine (Vector + BM25 with RRF) is ready.")