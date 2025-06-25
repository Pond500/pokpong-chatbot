import os
from llama_index.core import (
    StorageContext,
    load_index_from_storage,
    Settings,
    SimpleDirectoryReader, # [v6 UPGRADE] เพิ่มเข้ามา
)
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.postprocessor import SentenceTransformerRerank
# --- [v6 UPGRADE] Imports for Hybrid Search ---
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
# ---------------------------------------------
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
import tiktoken

# --- Token Counter Setup (เหมือนเดิม) ---
try:
    Settings.tokenizer = tiktoken.get_encoding("cl100k_base")
except:
    Settings.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
token_counter = TokenCountingHandler()
Settings.callback_manager = CallbackManager([token_counter])
print("Token counter initialized.")


# --- [v6 UPGRADE] Centralized Configuration ---
# กรุณาแก้ไข Path ทั้งสองให้ถูกต้องตามเครื่องของคุณ
FAISS_PATH = "/home/ai-intern02/Pond_Rag/Dopa_Indexing_Project/DOPA_INDEX_FINAL"
# DATA_PATH จำเป็นสำหรับ BM25Retriever เพื่อให้เข้าถึงเนื้อหาต้นฉบับได้
DATA_PATH = "/home/ai-intern02/Pond_Rag/Dopa_Indexing_Project/Dopa"
# -----------------------------------------------

if not os.path.exists(FAISS_PATH): raise FileNotFoundError(f"ไม่พบไดเรกทอรี FAISS index ที่: {FAISS_PATH}")
if not os.path.exists(DATA_PATH): raise FileNotFoundError(f"ไม่พบไดเรกทอรีข้อมูลสำหรับ BM25 ที่: {DATA_PATH}")

# --- AI Models Setup (เหมือนเดิม) ---
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
llm = OpenAILike(model='google/gemma-3-27b-it', api_base="http://203.156.3.45/vllm/v1", api_key="dopa", temperature=0.5, context_window=9000, max_new_tokens=1000, is_chat_model=True, timeout=120)
reranker = SentenceTransformerRerank(model="BAAI/bge-reranker-v2-m3", top_n=3)
print("Core AI models initialized.")


# --- [v6 UPGRADE] HYBRID SEARCH ENGINE SETUP ---
print("Initializing Hybrid Search Engine...")

# 1. โหลดเอกสาร (จำเป็นสำหรับ BM25 Retriever)
# หมายเหตุ: ใน Production จริง ส่วนนี้ควรทำแยกในสคริปต์ ingest ข้อมูล
print(f"Loading documents from {DATA_PATH} for BM25 index...")
documents = SimpleDirectoryReader(DATA_PATH, recursive=True).load_data()
# ใช้ Node Parser จาก Settings ที่มีอยู่แล้วเพื่อความสอดคล้องกัน
storage_context = StorageContext.from_defaults()
storage_context.docstore.add_documents(documents)
nodes = Settings.node_parser.get_nodes_from_documents(documents)
print(f"Loaded {len(documents)} documents, processed into {len(nodes)} nodes for BM25.")

# 2. โหลด Vector Index (FAISS) ที่มีอยู่แล้ว
print(f"Loading pre-built FAISS index from {FAISS_PATH}...")
vector_store = FaissVectorStore.from_persist_dir(FAISS_PATH)
index = load_index_from_storage(
    StorageContext.from_defaults(vector_store=vector_store, persist_dir=FAISS_PATH)
)
print("FAISS index loaded.")

# 3. สร้าง Retriever สองประเภท
# Vector Retriever: ค้นหาตามความหมาย
vector_retriever = index.as_retriever(similarity_top_k=10)
# Keyword Retriever (BM25): ค้นหาตามคีย์เวิร์ด
bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=10)
print("Vector and BM25 retrievers created.")

# 4. สร้าง Hybrid Retriever โดยการรวม (Fuse) ทั้งสองระบบ
retriever = QueryFusionRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    similarity_top_k=10,
    num_queries=1,  # ใช้คำถามเดิม ไม่ต้องสร้างคำถามใหม่
    mode="reciprocal_rerank",  # ใช้เทคนิค Reciprocal Rank Fusion (RRF)
    use_async=True,
)
print("Hybrid Search Engine (Vector + BM25 with RRF) is ready.")
# ---------------------------------------------------