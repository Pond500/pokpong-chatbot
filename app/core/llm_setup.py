import os
from llama_index.core import (StorageContext, load_index_from_storage, Settings)
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike
# --- [v5 UPGRADE] Import ที่ต้องเพิ่มเข้ามา ---
from llama_index.core.postprocessor import SentenceTransformerRerank
# -------------------------------------------
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

# --- CORE AI/LLM SETUP ---
# กรุณาแก้ไข Path ตรงนี้ให้ถูกต้องตามเครื่องของคุณ
FAISS_PATH = "/home/ai-intern02/Pond_Rag/Dopa_Indexing_Project/DOPA_INDEX_FINAL" # อาจจะใช้ Index ที่ chunk size เล็กลงเพื่อประสิทธิภาพที่ดีขึ้น

if not os.path.exists(FAISS_PATH):
    raise FileNotFoundError(f"ไม่พบไดเรกทอรี FAISS index ที่: {FAISS_PATH}")

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
llm = OpenAILike(model='google/gemma-3-27b-it', api_base="http://203.156.3.45/vllm/v1", api_key="dopa", temperature=0.5, context_window=9000, max_new_tokens=1000, is_chat_model=True, timeout=120)
index = load_index_from_storage(StorageContext.from_defaults(vector_store=FaissVectorStore.from_persist_dir(FAISS_PATH), persist_dir=FAISS_PATH))


# --- [v5 UPGRADE] ส่วนที่เปลี่ยนแปลงทั้งหมด ---

# 1. ตั้งค่า Re-ranker Model
print("Initializing Re-ranker model (BAAI/bge-reranker-v2-m3)...")
reranker = SentenceTransformerRerank(
    model="BAAI/bge-reranker-v2-m3",
    top_n=3  # คัดเลือกให้เหลือ 3 ชิ้นที่ดีที่สุดหลังจากการ re-rank
)
print("Re-ranker model initialized.")

# 2. สร้าง Retriever ให้หาตัวเลือกมาเผื่อ re-ranker
# เราจะลบ PostprocessingRetriever แบบเก่าทิ้งไป
retriever = index.as_retriever(
    similarity_top_k=10 # ค้นหา 10 อันดับแรกมาเป็นตัวเลือกให้ re-ranker
)
print("Retriever (v5) created to fetch top 10 candidates for re-ranking.")
# ---------------------------------------------