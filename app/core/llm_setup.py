import os
from typing import List
from llama_index.core import (StorageContext, load_index_from_storage, Settings)
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import QueryBundle, NodeWithScore
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
import tiktoken

# --- Token Counter Setup ---
try:
    Settings.tokenizer = tiktoken.get_encoding("cl100k_base")
except:
    Settings.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")

token_counter = TokenCountingHandler()
Settings.callback_manager = CallbackManager([token_counter])
print("Token counter initialized.")

# --- CORE AI/LLM SETUP (ส่วนนี้คือหัวใจของ v4) ---
# กรุณาแก้ไข Path ตรงนี้ให้ถูกต้องตามเครื่องของคุณ
FAISS_PATH = "/home/ai-intern02/Pond_Rag/Dopa_Indexing_Project/DOPA_INDEX_FINAL"

if not os.path.exists(FAISS_PATH):
    raise FileNotFoundError(f"ไม่พบไดเรกทอรี FAISS index ที่: {FAISS_PATH}")

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
llm = OpenAILike(model='google/gemma-3-27b-it', api_base="http://203.156.3.45/vllm/v1", api_key="dopa", temperature=0.5, context_window=9000, max_new_tokens=1000, is_chat_model=True, timeout=120)
index = load_index_from_storage(StorageContext.from_defaults(vector_store=FaissVectorStore.from_persist_dir(FAISS_PATH), persist_dir=FAISS_PATH))

class PostprocessingRetriever(BaseRetriever):
    def __init__(self, base_retriever: BaseRetriever, required_top_k: int = 3):
        self._base_retriever = base_retriever
        self._required_top_k = required_top_k
        self._base_retriever.similarity_top_k = required_top_k * 3
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        raise NotImplementedError()

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        return (await self._base_retriever.aretrieve(query_bundle.query_str))[:self._required_top_k]

retriever = PostprocessingRetriever(base_retriever=index.as_retriever(similarity_top_k=3), required_top_k=3)
print("Index, models, and retriever (v4) created successfully.")