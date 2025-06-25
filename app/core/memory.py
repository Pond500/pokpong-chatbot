import asyncio
from llama_index.core.memory import ChatMemoryBuffer

class UserChatMemory:
    def __init__(self, token_limit: int = 5000):
        self.user_memories = {}
        self.token_limit = token_limit
        self._lock = asyncio.Lock()

    async def get_memory(self, session_id: str) -> ChatMemoryBuffer:
        async with self._lock:
            if session_id not in self.user_memories:
                self.user_memories[session_id] = ChatMemoryBuffer.from_defaults(token_limit=self.token_limit)
            return self.user_memories[session_id]

    async def reset_memory(self, session_id: str):
        async with self._lock:
            if session_id in self.user_memories:
                self.user_memories[session_id].reset()

# สร้าง instance ไว้เพื่อให้ไฟล์อื่น import ไปใช้งานได้เลย
user_memory_manager = UserChatMemory()