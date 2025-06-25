# --- FastAPI & Standard Imports --
import os
import json
import time
import uvicorn
import asyncio
from typing import Dict, Any, AsyncGenerator

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse

# --- LlamaIndex ChatEngine Imports ---
from llama_index.core.chat_engine import CondensePlusContextChatEngine, SimpleChatEngine

# --- Professional Logging Setup ---
import logging
from logging.handlers import TimedRotatingFileHandler
from pythonjsonlogger import jsonlogger

# --- Monitoring ---
from prometheus_fastapi_instrumentator import Instrumentator

# --- Import from our own modules ---
# --- Import from our own modules ---
from .models import ChatRequest, SearchRequest
# import จาก llm_setup เฉพาะสิ่งที่อยู่ในไฟล์นั้นจริงๆ
from .core.llm_setup import llm, retriever, index, token_counter
# import prompt และ route_query จาก router.py
from .core.router import route_query, ROLE_INFO, DOPA_GENERAL_PROMPT, GENERAL_ROLE_INFO
# import จาก memory.py
from .core.memory import user_memory_manager

# --- Logging Setup ---
formatter = jsonlogger.JsonFormatter('%(asctime)s %(name)s %(levelname)s %(message)s', json_ensure_ascii=False)
log_handler = TimedRotatingFileHandler(filename='dopa_chatbot.jsonl', when='midnight', interval=1, backupCount=7, encoding='utf-8')
log_handler.setFormatter(formatter)
logger = logging.getLogger("pokpongV4Logger_C800_context9K_TL5K")
if not logger.handlers:
    logger.addHandler(log_handler)
    logger.setLevel(logging.INFO)

# --- SETUP FastAPI App ---
app = FastAPI(
    title="DOPA RAG Chatbot API (v4.0 - Baseline)",
    description="API สำหรับ 'น้องปกป้อง' เวอร์ชั่นพื้นฐานพร้อมระบบ Logging และ Token Counting",
    version="4.0.0",
)
instrumentator = Instrumentator().instrument(app)
instrumentator.expose(app)


# --- API ENDPOINTS ---
@app.post("/chat", response_class=JSONResponse)
async def chat_with_routing(request: ChatRequest) -> Dict[str, Any]:
    start_time = time.time()
    # กำหนดค่าเริ่มต้นสำหรับตัวแปรทั้งหมด
    decision, full_response_text, response_source, status = "N/A", "", "N/A", "completed"
    reference_path = []
    was_reset = False # <-- สร้างธงสำหรับตรวจสอบการ Reset

    try:
        token_counter.reset_counts() # รีเซ็ตตัวนับ Token
        chat_memory = await user_memory_manager.get_memory(request.session_id)
        
        # ตรรกะการจัดการ NEW_TOPIC
        initial_decision = await asyncio.to_thread(route_query, request.query, chat_memory.get())
        if initial_decision == 'NEW_TOPIC':
            await user_memory_manager.reset_memory(request.session_id)
            was_reset = True # <-- เมื่อมีการ Reset ให้ตั้งค่าธงเป็น True
            decision = await asyncio.to_thread(route_query, request.query, []) 
        else:
            decision = initial_decision

        # ตรรกะการเลือกเส้นทาง
        if decision == 'RAG':
            chat_engine = CondensePlusContextChatEngine.from_defaults(retriever=retriever, llm=llm, memory=chat_memory, system_prompt=ROLE_INFO)
            response = await chat_engine.achat(request.query)
            full_response_text = response.response
            if response.source_nodes:
                response_source = "QA"
                for source_node in response.source_nodes:
                    meta = source_node.metadata
                    path = os.path.join(meta.get("category", ""), meta.get("file_name", "N/A"))
                    reference_path.append(path)
            else:
                response_source, reference_path = "QA", ["ฐานข้อมูลภายใน"]

        elif decision == 'DOPA_GENERAL':
            chat_engine = SimpleChatEngine.from_defaults(llm=llm, memory=chat_memory, system_prompt=DOPA_GENERAL_PROMPT)
            response = await chat_engine.achat(request.query)
            full_response_text = response.response
            response_source, reference_path = "General DOPA Knowledge", []

        elif decision == 'GENERAL_LLM':
            chat_engine = SimpleChatEngine.from_defaults(llm=llm, memory=chat_memory, system_prompt=GENERAL_ROLE_INFO)
            response = await chat_engine.achat(request.query)
            full_response_text = response.response
            response_source, reference_path = "N/A", []
            
        else:
            raise ValueError(f"Invalid final decision: {decision}")
            
    except Exception as e:
        full_response_text, status = f"ขออภัยค่ะ เกิดข้อผิดพลาดในการประมวลผล: {e}", "error"
        logger.error("Exception in /chat endpoint", exc_info=True, extra={"extra": {"request_body": request.model_dump()}})
    finally:
        elapsed = time.time() - start_time
        log_data = {
            "endpoint": "/chat",
            "session_id": request.session_id,
            "user_query": request.query,
            "routing_decision": decision,
            "memory_reset": "Done" if was_reset else "N/A", # <-- ใส่สถานะการ Reset ลง Log
            "source": response_source,
            "reference": reference_path,
            "response": full_response_text.strip(),
            "elapsed_time": f"{elapsed:.3f}s",
            "status": status,
            "token_usage": {
                "prompt_tokens": token_counter.prompt_llm_token_count,
                "completion_tokens": token_counter.completion_llm_token_count,
                "total_tokens": token_counter.total_llm_token_count,
            }
        }
        if status == "error":
            logger.error("Chat request failed", extra=log_data)
        else:
            logger.info("Chat request processed", extra=log_data)
    
    return JSONResponse(content={"session_id": request.session_id, "query": request.query, "decision": decision, "response": full_response_text, "source": response_source, "reference": reference_path, "status": status, "time_seconds": f"{elapsed:.3f}s"})


@app.post("/chat/stream", response_class=StreamingResponse)
async def stream_chat_with_routing(request: ChatRequest) -> StreamingResponse:
    async def generate_stream_response() -> AsyncGenerator[str, None]:
        start_time = time.time()
        session_id, user_query = request.session_id, request.query
        
        # กำหนดค่าเริ่มต้นสำหรับตัวแปรทั้งหมด
        decision, full_response_text, response_source, status = "N/A", "", "N/A", "completed"
        reference_path = []
        was_reset = False # <-- สร้างธงสำหรับตรวจสอบการ Reset
        final_payload = {}

        try:
            token_counter.reset_counts() # รีเซ็ตตัวนับ Token
            chat_memory = await user_memory_manager.get_memory(session_id)

            # ตรรกะการจัดการ NEW_TOPIC
            initial_decision = await asyncio.to_thread(route_query, user_query, chat_memory.get())
            if initial_decision == 'NEW_TOPIC':
                await user_memory_manager.reset_memory(session_id)
                was_reset = True # <-- เมื่อมีการ Reset ให้ตั้งค่าธงเป็น True
                decision = await asyncio.to_thread(route_query, user_query, [])
            else:
                decision = initial_decision

            # ตรรกะการเลือกเส้นทาง
            if decision == 'RAG':
                chat_engine = CondensePlusContextChatEngine.from_defaults(retriever=retriever, llm=llm, memory=chat_memory, system_prompt=ROLE_INFO)
                streaming_response = await chat_engine.astream_chat(user_query)
                async for token in streaming_response.async_response_gen():
                    full_response_text += token
                    yield f"data: {json.dumps({'type': 'token', 'content': token}, ensure_ascii=False)}\n\n"
                
                if streaming_response.source_nodes:
                    response_source = "QA"
                    for source_node in streaming_response.source_nodes:
                        meta = source_node.metadata
                        path = os.path.join(meta.get("category", ""), meta.get("file_name", "N/A"))
                        reference_path.append(path)
                else:
                    response_source, reference_path = "QA", ["ฐานข้อมูลภายใน"]

            elif decision == 'DOPA_GENERAL':
                chat_engine = SimpleChatEngine.from_defaults(llm=llm, memory=chat_memory, system_prompt=DOPA_GENERAL_PROMPT)
                streaming_response = await chat_engine.astream_chat(user_query)
                async for token in streaming_response.async_response_gen():
                    full_response_text += token
                    yield f"data: {json.dumps({'type': 'token', 'content': token}, ensure_ascii=False)}\n\n"
                response_source, reference_path = "General DOPA Knowledge", []

            elif decision == 'GENERAL_LLM':
                chat_engine = SimpleChatEngine.from_defaults(llm=llm, memory=chat_memory, system_prompt=GENERAL_ROLE_INFO)
                streaming_response = await chat_engine.astream_chat(user_query)
                async for token in streaming_response.async_response_gen():
                    full_response_text += token
                    yield f"data: {json.dumps({'type': 'token', 'content': token}, ensure_ascii=False)}\n\n"
                response_source, reference_path = "N/A", []
            else:
                raise ValueError(f"Invalid final decision: {decision}")
        
        except Exception as e:
            status, full_response_text = "error", f"ขออภัยค่ะ เกิดข้อผิดพลาดในการประมวลผล: {e}"
            logger.error("Exception in /chat/stream endpoint", exc_info=True, extra={"extra": {"request_body": request.model_dump()}})
            yield f"data: {json.dumps({'type': 'error', 'content': full_response_text}, ensure_ascii=False)}\n\n"
        
        finally:
            elapsed = time.time() - start_time
            final_payload = {
                'type': 'final', 'session_id': session_id, 'query': user_query, 
                'decision': decision, 'response': full_response_text.strip(), 
                'source': response_source, 'reference': reference_path, 'status': status, 
                'time_seconds': f"{elapsed:.3f}s"
            }
            yield f"data: {json.dumps(final_payload, ensure_ascii=False)}\n\n"
            
            log_data = final_payload.copy()
            if 'type' in log_data: del log_data['type']
            
            log_data["memory_reset"] = "Done" if was_reset else "N/A" # <-- ใส่สถานะการ Reset ลง Log
            log_data["token_usage"] = {
                "prompt_tokens": token_counter.prompt_llm_token_count,
                "completion_tokens": token_counter.completion_llm_token_count,
                "total_tokens": token_counter.total_llm_token_count,
            }

            if status == "error":
                logger.error("Streamed chat request failed", extra=log_data)
            else:
                logger.info("Streamed chat request processed", extra=log_data)
                
    return StreamingResponse(generate_stream_response(), media_type="text/event-stream")


@app.post("/search", response_class=JSONResponse)
async def diagnostic_search(request: SearchRequest) -> Dict[str, Any]:
    start_time = time.time()
    user_query = request.query
    final_response = {}
    log_data = {"endpoint": "/search", "session_id": request.session_id, "user_query": user_query}
    status = "completed"
    try:
        token_counter.reset_counts()
        routing_decision = await asyncio.to_thread(route_query, user_query, [])
        log_data["routing_decision"] = routing_decision
        
        if routing_decision != 'RAG':
            final_response = {"query": user_query, "queryAnalysis": {"routingDecision": routing_decision, "notes": "Query was not routed to RAG, no retrieval performed."}}
        else:
            search_retriever = index.as_retriever(similarity_top_k=request.top_k)
            nodes_with_scores = await search_retriever.aretrieve(user_query)
            retrieval_results = [{"score": float(node.score), "metadata": node.metadata, "text_preview": node.text[:250] + "..."} for node in nodes_with_scores]
            log_data["results_count"] = len(retrieval_results)
            context_for_llm = "\n\n---\n\n".join([node.get_content() for node in nodes_with_scores])
            full_prompt_for_llm = f"{ROLE_INFO}\n\n### Context from Knowledge Base:\n{context_for_llm}\n\n### User's Query:\n{user_query}\n\n### Your Answer:\n"
            final_response = {"query": user_query, "queryAnalysis": {"routingDecision": routing_decision}, "retrievalResults": {"count": len(retrieval_results), "nodes": retrieval_results}, "contextThatWouldBeSentToLLM": context_for_llm, "fullPromptForLLM": full_prompt_for_llm}
    except Exception as e:
        status = "error"; final_response = {"error": f"An error occurred during search: {e}"}
        logger.error("Exception in /search endpoint", exc_info=True, extra={"extra": {"request_body": request.model_dump()}})
    finally:
        elapsed = time.time() - start_time
        final_response["time_seconds"] = f"{elapsed:.3f}s"
        log_data["elapsed_time"] = f"{elapsed:.3f}s"; log_data["status"] = status
        log_data["token_usage"] = {"embedding_tokens": token_counter.total_embedding_token_count}
        if status == "error": logger.error("Search request failed", extra=log_data)
        else: logger.info("Search request processed", extra=log_data)
    return JSONResponse(content=final_response)

@app.post("/reset_chat")
async def reset_chat_session(request: ChatRequest):
    await user_memory_manager.reset_memory(request.session_id)
    logger.info(f"Chat session reset", extra={"extra": {"session_id": request.session_id}})
    return {"status": "success", "message": f"Chat history for session_id '{request.session_id}' has been reset."}


# --- RUN ---
if __name__ == "__main__":
    # สังเกตว่า path การ run เปลี่ยนไป
    uvicorn.run("app.main:app", host="0.0.0.0", port=6666, reload=True)