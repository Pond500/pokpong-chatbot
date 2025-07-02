# app/main.py

import os
import json
import time
import uvicorn
import asyncio
from typing import Dict, Any, List, AsyncGenerator

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from llama_index.core import QueryBundle

from llama_index.core.chat_engine import CondensePlusContextChatEngine, SimpleChatEngine

import logging
from logging.handlers import TimedRotatingFileHandler
from pythonjsonlogger import jsonlogger
from prometheus_fastapi_instrumentator import Instrumentator

# --- [v8] Import from our own modules ---
from .models import ChatRequest, SearchRequest
# import สิ่งที่อยู่ใน llm_setup.py
from .core.llm_setup import (
    llm, retriever, reranker, token_counter,
    faq_retriever, vector_retriever, bm25_retriever,
    PROJECT_ROOT_PATH
)
# import สิ่งที่อยู่ใน router.py
from .core.router import route_query, ROLE_INFO, GENERAL_ROLE_INFO
# import สิ่งที่อยู่ใน memory.py
from .core.memory import user_memory_manager

# --- Logging Setup ---
formatter = jsonlogger.JsonFormatter('%(asctime)s %(name)s %(levelname)s %(message)s', json_ensure_ascii=False)
log_handler = TimedRotatingFileHandler(filename='dopa_chatbot_final.jsonl', when='midnight', interval=1, backupCount=7, encoding='utf-8')
log_handler.setFormatter(formatter)
logger = logging.getLogger("dopa_chatbot_final")
if not logger.handlers:
    logger.addHandler(log_handler)
    logger.setLevel(logging.INFO)

# --- FastAPI App Setup ---
app = FastAPI(
    title="DOPA RAG Chatbot API (v8.0 - Hybrid+FAQ Engine)",
    description="API 'น้องปกป้อง' พร้อมระบบ Hybrid Search, FAQ Link Retrieval, และ Re-ranking",
    version="8.0.0",
)
instrumentator.instrument(app)
instrumentator.expose(app)

@app.post("/chat", response_class=JSONResponse)
async def chat_with_routing(request: ChatRequest) -> Dict[str, Any]:
    start_time = time.time()
    decision, full_response_text, response_source, status = "N/A", "", "N/A", "completed"
    reference_path, chunk_preview_data = [], []
    was_reset = False
    
    try:
        token_counter.reset_counts()
        chat_memory = await user_memory_manager.get_memory(request.session_id)
        
        initial_decision = await asyncio.to_thread(route_query, request.query, chat_memory.get())
        if initial_decision == 'NEW_TOPIC':
            await user_memory_manager.reset_memory(request.session_id)
            was_reset = True
            decision = await asyncio.to_thread(route_query, request.query, [])
        else:
            decision = initial_decision

        if decision == 'RAG':
            source_lookup = {}
            vector_nodes = await vector_retriever.aretrieve(request.query)
            for node in vector_nodes:
                source_lookup[node.id_] = 'vector_retriever'
            
            bm25_nodes = await bm25_retriever.aretrieve(request.query)
            for node in bm25_nodes:
                if node.id_ in source_lookup:
                    source_lookup[node.id_] += ', bm25_retriever'
                else:
                    source_lookup[node.id_] = 'bm25_retriever'

            chat_engine = CondensePlusContextChatEngine.from_defaults(
                retriever=retriever, llm=llm, memory=chat_memory,
                system_prompt=ROLE_INFO, node_postprocessors=[reranker]
            )
            response = await chat_engine.achat(request.query)
            full_response_text = response.response
            response_source = "RAG Hybrid Search"

            if response.source_nodes:
                for n in response.source_nodes:
                    final_metadata = n.metadata.copy()
                    final_metadata['retrieval_source'] = source_lookup.get(n.id_, 'unknown_after_fusion')
                    
                    path_to_process = None
                    if 'file_path' in final_metadata:
                        path_to_process = final_metadata['file_path']
                    elif 'category' in final_metadata and 'file_name' in final_metadata:
                        try:
                            full_path_from_vector = os.path.join(final_metadata['category'], final_metadata['file_name'])
                            path_to_process = full_path_from_vector
                        except Exception:
                            path_to_process = final_metadata['category']
                        del final_metadata['category'] 

                    if path_to_process:
                        try:
                            final_metadata['file_path'] = os.path.relpath(path_to_process, PROJECT_ROOT_PATH)
                        except Exception:
                            final_metadata['file_path'] = path_to_process
                        
                        if isinstance(final_metadata.get('file_path'), str):
                            final_metadata['file_path'] = final_metadata['file_path'].replace("New_dopa_url/", "")
                    
                    # จัดลำดับคีย์ใน metadata เพื่อความสวยงาม
                    ordered_metadata = {}
                    key_order = [
                        "file_name", "file_path", "retrieval_source", "type", 
                        "document_title", "tags", "file_type", "file_size", 
                        "creation_date", "last_modified_date"
                    ]

                    for key in key_order:
                        if key in final_metadata:
                            ordered_metadata[key] = final_metadata[key]
                    
                    for key, value in final_metadata.items():
                        if key not in ordered_metadata:
                            ordered_metadata[key] = value
                    
                    reference_path.append({"metadata": ordered_metadata, "score": float(n.score)})

                chunk_preview_data = [
                    {
                        "rank": i + 1,
                        "document_title": n.metadata.get('file_name', 'N/A'),
                        "content": n.get_content()
                    }
                    for i, n in enumerate(response.source_nodes)
                ]

        elif decision == 'FAQ_LINK':
            response_source = "FAQ Link Retrieval"
            if not faq_retriever:
                full_response_text = "ขออภัยค่ะ ขณะนี้ระบบฐานข้อมูลคู่มือบริการขัดข้องค่ะ"
            else:
                initial_results = await faq_retriever.aretrieve(request.query)
                if not initial_results:
                    full_response_text = "น้องปกป้องยังไม่พบข้อมูลที่ตรงกับคำถามของคุณค่ะ"
                else:
                    reranked_results = reranker.postprocess_nodes(initial_results, query_bundle=QueryBundle(request.query))
                    response_parts = ["สำหรับคำถามของคุณนะคะ น้องปกป้องพบหัวข้อที่เกี่ยวข้องดังนี้ค่ะ:", ""]
                    ref_data = []
                    for node in reranked_results:
                        title = node.metadata.get('title', 'N/A')
                        url = node.metadata.get('url', 'N/A')
                        response_parts.append(f"• **{title}**")
                        response_parts.append(f"  ดูรายละเอียดเพิ่มเติมได้ที่: {url}")
                        ref_data.append({"title": title, "url": url, "score": float(node.score)})
                    full_response_text = "\n".join(response_parts)
                    reference_path = ref_data

        elif decision == 'GENERAL_LLM':
            chat_engine = SimpleChatEngine.from_defaults(llm=llm, memory=chat_memory, system_prompt=GENERAL_ROLE_INFO)
            response = await chat_engine.achat(request.query)
            full_response_text, response_source = response.response, "General LLM"
        
        else:
            raise ValueError(f"Invalid final decision: {decision}")

    except Exception as e:
        full_response_text, status = f"ขออภัยค่ะ เกิดข้อผิดพลาดในการประมวลผล: {e}", "error"
        logger.error("Exception in /chat endpoint", exc_info=True, extra={"extra": {"request_body": request.model_dump()}})
        
    finally:
        elapsed = time.time() - start_time
        log_data = {
            "endpoint": "/chat", "session_id": request.session_id, "user_query": request.query,
            "routing_decision": decision, "memory_reset": "Done" if was_reset else "N/A", "source": response_source,
            "response": full_response_text.strip(), "elapsed_time": f"{elapsed:.3f}s", "status": status,
            "reference": reference_path,
            "token_usage": {"prompt_tokens": token_counter.prompt_llm_token_count, "completion_tokens": token_counter.completion_llm_token_count, "total_tokens": token_counter.total_llm_token_count}
        }
        if status == "error": logger.error("Chat request failed", extra=log_data)
        else: logger.info("Chat request processed", extra=log_data)
        
        return JSONResponse(content={
            "session_id": request.session_id, "query": request.query, "decision": decision, 
            "response": full_response_text, "source": response_source, "reference": reference_path, 
            "chunk_preview": chunk_preview_data, "status": status, "time_seconds": f"{elapsed:.3f}s"
        })
    
    
@app.post("/chat/stream", response_class=StreamingResponse, response_model=None)
async def stream_chat_with_routing(request: ChatRequest) -> AsyncGenerator[str, None]:
    async def generate_stream_response() -> AsyncGenerator[str, None]:
        start_time = time.time()
        session_id, user_query = request.session_id, request.query
        decision, full_response_text, response_source, status = "N/A", "", "N/A", "completed"
        reference_path, chunk_preview_data = [], []
        was_reset = False
        final_payload = {}
        
        try:
            token_counter.reset_counts()
            chat_memory = await user_memory_manager.get_memory(session_id)
            
            initial_decision = await asyncio.to_thread(route_query, request.query, chat_memory.get())
            if initial_decision == 'NEW_TOPIC':
                await user_memory_manager.reset_memory(session_id)
                was_reset = True
                decision = await asyncio.to_thread(route_query, request.query, [])
            else:
                decision = initial_decision

            if decision == 'RAG':
                # --- 1. สร้างแผนที่ติดตามที่มา (เหมือนกับใน /chat) ---
                source_lookup = {}
                vector_nodes = await vector_retriever.aretrieve(user_query)
                for node in vector_nodes:
                    source_lookup[node.id_] = 'vector_retriever'
                
                bm25_nodes = await bm25_retriever.aretrieve(user_query)
                for node in bm25_nodes:
                    if node.id_ in source_lookup:
                        source_lookup[node.id_] += ', bm25_retriever'
                    else:
                        source_lookup[node.id_] = 'bm25_retriever'
                
                # --- 2. รัน Chat Engine และ Stream คำตอบ ---
                chat_engine = CondensePlusContextChatEngine.from_defaults(retriever=retriever, llm=llm, memory=chat_memory, system_prompt=ROLE_INFO, node_postprocessors=[reranker])
                streaming_response = await chat_engine.astream_chat(user_query)
                
                # วนลูปเพื่อส่ง token แต่ละตัว
                async for token in streaming_response.async_response_gen():
                    full_response_text += token
                    yield f"data: {json.dumps({'type': 'token', 'content': token}, ensure_ascii=False)}\n\n"
                
                response_source = "RAG Hybrid Search"

                # --- 3. สร้าง reference และ chunk_preview (หลัง stream) ---
                source_nodes = streaming_response.source_nodes
                if source_nodes:
                    # ประมวลผล metadata ทั้งหมดเหมือนใน /chat
                    for n in source_nodes:
                        final_metadata = n.metadata.copy()
                        final_metadata['retrieval_source'] = source_lookup.get(n.id_, 'unknown_after_fusion')
                        
                        path_to_process = None
                        if 'file_path' in final_metadata:
                            path_to_process = final_metadata['file_path']
                        elif 'category' in final_metadata and 'file_name' in final_metadata:
                            try:
                                full_path_from_vector = os.path.join(final_metadata['category'], final_metadata['file_name'])
                                path_to_process = full_path_from_vector
                            except Exception:
                                path_to_process = final_metadata['category']
                            del final_metadata['category']

                        if path_to_process:
                            try:
                                final_metadata['file_path'] = os.path.relpath(path_to_process, PROJECT_ROOT_PATH)
                            except Exception:
                                final_metadata['file_path'] = path_to_process
                            
                            if isinstance(final_metadata.get('file_path'), str):
                                final_metadata['file_path'] = final_metadata['file_path'].replace("New_dopa_url/", "")
                        
                        # จัดลำดับคีย์
                        ordered_metadata = {}
                        key_order = [
                            "file_name", "file_path", "retrieval_source", "type", 
                            "document_title", "tags", "file_type", "file_size", 
                            "creation_date", "last_modified_date"
                        ]
                        for key in key_order:
                            if key in final_metadata:
                                ordered_metadata[key] = final_metadata[key]
                        for key, value in final_metadata.items():
                            if key not in ordered_metadata:
                                ordered_metadata[key] = value
                        
                        reference_path.append({"metadata": ordered_metadata, "score": float(n.score)})

                    # สร้าง chunk_preview
                    chunk_preview_data = [
                        { "rank": i + 1, "document_title": n.metadata.get('file_name', 'N/A'), "content": n.get_content() }
                        for i, n in enumerate(source_nodes)
                    ]

            elif decision == 'FAQ_LINK':
                # ... (ส่วนนี้เหมือนเดิม) ...
                response_source = "FAQ Link Retrieval"
                if not faq_retriever:
                    full_response_text = "ขออภัยค่ะ ขณะนี้ระบบฐานข้อมูลคู่มือบริการขัดข้องค่ะ"
                else:
                    initial_results = await faq_retriever.aretrieve(user_query)
                    if not initial_results:
                        full_response_text = "น้องปกป้องยังไม่พบข้อมูลที่ตรงกับคำถามของคุณค่ะ"
                    else:
                        reranked_results = reranker.postprocess_nodes(initial_results, query_bundle=QueryBundle(user_query))
                        response_parts = ["สำหรับคำถามของคุณนะคะ น้องปกป้องพบหัวข้อที่เกี่ยวข้องดังนี้ค่ะ:", ""]
                        ref_data = []
                        for node in reranked_results:
                            title, url = node.metadata.get('title', 'N/A'), node.metadata.get('url', 'N/A')
                            response_parts.append(f"• **{title}**")
                            response_parts.append(f"  ดูรายละเอียดเพิ่มเติมได้ที่: {url}")
                            ref_data.append({"title": title, "url": url, "score": float(node.score)})
                        full_response_text = "\n".join(response_parts)
                        reference_path = ref_data
                yield f"data: {json.dumps({'type': 'token', 'content': full_response_text}, ensure_ascii=False)}\n\n"

            elif decision == 'GENERAL_LLM':
                # ... (ส่วนนี้เหมือนเดิม) ...
                chat_engine = SimpleChatEngine.from_defaults(llm=llm, memory=chat_memory, system_prompt=GENERAL_ROLE_INFO)
                streaming_response = await chat_engine.astream_chat(user_query)
                async for token in streaming_response.async_response_gen():
                    full_response_text += token
                    yield f"data: {json.dumps({'type': 'token', 'content': token}, ensure_ascii=False)}\n\n"
                response_source, reference_path = "General LLM", []
            else:
                raise ValueError(f"Invalid final decision: {decision}")
        except Exception as e:
            status, full_response_text = "error", f"ขออภัยค่ะ เกิดข้อผิดพลาดในการประมวลผล: {e}"
            logger.error("Exception in /chat/stream endpoint", exc_info=True, extra={"extra": {"request_body": request.model_dump()}})
            yield f"data: {json.dumps({'type': 'error', 'content': full_response_text}, ensure_ascii=False)}\n\n"
        finally:
            elapsed = time.time() - start_time
            # --- 4. สร้าง Final Payload ที่มีข้อมูลครบถ้วน ---
            final_payload = {
                'type': 'final', 
                'session_id': session_id, 
                'query': user_query, 
                'decision': decision, 
                'response': full_response_text.strip(), 
                'source': response_source, 
                'reference': reference_path, 
                'chunk_preview': chunk_preview_data, # เพิ่ม chunk_preview
                'status': status, 
                'time_seconds': f"{elapsed:.3f}s"
            }
            yield f"data: {json.dumps(final_payload, ensure_ascii=False)}\n\n"
            
            # --- 5. บันทึก Log ---
            log_data = final_payload.copy()
            if 'type' in log_data: del log_data['type']
            log_data["memory_reset"] = "Done" if was_reset else "N/A"
            log_data["token_usage"] = {"prompt_tokens": token_counter.prompt_llm_token_count, "completion_tokens": token_counter.completion_llm_token_count, "total_tokens": token_counter.total_llm_token_count}
            if status == "error": logger.error("Streamed chat request failed", extra=log_data)
            else: logger.info("Streamed chat request processed", extra=log_data)
            
    return StreamingResponse(generate_stream_response(), media_type="text/event-stream")


@app.post("/search", response_class=JSONResponse)
async def diagnostic_search(request: SearchRequest) -> Dict[str, Any]:
    start_time = time.time()
    user_query = request.query
    final_response = {}
    log_data = {"endpoint": "/search", "session_id": request.session_id, "user_query": user_query}
    status = "completed"
    
    def format_nodes(nodes: List[NodeWithScore], preview_len=150):
        return [{"score": float(node.score), "metadata": node.metadata, "text_preview": node.text[:preview_len] + "..."} for node in nodes]
    
    def format_faq_nodes(nodes: List[NodeWithScore]):
        return [{"score": float(node.score), "title": node.metadata['title'], "url": node.metadata['url']} for node in nodes]

    try:
        token_counter.reset_counts()
        routing_decision = await asyncio.to_thread(route_query, user_query, [])
        log_data["routing_decision"] = routing_decision
        
        if routing_decision == 'RAG':
            vector_nodes = await vector_retriever.aretrieve(user_query)
            bm25_nodes = await bm25_retriever.aretrieve(user_query)
            fused_nodes = await retriever.aretrieve(user_query)
            reranked_nodes = reranker.postprocess_nodes(fused_nodes, query_bundle=QueryBundle(user_query))
            final_response = {
                "query": user_query,
                "queryAnalysis": {"routingDecision": routing_decision},
                "retrievalSteps": {
                    "1_vector_search_results": {"count": len(vector_nodes), "nodes": format_nodes(vector_nodes)},
                    "2_keyword_search_results_bm25": {"count": len(bm25_nodes), "nodes": format_nodes(bm25_nodes)},
                    "3_fused_results_rrf": {"count": len(fused_nodes), "nodes": format_nodes(fused_nodes)},
                    "4_final_reranked_results": {"count": len(reranked_nodes), "nodes": format_nodes(reranked_nodes, 250)},
                }
            }
        
        elif routing_decision == 'FAQ_LINK':
            if not faq_retriever:
                 final_response = {"query": user_query, "queryAnalysis": {"routingDecision": routing_decision, "notes": "FAQ Retriever is not available."}}
            else:
                initial_nodes = await faq_retriever.aretrieve(user_query)
                reranked_nodes = reranker.postprocess_nodes(initial_nodes, query_bundle=QueryBundle(user_query))
                final_response = {
                    "query": user_query,
                    "queryAnalysis": {"routingDecision": routing_decision},
                    "retrievalSteps": {
                        "1_initial_retrieval (Top 5 Titles)": {"count": len(initial_nodes), "nodes": format_faq_nodes(initial_nodes)},
                        "2_reranked_results (Top 3 Titles)": {"count": len(reranked_nodes), "nodes": format_faq_nodes(reranked_nodes)}
                    }
                }
        else:
             final_response = {"query": user_query, "queryAnalysis": {"routingDecision": routing_decision, "notes": "Query was not routed to RAG or FAQ."}}

    except Exception as e:
        status = "error"
        final_response = {"error": f"An error occurred during search: {e}"}
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

# ==============================================================================
# 6. RUN
# ==============================================================================
if __name__ == "__main__":
    uvicorn.run(
        "__main__:app",
        host="0.0.0.0",
        port=6667,
        reload=False # Set to False for production
    )