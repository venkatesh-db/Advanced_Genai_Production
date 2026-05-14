# ============================================================
# production_streaming_ai_server.py
# ============================================================
#
# Production Streaming GenAI Server
#
# Features:
# ------------------------------------------------------------
# 1. Token Streaming
# 2. Async Streaming
# 3. WebSocket Handling
# 4. Real-Time AI UX
# 5. Concurrent Connections
# 6. Observability
# 7. Connection Management
# 8. Streaming Metrics
#
# Production Use Cases:
# ------------------------------------------------------------
# - AI copilots
# - chat systems
# - realtime AI UX
# - coding assistants
# - infrastructure copilots
#
# ============================================================

# ============================================================
# IMPORTS
# ============================================================

import os
import json
import uuid
import time
import asyncio

from typing import Dict
from dataclasses import dataclass

from dotenv import load_dotenv

from fastapi import FastAPI
from fastapi import WebSocket
from fastapi import WebSocketDisconnect

from openai import AsyncOpenAI

import uvicorn

# ============================================================
# ENV CONFIG
# ============================================================

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:

    raise ValueError(
        """
        OPENAI_API_KEY missing.

        Create .env:

        OPENAI_API_KEY=sk-xxxxxxxx
        """
    )

# ============================================================
# OPENAI CLIENT
# ============================================================

client = AsyncOpenAI(
    api_key=api_key
)

# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI()

# ============================================================
# ACTIVE CONNECTIONS
# ============================================================

active_connections = {}

# ============================================================
# OBSERVABILITY MODEL
# ============================================================

@dataclass
class StreamTrace:

    trace_id: str

    client_id: str

    query: str

    total_tokens_streamed: int = 0

    stream_latency_ms: float = 0

    connection_start_time: float = 0

# ============================================================
# CONNECTION MANAGER
# ============================================================

class ConnectionManager:

    def __init__(self):

        self.active_connections = {}

    async def connect(

        self,

        websocket: WebSocket,

        client_id: str
    ):

        await websocket.accept()

        self.active_connections[client_id] = websocket

        print(f"""
        CONNECTED:
        {client_id}
        """)

    def disconnect(self, client_id: str):

        if client_id in self.active_connections:

            del self.active_connections[client_id]

            print(f"""
            DISCONNECTED:
            {client_id}
            """)

    async def send_json(

        self,

        client_id: str,

        payload: Dict
    ):

        websocket = self.active_connections.get(client_id)

        if websocket:

            await websocket.send_text(
                json.dumps(payload)
            )

# ============================================================
# MANAGER INSTANCE
# ============================================================

manager = ConnectionManager()

# ============================================================
# STREAMING LLM
# ============================================================

async def stream_llm_response(

    trace: StreamTrace,

    client_id: str
):

    print("""
    STARTING TOKEN STREAM...
    """)

    start = time.time()

    # --------------------------------------------------------
    # STREAMING REQUEST
    # --------------------------------------------------------

    stream = await client.chat.completions.create(

        model="gpt-4.1-mini",

        messages=[

            {
                "role": "system",

                "content": """
                You are a production AI copilot.

                Keep responses concise.
                """
            },

            {
                "role": "user",

                "content": trace.query
            }
        ],

        temperature=0,

        stream=True
    )

    complete_response = ""

    # --------------------------------------------------------
    # TOKEN STREAMING LOOP
    # --------------------------------------------------------

    async for chunk in stream:

        delta = chunk.choices[0].delta.content

        if delta:

            complete_response += delta

            trace.total_tokens_streamed += 1

            payload = {

                "type": "token",

                "trace_id": trace.trace_id,

                "token": delta
            }

            await manager.send_json(

                client_id,

                payload
            )

    # --------------------------------------------------------
    # FINAL METRICS
    # --------------------------------------------------------

    trace.stream_latency_ms = (
        time.time() - start
    ) * 1000

    print(f"""
    STREAM COMPLETE

    Trace ID:
    {trace.trace_id}

    Tokens Streamed:
    {trace.total_tokens_streamed}

    Stream Latency:
    {trace.stream_latency_ms:.2f} ms
    """)

    # --------------------------------------------------------
    # FINAL EVENT
    # --------------------------------------------------------

    await manager.send_json(

        client_id,

        {

            "type": "complete",

            "trace_id": trace.trace_id,

            "latency_ms": trace.stream_latency_ms,

            "tokens_streamed":
                trace.total_tokens_streamed,

            "final_response":
                complete_response
        }
    )

# ============================================================
# WEBSOCKET ENDPOINT
# ============================================================

@app.websocket("/ws/{client_id}")

async def websocket_endpoint(

    websocket: WebSocket,

    client_id: str
):

    await manager.connect(

        websocket,

        client_id
    )

    try:

        while True:

            # ------------------------------------------------
            # RECEIVE CLIENT MESSAGE
            # ------------------------------------------------

            raw_message = await websocket.receive_text()

            data = json.loads(raw_message)

            user_query = data.get("query")

            print(f"""
            RECEIVED QUERY

            Client:
            {client_id}

            Query:
            {user_query}
            """)

            # ------------------------------------------------
            # CREATE TRACE
            # ------------------------------------------------

            trace = StreamTrace(

                trace_id=str(uuid.uuid4()),

                client_id=client_id,

                query=user_query,

                connection_start_time=time.time()
            )

            # ------------------------------------------------
            # ACKNOWLEDGEMENT
            # ------------------------------------------------

            await manager.send_json(

                client_id,

                {

                    "type": "start",

                    "trace_id": trace.trace_id,

                    "message":
                        "Streaming started"
                }
            )

            # ------------------------------------------------
            # STREAM TOKENS
            # ------------------------------------------------

            await stream_llm_response(

                trace,

                client_id
            )

    except WebSocketDisconnect:

        manager.disconnect(client_id)

# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")

async def health():

    return {

        "status": "healthy",

        "active_connections":
            len(manager.active_connections)
    }

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("""
    ==================================================
    PRODUCTION STREAMING AI SERVER
    ==================================================

    WebSocket Endpoint:
    ws://localhost:8000/ws/{client_id}

    Health Endpoint:
    http://localhost:8000/health
    """)

    uvicorn.run(

        "production_streaming_ai_server:app",

        host="0.0.0.0",

        port=8000,

        reload=True
    )
    
    
    # run server
    # run client from test_client.py to see streaming in action