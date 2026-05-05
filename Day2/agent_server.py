
 # uvicorn agent_server:app --reload
 
 # python3 agent_client.py
 
 


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Optional
import logging
import time

app = FastAPI(title="DebugAgent A2A Service")

# -------------------------
# Logging (production must-have)
# -------------------------
logging.basicConfig(level=logging.INFO)

# -------------------------
# A2A Schema
# -------------------------
class A2ARequest(BaseModel):
    sender: str
    receiver: str
    problem: str
    context: Dict
    metadata: Dict


class A2AResponse(BaseModel):
    status: str
    solution: str
    handled_by: str
    metadata: Dict


# -------------------------
# Core Processing Logic
# -------------------------
def analyze_problem(context: Dict) -> str:
    cpu = context.get("cpu_usage", "0%")
    error_rate = context.get("error_rate", "0%")
    rps = context.get("requests_per_sec", 0)

    # Slightly smarter logic
    if cpu == "85%" and rps > 1000:
        return "Scale horizontally, introduce auto-scaling and load balancing"

    if error_rate == "5%":
        return "Check error logs, add retry logic and circuit breaker"

    return "Perform deep diagnostics with tracing and profiling"


# -------------------------
# API Endpoint
# -------------------------
@app.post("/process", response_model=A2AResponse)
async def process_request(request: A2ARequest):
    start_time = time.time()

    try:
        logging.info(f"Received request from {request.sender}")

        # Validation (important)
        if not request.problem:
            raise HTTPException(status_code=400, detail="Problem is required")

        solution = analyze_problem(request.context)

        response = A2AResponse(
            status="SUCCESS",
            solution=solution,
            handled_by="DebugAgent",
            metadata={
                "type": "A2A_RESPONSE",
                "version": "2.1",
                "latency_ms": int((time.time() - start_time) * 1000)
            }
        )

        logging.info("Processed successfully")

        return response

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))