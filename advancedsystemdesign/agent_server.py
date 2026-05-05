

from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from typing import Dict, Optional
import logging, time

app = FastAPI(title="DebugAgent A2A Service")
logging.basicConfig(level=logging.INFO)

API_KEY = "secret123"  # move to env in real systems

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

def analyze_problem(ctx: Dict) -> str:
    cpu = ctx.get("cpu_usage", "0%")
    err = ctx.get("error_rate", "0%")
    rps = ctx.get("requests_per_sec", 0)

    if cpu == "85%" and rps > 1000:
        return "Scale horizontally, introduce auto-scaling and load balancing"
    if err == "5%":
        return "Inspect error logs, add retries + circuit breaker"
    return "Run tracing and profiling"

@app.post("/process", response_model=A2AResponse)
async def process_request(
    request: A2ARequest,
    x_api_key: Optional[str] = Header(None)
):
    start = time.time()

    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    if not request.problem:
        raise HTTPException(status_code=400, detail="Problem required")

    try:
        solution = analyze_problem(request.context)
        latency = int((time.time() - start) * 1000)

        return A2AResponse(
            status="SUCCESS",
            solution=solution,
            handled_by="DebugAgent",
            metadata={
                "type": "A2A_RESPONSE",
                "version": "2.2",
                "latency_ms": latency
            }
        )
    except Exception as e:
        logging.exception("processing error")
        raise HTTPException(status_code=500, detail=str(e))