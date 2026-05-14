
# ============================================================
# production_multi_agent_stabilizer.py
# ============================================================
#
# Production Multi-Agent Stabilization System
#
# Scenario:
# ------------------------------------------------------------
# planning agent
# retrieval agent
# coding agent
# validation agent
#
# Problems:
# ------------------------------------------------------------
# 1. Infinite loops
# 2. Token explosion
# 3. Inconsistent reasoning
# 4. High latency
#
# This implementation demonstrates:
# ------------------------------------------------------------
# 1. Central Orchestrator
# 2. Shared State Management
# 3. Iteration Limits
# 4. Token Budget Control
# 5. Agent Routing
# 6. Context Compression
# 7. Validation Layer
# 8. Observability / Tracing
# 9. Failure Handling
#
# ============================================================

# ============================================================
# IMPORTS
# ============================================================

import os
import time
import uuid
from typing import Dict, List
from dataclasses import dataclass, field

from dotenv import load_dotenv
from openai import OpenAI

# ============================================================
# ENV CONFIG
# ============================================================

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:

    raise ValueError(
        """
        OPENAI_API_KEY missing.

        Create .env file:

        OPENAI_API_KEY=sk-xxxxxxxx
        """
    )

# ============================================================
# OPENAI CLIENT
# ============================================================

client = OpenAI(
    api_key=api_key
)

# ============================================================
# SYSTEM CONFIG
# ============================================================

MAX_ITERATIONS = 4

MAX_TOTAL_TOKENS = 800

MAX_CONTEXT_CHARS = 500

# ============================================================
# SHARED STATE
# ============================================================

@dataclass
class AgentState:

    trace_id: str

    user_query: str

    current_plan: str = ""

    retrieved_context: str = ""

    generated_code: str = ""

    validation_result: str = ""

    iteration_count: int = 0

    total_tokens: int = 0

    total_latency_ms: float = 0

    agent_history: List[str] = field(default_factory=list)

    final_answer: str = ""

# ============================================================
# OBSERVABILITY TRACE
# ============================================================

def log_step(agent_name, message):

    print(f"""
    ------------------------------------------------
    AGENT: {agent_name}
    ------------------------------------------------
    {message}
    """)

# ============================================================
# TOKEN BUDGET CHECK
# ============================================================

def token_budget_exceeded(state):

    return state.total_tokens >= MAX_TOTAL_TOKENS

# ============================================================
# CONTEXT COMPRESSION
# ============================================================

def compress_context(text):

    if len(text) > MAX_CONTEXT_CHARS:

        return text[:MAX_CONTEXT_CHARS]

    return text

# ============================================================
# GPT CALL WRAPPER
# ============================================================

def call_llm(system_prompt, user_prompt):

    start = time.time()

    response = client.chat.completions.create(

        model="gpt-4.1-mini",

        messages=[

            {
                "role": "system",

                "content": system_prompt
            },

            {
                "role": "user",

                "content": user_prompt
            }
        ],

        temperature=0,

        max_tokens=150
    )

    latency = (
        time.time() - start
    ) * 1000

    content = response.choices[0].message.content

    tokens = response.usage.total_tokens

    return {

        "content": content,

        "tokens": tokens,

        "latency": latency
    }

# ============================================================
# PLANNING AGENT
# ============================================================

def planning_agent(state):

    log_step(
        "PLANNING_AGENT",
        "Generating execution plan"
    )

    prompt = f"""
    Create concise execution plan.

    USER REQUEST:
    {state.user_query}

    Keep response short.
    """

    result = call_llm(

        system_prompt="""
        You are a planning agent.

        Avoid unnecessary reasoning.
        """,

        user_prompt=prompt
    )

    state.current_plan = result["content"]

    state.total_tokens += result["tokens"]

    state.total_latency_ms += result["latency"]

    state.agent_history.append("planning_agent")

# ============================================================
# RETRIEVAL AGENT
# ============================================================

def retrieval_agent(state):

    log_step(
        "RETRIEVAL_AGENT",
        "Retrieving context"
    )

    fake_context = """
    API latency optimization techniques:
    - caching
    - indexing
    - load balancing
    - async processing
    """

    compressed = compress_context(fake_context)

    state.retrieved_context = compressed

    state.agent_history.append("retrieval_agent")

# ============================================================
# CODING AGENT
# ============================================================

def coding_agent(state):

    log_step(
        "CODING_AGENT",
        "Generating implementation"
    )

    prompt = f"""
    PLAN:
    {state.current_plan}

    CONTEXT:
    {state.retrieved_context}

    Generate concise Python solution.
    """

    result = call_llm(

        system_prompt="""
        You are a coding agent.

        Generate minimal production-safe code.
        """,

        user_prompt=prompt
    )

    state.generated_code = result["content"]

    state.total_tokens += result["tokens"]

    state.total_latency_ms += result["latency"]

    state.agent_history.append("coding_agent")

# ============================================================
# VALIDATION AGENT
# ============================================================

def validation_agent(state):

    log_step(
        "VALIDATION_AGENT",
        "Validating generated code"
    )

    prompt = f"""
    Validate this code:

    {state.generated_code}

    Check:
    - correctness
    - infinite loops
    - unnecessary complexity
    """

    result = call_llm(

        system_prompt="""
        You are a validation agent.

        Reject unsafe or unstable code.
        """,

        user_prompt=prompt
    )

    state.validation_result = result["content"]

    state.total_tokens += result["tokens"]

    state.total_latency_ms += result["latency"]

    state.agent_history.append("validation_agent")

# ============================================================
# VALIDATION CHECK
# ============================================================

def validation_passed(state):

    validation = state.validation_result.lower()

    bad_patterns = [

        "infinite loop",

        "unsafe",

        "critical issue"
    ]

    for pattern in bad_patterns:

        if pattern in validation:

            return False

    return True

# ============================================================
# ORCHESTRATOR
# ============================================================

def orchestrator(state):

    log_step(
        "ORCHESTRATOR",
        "Starting multi-agent workflow"
    )

    while True:

        # ----------------------------------------------------
        # ITERATION CONTROL
        # ----------------------------------------------------

        if state.iteration_count >= MAX_ITERATIONS:

            state.final_answer = """
            Workflow stopped:
            max iterations exceeded.
            """

            break

        # ----------------------------------------------------
        # TOKEN BUDGET CONTROL
        # ----------------------------------------------------

        if token_budget_exceeded(state):

            state.final_answer = """
            Workflow stopped:
            token budget exceeded.
            """

            break

        # ----------------------------------------------------
        # EXECUTION FLOW
        # ----------------------------------------------------

        planning_agent(state)

        retrieval_agent(state)

        coding_agent(state)

        validation_agent(state)

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        if validation_passed(state):

            state.final_answer = """
            Multi-agent workflow completed successfully.
            """

            break

        log_step(
            "ORCHESTRATOR",
            "Validation failed. Retrying..."
        )

        state.iteration_count += 1

# ============================================================
# TRACE REPORT
# ============================================================

def print_trace(state):

    print("\n================================================")
    print("PRODUCTION MULTI-AGENT TRACE")
    print("================================================")

    print(f"\nTrace ID:\n{state.trace_id}")

    print(f"\nUser Query:\n{state.user_query}")

    print(f"\nIterations:\n{state.iteration_count}")

    print(f"""
    \nAgent Execution History:
    {state.agent_history}
    """)

    print(f"""
    \nTotal Tokens:
    {state.total_tokens}
    """)

    print(f"""
    \nTotal Latency:
    {state.total_latency_ms:.2f} ms
    """)

    print(f"""
    \nGenerated Plan:
    {state.current_plan}
    """)

    print(f"""
    \nRetrieved Context:
    {state.retrieved_context}
    """)

    print(f"""
    \nGenerated Code:
    {state.generated_code}
    """)

    print(f"""
    \nValidation Result:
    {state.validation_result}
    """)

    print(f"""
    \nFinal Status:
    {state.final_answer}
    """)

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    query = """
    Build scalable API latency optimization module
    for distributed infrastructure systems.
    """

    state = AgentState(

        trace_id=str(uuid.uuid4()),

        user_query=query
    )

    orchestrator(state)

    print_trace(state)