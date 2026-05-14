
# ============================================================
# observability_genai_pipeline.py
# ============================================================
#
# Production-Style GenAI Observability Pipeline
#
# Concepts Covered:
# ------------------------------------------------------------
# 1. Full Request Tracing
# 2. Token Usage Tracking
# 3. Latency Monitoring
# 4. Semantic Retrieval Tracing
# 5. Cache Hit Monitoring
# 6. LLM Route Tracking
# 7. Error Tracking
# 8. Cost Visibility
# 9. Production Debugging
#
# ============================================================

# ============================================================
# IMPORTS
# ============================================================

import os
import time
import uuid
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from openai import OpenAI

# ============================================================
# LOAD ENV
# ============================================================

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:

    raise ValueError(
        """
        OPENAI_API_KEY missing.

        Create .env:

        OPENAI_API_KEY=sk-xxxxxxxxxxxx
        """
    )

# ============================================================
# OPENAI CLIENT
# ============================================================

client = OpenAI(
    api_key=api_key
)

# ============================================================
# EMBEDDING MODEL
# ============================================================

print("\nLoading Embedding Model...\n")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# ============================================================
# KNOWLEDGE BASE
# ============================================================

knowledge_base = [

    {
        "title": "Kubernetes Scaling",

        "content": """
        Kubernetes scaling uses:
        - Horizontal Pod Autoscaler
        - Cluster Autoscaler
        - Load balancing
        """
    },

    {
        "title": "Kafka Consumer Lag",

        "content": """
        Kafka lag occurs when consumers
        cannot process messages quickly enough.
        """
    },

    {
        "title": "API Latency",

        "content": """
        API latency can be reduced using:
        - caching
        - indexing
        - load balancing
        """
    }
]

# ============================================================
# BUILD VECTOR STORE
# ============================================================

document_texts = [

    item["content"]

    for item in knowledge_base
]

document_embeddings = embedding_model.encode(
    document_texts
)

# ============================================================
# SEMANTIC CACHE
# ============================================================

semantic_cache = {}

# ============================================================
# TRACE MODEL
# ============================================================

@dataclass
class TraceLog:

    trace_id: str

    query: str

    route: str

    cache_hit: bool

    retrieved_docs: List[str]

    retrieval_scores: List[float]

    embedding_latency_ms: float

    retrieval_latency_ms: float

    llm_latency_ms: float

    total_latency_ms: float

    prompt_tokens: int

    completion_tokens: int

    total_tokens: int

    estimated_cost_usd: float

    response: str

# ============================================================
# SIMPLE TOKEN ESTIMATOR
# ============================================================

def estimate_cost(total_tokens):

    # Approx rough estimate for GPT-4.1-mini

    estimated_price_per_1k = 0.0008

    return round(
        (total_tokens / 1000) * estimated_price_per_1k,
        6
    )

# ============================================================
# QUERY CLASSIFIER
# ============================================================

def classify_query(query):

    simple_keywords = [

        "what is",

        "define",

        "meaning"
    ]

    lower_query = query.lower()

    for keyword in simple_keywords:

        if keyword in lower_query:

            return "simple"

    return "complex"

# ============================================================
# CACHE SEARCH
# ============================================================

def semantic_cache_lookup(query):

    if not semantic_cache:

        return None

    query_embedding = embedding_model.encode([query])

    cache_queries = list(semantic_cache.keys())

    cache_embeddings = embedding_model.encode(
        cache_queries
    )

    similarities = cosine_similarity(
        query_embedding,
        cache_embeddings
    )[0]

    best_idx = np.argmax(similarities)

    best_score = similarities[best_idx]

    print(f"\nCache Similarity: {best_score:.4f}")

    if best_score > 0.90:

        matched_query = cache_queries[best_idx]

        return semantic_cache[matched_query]

    return None

# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_documents(query):

    retrieval_start = time.time()

    query_embedding = embedding_model.encode([query])

    similarities = cosine_similarity(
        query_embedding,
        document_embeddings
    )[0]

    ranked_indices = np.argsort(similarities)[::-1]

    results = []

    for idx in ranked_indices[:2]:

        results.append({

            "title": knowledge_base[idx]["title"],

            "content": knowledge_base[idx]["content"],

            "score": float(similarities[idx])
        })

    retrieval_latency = (
        time.time() - retrieval_start
    ) * 1000

    return results, retrieval_latency

# ============================================================
# GPT RESPONSE
# ============================================================

def generate_response(query, docs):

    llm_start = time.time()

    context = "\n".join([

        doc["content"]

        for doc in docs
    ])

    prompt = f"""
    Answer infrastructure question
    using ONLY context.

    USER QUESTION:
    {query}

    CONTEXT:
    {context}

    Keep response concise.
    """

    response = client.chat.completions.create(

        model="gpt-4.1-mini",

        messages=[

            {
                "role": "system",

                "content": """
                You are an infrastructure AI assistant.
                """
            },

            {
                "role": "user",

                "content": prompt
            }
        ],

        temperature=0,

        max_tokens=120
    )

    llm_latency = (
        time.time() - llm_start
    ) * 1000

    usage = response.usage

    return {

        "answer":
            response.choices[0].message.content,

        "prompt_tokens":
            usage.prompt_tokens,

        "completion_tokens":
            usage.completion_tokens,

        "total_tokens":
            usage.total_tokens,

        "llm_latency_ms":
            llm_latency
    }

# ============================================================
# OBSERVABILITY PIPELINE
# ============================================================

def observability_pipeline(query):

    total_start = time.time()

    trace_id = str(uuid.uuid4())

    print("\n================================================")
    print("PRODUCTION GENAI OBSERVABILITY PIPELINE")
    print("================================================")

    print(f"\nTrace ID: {trace_id}")

    # --------------------------------------------------------
    # STEP 1: CACHE CHECK
    # --------------------------------------------------------

    print("\nSTEP 1: Semantic Cache")

    cached = semantic_cache_lookup(query)

    if cached:

        total_latency = (
            time.time() - total_start
        ) * 1000

        trace = TraceLog(

            trace_id=trace_id,

            query=query,

            route="CACHE",

            cache_hit=True,

            retrieved_docs=[],

            retrieval_scores=[],

            embedding_latency_ms=0,

            retrieval_latency_ms=0,

            llm_latency_ms=0,

            total_latency_ms=total_latency,

            prompt_tokens=0,

            completion_tokens=0,

            total_tokens=0,

            estimated_cost_usd=0,

            response=cached
        )

        return trace

    # --------------------------------------------------------
    # STEP 2: ROUTING
    # --------------------------------------------------------

    print("\nSTEP 2: Query Routing")

    route = classify_query(query)

    print(f"\nRoute: {route}")

    # --------------------------------------------------------
    # STEP 3: SMALL MODEL ROUTE
    # --------------------------------------------------------

    if route == "simple":

        answer = "Simple infrastructure query answered."

        semantic_cache[query] = answer

        total_latency = (
            time.time() - total_start
        ) * 1000

        trace = TraceLog(

            trace_id=trace_id,

            query=query,

            route="SMALL_MODEL",

            cache_hit=False,

            retrieved_docs=[],

            retrieval_scores=[],

            embedding_latency_ms=0,

            retrieval_latency_ms=0,

            llm_latency_ms=0,

            total_latency_ms=total_latency,

            prompt_tokens=0,

            completion_tokens=0,

            total_tokens=0,

            estimated_cost_usd=0,

            response=answer
        )

        return trace

    # --------------------------------------------------------
    # STEP 4: RETRIEVAL
    # --------------------------------------------------------

    print("\nSTEP 4: Retrieval")

    docs, retrieval_latency = retrieve_documents(query)

    for doc in docs:

        print(
            f"""
            Retrieved:
            {doc['title']}

            Score:
            {doc['score']:.4f}
            """
        )

    # --------------------------------------------------------
    # STEP 5: GPT
    # --------------------------------------------------------

    print("\nSTEP 5: GPT-4 Reasoning")

    llm_result = generate_response(
        query,
        docs
    )

    semantic_cache[query] = llm_result["answer"]

    total_latency = (
        time.time() - total_start
    ) * 1000

    # --------------------------------------------------------
    # TRACE OBJECT
    # --------------------------------------------------------

    trace = TraceLog(

        trace_id=trace_id,

        query=query,

        route="GPT4",

        cache_hit=False,

        retrieved_docs=[
            doc["title"]
            for doc in docs
        ],

        retrieval_scores=[
            doc["score"]
            for doc in docs
        ],

        embedding_latency_ms=0,

        retrieval_latency_ms=retrieval_latency,

        llm_latency_ms=llm_result["llm_latency_ms"],

        total_latency_ms=total_latency,

        prompt_tokens=llm_result["prompt_tokens"],

        completion_tokens=llm_result["completion_tokens"],

        total_tokens=llm_result["total_tokens"],

        estimated_cost_usd=estimate_cost(
            llm_result["total_tokens"]
        ),

        response=llm_result["answer"]
    )

    return trace

# ============================================================
# TRACE VISUALIZATION
# ============================================================

def print_trace(trace):

    print("\n================================================")
    print("TRACE OBSERVABILITY REPORT")
    print("================================================")

    print(f"\nTrace ID:\n{trace.trace_id}")

    print(f"\nQuery:\n{trace.query}")

    print(f"\nRoute:\n{trace.route}")

    print(f"\nCache Hit:\n{trace.cache_hit}")

    print(f"""
    \nRetrieved Docs:
    {trace.retrieved_docs}
    """)

    print(f"""
    \nRetrieval Scores:
    {trace.retrieval_scores}
    """)

    print(f"""
    \nRetrieval Latency:
    {trace.retrieval_latency_ms:.2f} ms
    """)

    print(f"""
    \nLLM Latency:
    {trace.llm_latency_ms:.2f} ms
    """)

    print(f"""
    \nTotal Latency:
    {trace.total_latency_ms:.2f} ms
    """)

    print(f"""
    \nPrompt Tokens:
    {trace.prompt_tokens}
    """)

    print(f"""
    \nCompletion Tokens:
    {trace.completion_tokens}
    """)

    print(f"""
    \nTotal Tokens:
    {trace.total_tokens}
    """)

    print(f"""
    \nEstimated Cost:
    ${trace.estimated_cost_usd}
    """)

    print(f"\nFinal Response:\n{trace.response}")

# ============================================================
# TEST CASES
# ============================================================

if __name__ == "__main__":

    queries = [

        "What is Kubernetes",

        "How can we reduce API latency?",

        "How can we reduce API latency?"
    ]

    for query in queries:

        trace = observability_pipeline(query)

        print_trace(trace)