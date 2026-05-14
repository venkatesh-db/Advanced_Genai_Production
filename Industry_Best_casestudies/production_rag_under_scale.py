
# ============================================================
# production_rag_under_scale.py
# ============================================================
#
# Production-Style RAG System Under Scale
#
# Scenario:
# ------------------------------------------------------------
# 50 Million Documents
# 100K Concurrent Users
#
# Problems:
# ------------------------------------------------------------
# 1. Retrieval latency high
# 2. Hallucinations increasing
# 3. Token costs exploding
# 4. Irrelevant context retrieved
#
# This code demonstrates:
# ------------------------------------------------------------
# 1. Semantic Cache
# 2. Hybrid Retrieval
# 3. Reranking
# 4. Context Compression
# 5. GPT Routing
# 6. Observability
# 7. Cost Tracking
# 8. Hallucination Reduction
#
# ============================================================

# ============================================================
# IMPORTS
# ============================================================

import os
import time
import uuid
from typing import List, Dict
from dataclasses import dataclass

import numpy as np

from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

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

        Create .env:

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
# EMBEDDING MODEL
# ============================================================

print("\nLoading Embedding Model...\n")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# ============================================================
# LARGE KNOWLEDGE BASE
# ============================================================

knowledge_base = [

    {
        "title": "Kubernetes Scaling",

        "domain": "kubernetes",

        "content": """
        Kubernetes scaling uses:
        Horizontal Pod Autoscaler,
        Cluster Autoscaler,
        and load balancing.
        """
    },

    {
        "title": "Kafka Consumer Lag",

        "domain": "kafka",

        "content": """
        Kafka lag occurs when consumers
        cannot process messages quickly enough.
        """
    },

    {
        "title": "API Latency",

        "domain": "api",

        "content": """
        API latency can be reduced using:
        caching,
        indexing,
        and load balancing.
        """
    },

    {
        "title": "Database Bottleneck",

        "domain": "database",

        "content": """
        Database bottlenecks occur because of:
        slow queries,
        missing indexes,
        and lock contention.
        """
    },

    {
        "title": "Network Packet Loss",

        "domain": "network",

        "content": """
        Packet loss happens because of:
        congestion,
        routing instability,
        or overloaded network devices.
        """
    }
]

# ============================================================
# VECTOR STORE
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
class ProductionTrace:

    trace_id: str

    query: str

    cache_hit: bool

    routed_model: str

    retrieved_docs: List[str]

    reranked_docs: List[str]

    retrieval_scores: List[float]

    rerank_scores: List[float]

    total_tokens: int

    estimated_cost: float

    retrieval_latency_ms: float

    llm_latency_ms: float

    total_latency_ms: float

    final_answer: str

# ============================================================
# TOKEN COST ESTIMATION
# ============================================================

def estimate_cost(tokens):

    # Approx estimate

    return round(
        (tokens / 1000) * 0.0008,
        6
    )

# ============================================================
# QUERY ROUTER
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
# DOMAIN DETECTION
# ============================================================

def detect_domain(query):

    query = query.lower()

    if "kafka" in query:
        return "kafka"

    if "kubernetes" in query:
        return "kubernetes"

    if "latency" in query:
        return "api"

    if "database" in query:
        return "database"

    if "network" in query:
        return "network"

    return "general"

# ============================================================
# SEMANTIC CACHE
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

    print(f"\nSemantic Cache Score: {best_score:.4f}")

    if best_score > 0.90:

        matched_query = cache_queries[best_idx]

        return semantic_cache[matched_query]

    return None

# ============================================================
# HYBRID RETRIEVAL
# ============================================================

def hybrid_retrieval(query):

    retrieval_start = time.time()

    domain = detect_domain(query)

    print(f"\nDetected Domain: {domain}")

    # --------------------------------------------------------
    # METADATA FILTERING
    # --------------------------------------------------------

    filtered_docs = []

    filtered_embeddings = []

    for idx, doc in enumerate(knowledge_base):

        if domain == "general" or doc["domain"] == domain:

            filtered_docs.append(doc)

            filtered_embeddings.append(
                document_embeddings[idx]
            )

    filtered_embeddings = np.array(filtered_embeddings)

    # --------------------------------------------------------
    # VECTOR SEARCH
    # --------------------------------------------------------

    query_embedding = embedding_model.encode([query])

    similarities = cosine_similarity(
        query_embedding,
        filtered_embeddings
    )[0]

    ranked_indices = np.argsort(similarities)[::-1]

    results = []

    for idx in ranked_indices:

        score = float(similarities[idx])

        if score > 0.30:

            results.append({

                "title": filtered_docs[idx]["title"],

                "content": filtered_docs[idx]["content"],

                "score": score
            })

    retrieval_latency = (
        time.time() - retrieval_start
    ) * 1000

    return results, retrieval_latency

# ============================================================
# RERANKING
# ============================================================

def rerank_documents(query, docs):

    reranked = []

    for doc in docs:

        score = doc["score"]

        # Boost if keyword overlap exists

        overlap = 0

        for word in query.lower().split():

            if word in doc["content"].lower():

                overlap += 1

        rerank_score = score + (overlap * 0.05)

        reranked.append({

            "title": doc["title"],

            "content": doc["content"],

            "rerank_score": rerank_score
        })

    reranked = sorted(
        reranked,
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return reranked[:2]

# ============================================================
# CONTEXT COMPRESSION
# ============================================================

def compress_context(docs):

    compressed = []

    for doc in docs:

        compressed.append(
            doc["content"][:180]
        )

    return "\n".join(compressed)

# ============================================================
# SMALL MODEL
# ============================================================

def small_model_answer(query):

    return "Simple infrastructure query answered locally."

# ============================================================
# GPT RESPONSE
# ============================================================

def gpt_answer(query, context):

    llm_start = time.time()

    prompt = f"""
    Answer ONLY using provided context.

    USER QUESTION:
    {query}

    CONTEXT:
    {context}

    Keep answer concise.
    Avoid hallucinations.
    """

    response = client.chat.completions.create(

        model="gpt-4.1-mini",

        messages=[

            {
                "role": "system",

                "content": """
                You are a production infrastructure AI assistant.

                Never invent infrastructure details.
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

    return {

        "answer":
            response.choices[0].message.content,

        "tokens":
            response.usage.total_tokens,

        "latency":
            llm_latency
    }

# ============================================================
# MAIN PRODUCTION PIPELINE
# ============================================================

def production_rag_pipeline(query):

    total_start = time.time()

    trace_id = str(uuid.uuid4())

    print("\n================================================")
    print("PRODUCTION RAG UNDER SCALE")
    print("================================================")

    print(f"\nTrace ID: {trace_id}")

    # --------------------------------------------------------
    # STEP 1: CACHE
    # --------------------------------------------------------

    print("\nSTEP 1: Semantic Cache")

    cached = semantic_cache_lookup(query)

    if cached:

        total_latency = (
            time.time() - total_start
        ) * 1000

        return ProductionTrace(

            trace_id=trace_id,

            query=query,

            cache_hit=True,

            routed_model="CACHE",

            retrieved_docs=[],

            reranked_docs=[],

            retrieval_scores=[],

            rerank_scores=[],

            total_tokens=0,

            estimated_cost=0,

            retrieval_latency_ms=0,

            llm_latency_ms=0,

            total_latency_ms=total_latency,

            final_answer=cached
        )

    # --------------------------------------------------------
    # STEP 2: ROUTING
    # --------------------------------------------------------

    print("\nSTEP 2: Query Classification")

    route = classify_query(query)

    print(f"\nQuery Type: {route}")

    # --------------------------------------------------------
    # STEP 3: SMALL MODEL ROUTE
    # --------------------------------------------------------

    if route == "simple":

        answer = small_model_answer(query)

        semantic_cache[query] = answer

        total_latency = (
            time.time() - total_start
        ) * 1000

        return ProductionTrace(

            trace_id=trace_id,

            query=query,

            cache_hit=False,

            routed_model="SMALL_MODEL",

            retrieved_docs=[],

            reranked_docs=[],

            retrieval_scores=[],

            rerank_scores=[],

            total_tokens=0,

            estimated_cost=0,

            retrieval_latency_ms=0,

            llm_latency_ms=0,

            total_latency_ms=total_latency,

            final_answer=answer
        )

    # --------------------------------------------------------
    # STEP 4: HYBRID RETRIEVAL
    # --------------------------------------------------------

    print("\nSTEP 4: Hybrid Retrieval")

    retrieved_docs, retrieval_latency = hybrid_retrieval(
        query
    )

    for doc in retrieved_docs:

        print(
            f"""
            Retrieved:
            {doc['title']}

            Score:
            {doc['score']:.4f}
            """
        )

    # --------------------------------------------------------
    # STEP 5: RERANKING
    # --------------------------------------------------------

    print("\nSTEP 5: Reranking")

    reranked_docs = rerank_documents(
        query,
        retrieved_docs
    )

    for doc in reranked_docs:

        print(
            f"""
            Reranked:
            {doc['title']}

            Rerank Score:
            {doc['rerank_score']:.4f}
            """
        )

    # --------------------------------------------------------
    # STEP 6: CONTEXT COMPRESSION
    # --------------------------------------------------------

    print("\nSTEP 6: Context Compression")

    compressed_context = compress_context(
        reranked_docs
    )

    # --------------------------------------------------------
    # STEP 7: GPT
    # --------------------------------------------------------

    print("\nSTEP 7: GPT Reasoning")

    gpt_result = gpt_answer(

        query,

        compressed_context
    )

    semantic_cache[query] = gpt_result["answer"]

    total_latency = (
        time.time() - total_start
    ) * 1000

    # --------------------------------------------------------
    # FINAL TRACE
    # --------------------------------------------------------

    return ProductionTrace(

        trace_id=trace_id,

        query=query,

        cache_hit=False,

        routed_model="GPT4",

        retrieved_docs=[
            doc["title"]
            for doc in retrieved_docs
        ],

        reranked_docs=[
            doc["title"]
            for doc in reranked_docs
        ],

        retrieval_scores=[
            doc["score"]
            for doc in retrieved_docs
        ],

        rerank_scores=[
            doc["rerank_score"]
            for doc in reranked_docs
        ],

        total_tokens=gpt_result["tokens"],

        estimated_cost=estimate_cost(
            gpt_result["tokens"]
        ),

        retrieval_latency_ms=retrieval_latency,

        llm_latency_ms=gpt_result["latency"],

        total_latency_ms=total_latency,

        final_answer=gpt_result["answer"]
    )

# ============================================================
# TRACE REPORT
# ============================================================

def print_trace(trace):

    print("\n================================================")
    print("PRODUCTION TRACE REPORT")
    print("================================================")

    print(f"\nTrace ID:\n{trace.trace_id}")

    print(f"\nQuery:\n{trace.query}")

    print(f"\nCache Hit:\n{trace.cache_hit}")

    print(f"\nModel Route:\n{trace.routed_model}")

    print(f"""
    \nRetrieved Docs:
    {trace.retrieved_docs}
    """)

    print(f"""
    \nReranked Docs:
    {trace.reranked_docs}
    """)

    print(f"""
    \nRetrieval Scores:
    {trace.retrieval_scores}
    """)

    print(f"""
    \nRerank Scores:
    {trace.rerank_scores}
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
    \nTotal Tokens:
    {trace.total_tokens}
    """)

    print(f"""
    \nEstimated Cost:
    ${trace.estimated_cost}
    """)

    print(f"\nFinal Answer:\n{trace.final_answer}")

# ============================================================
# TEST CASES
# ============================================================

if __name__ == "__main__":

    queries = [

        "What is Kubernetes",

        "How can we reduce API latency in distributed systems?",

        "How can we reduce API latency in distributed systems?"
    ]

    for query in queries:

        trace = production_rag_pipeline(query)

        print_trace(trace)