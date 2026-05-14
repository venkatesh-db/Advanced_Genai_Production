
# ============================================================
# token_cost_optimized_genai.py
# ============================================================
#
# Production-Style GenAI Cost Optimization System
#
# Scenario:
# Enterprise AI assistant costs $250K/month
#
# Problems:
# ------------------------------------------------------------
# 1. Every request calls GPT-4
# 2. Long prompts
# 3. Repeated queries
# 4. Unnecessary Chain-of-Thought
#
# This implementation demonstrates:
# ------------------------------------------------------------
# 1. Semantic Cache
# 2. Small-vs-Large Model Routing
# 3. Prompt Compression
# 4. Query Classification
# 5. Retrieval-First Architecture
# 6. Cost Optimization Strategy
#
# ============================================================

# ============================================================
# IMPORTS
# ============================================================

import os
import hashlib
from typing import Dict, List
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
# SEMANTIC CACHE
# ============================================================

semantic_cache = {}

# ============================================================
# KNOWLEDGE BASE
# ============================================================

knowledge_base = [

    {
        "id": 1,

        "title": "Kubernetes Scaling",

        "content": """
        Kubernetes scaling can be achieved using:
        - Horizontal Pod Autoscaler
        - Cluster Autoscaler
        - Load balancing
        """
    },

    {
        "id": 2,

        "title": "Kafka Consumer Lag",

        "content": """
        Kafka consumer lag occurs when consumers
        cannot process messages fast enough.
        """
    },

    {
        "id": 3,

        "title": "API Latency",

        "content": """
        API latency can be reduced by:
        - caching
        - load balancing
        - database indexing
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
# RESPONSE MODEL
# ============================================================

@dataclass
class AIResponse:

    query: str

    route: str

    cache_hit: bool

    retrieved_docs: List[str]

    answer: str

# ============================================================
# SEMANTIC CACHE SEARCH
# ============================================================

def search_semantic_cache(query):

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
    #Position of highest similarity score

    best_score = similarities[best_idx]

    print(f"\nSemantic Cache Score: {best_score:.4f}")

    if best_score > 0.90:

        matched_query = cache_queries[best_idx]

        return semantic_cache[matched_query]

    return None

# ============================================================
# SAVE CACHE
# ============================================================

def save_cache(query, answer):

    semantic_cache[query] = answer

# ============================================================
# QUERY CLASSIFIER
# ============================================================

def classify_query(query):

    simple_keywords = [

        "what is",

        "define",

        "explain",

        "meaning",

        "list"
    ]

    lower_query = query.lower()

    for keyword in simple_keywords:

        if keyword in lower_query:

            return "simple"

    return "complex"

# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_documents(query, top_k=2):

    query_embedding = embedding_model.encode([query])
    # High-dimensional vector numpy.ndarray 
    
    # (3, 384)
    

    similarities = cosine_similarity(
        query_embedding,
        document_embeddings
    )[0]
    
    # How close vectors are numpy.ndarray

    ranked_indices = np.argsort(similarities)[::-1]     # Reverses array.
    
    # Indices sorted by score

    results = []

    for idx in ranked_indices[:top_k]:

        results.append({

            "title": knowledge_base[idx]["title"],

            "content": knowledge_base[idx]["content"],

            "score": float(similarities[idx])
        })

    return results

# ============================================================
# PROMPT COMPRESSION
# ============================================================

def compress_context(retrieved_docs):

    compressed = []

    for doc in retrieved_docs:

        content = doc["content"]

        compressed.append(content[:150])

    return "\n".join(compressed)

# ============================================================
# SMALL MODEL RESPONSE
# ============================================================

def small_model_response(query):

    print("\nUsing SMALL model route...\n")

    simple_responses = {

        "what is kubernetes":
            "Kubernetes is a container orchestration platform.",

        "what is kafka":
            "Kafka is a distributed event streaming platform."
    }

    lower_query = query.lower()

    return simple_responses.get(

        lower_query,

        "Simple infrastructure query processed."
    )

# ============================================================
# LARGE MODEL RESPONSE
# ============================================================

def large_model_response(query, retrieved_docs):

    print("\nUsing GPT-4 LARGE model route...\n")

    context = compress_context(retrieved_docs)

    prompt = f"""
    Answer infrastructure question using ONLY context.

    USER QUESTION:
    {query}

    CONTEXT:
    {context}

    Keep answer concise.
    Avoid unnecessary reasoning.
    """

    response = client.chat.completions.create(

        model="gpt-4.1-mini",

        messages=[

            {
                "role": "system",

                "content": """
                You are a concise infrastructure AI assistant.

                Avoid unnecessary Chain-of-Thought.
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

    return response.choices[0].message.content

# ============================================================
# MAIN PIPELINE
# ============================================================

def optimized_ai_pipeline(query):

    print("\n================================================")
    print("TOKEN COST OPTIMIZED AI PIPELINE")
    print("================================================")

    # --------------------------------------------------------
    # STEP 1: SEMANTIC CACHE
    # --------------------------------------------------------

    print("\nSTEP 1: Semantic Cache Check")

    cached_response = search_semantic_cache(query)

    if cached_response:

        return AIResponse(

            query=query,

            route="CACHE",

            cache_hit=True,

            retrieved_docs=[],

            answer=cached_response
        )

    # --------------------------------------------------------
    # STEP 2: QUERY CLASSIFICATION
    # --------------------------------------------------------

    print("\nSTEP 2: Query Classification")

    query_type = classify_query(query)

    print(f"\nQuery Type: {query_type}")

    # --------------------------------------------------------
    # STEP 3: SMALL MODEL ROUTING
    # --------------------------------------------------------

    if query_type == "simple":

        answer = small_model_response(query)

        save_cache(query, answer)

        return AIResponse(

            query=query,

            route="SMALL_MODEL",

            cache_hit=False,

            retrieved_docs=[],

            answer=answer
        )

    # --------------------------------------------------------
    # STEP 4: RETRIEVAL-FIRST
    # --------------------------------------------------------

    print("\nSTEP 4: Retrieval")

    retrieved_docs = retrieve_documents(query)

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
    # STEP 5: LARGE MODEL ONLY IF NEEDED
    # --------------------------------------------------------

    print("\nSTEP 5: Large Model Reasoning")

    answer = large_model_response(

        query,

        retrieved_docs
    )

    # --------------------------------------------------------
    # STEP 6: SAVE CACHE
    # --------------------------------------------------------

    save_cache(query, answer)

    # --------------------------------------------------------
    # FINAL RESPONSE
    # --------------------------------------------------------

    return AIResponse(

        query=query,

        route="GPT4",

        cache_hit=False,

        retrieved_docs=[

            doc["title"]

            for doc in retrieved_docs
        ],

        answer=answer
    )

# ============================================================
# TEST CASES
# ============================================================

if __name__ == "__main__":

    queries = [

        # SIMPLE QUERY
        "What is Kubernetes",

        # COMPLEX QUERY
        "How can we reduce API latency in distributed systems?",

        # REPEATED QUERY
        "How can we reduce API latency in distributed systems?"
    ]

    for query in queries:

        print("\n\n================================================")
        print(f"USER QUERY: {query}")
        print("================================================")

        result = optimized_ai_pipeline(query)

        print("\nFINAL RESPONSE")
        print("------------------------------------------------")

        print(f"\nQuery:\n{result.query}")

        print(f"\nRoute:\n{result.route}")

        print(f"\nCache Hit:\n{result.cache_hit}")

        print(f"""
        \nRetrieved Docs:
        {result.retrieved_docs}
        """)

        print(f"\nAnswer:\n{result.answer}")

        print("\n================================================")