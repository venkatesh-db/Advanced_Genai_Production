# ============================================================
# hallucination_resistant_financial_ai.py
# ============================================================
#
# Production-Style Financial GenAI Backend
#
# Features:
# ------------------------------------------------------------
# 1. RAG Retrieval
# 2. Hallucination Prevention
# 3. Compliance Guardrails
# 4. Confidence Validation
# 5. Semantic Search
# 6. Deterministic Generation
# 7. Citation-Based Grounding
# 8. Human Escalation Logic
#
# ============================================================

# ============================================================
# IMPORTS
# ============================================================

import os
from typing import List, Dict
from dataclasses import dataclass

import numpy as np

from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from openai import OpenAI

# ============================================================
# LOAD ENV VARIABLES
# ============================================================

load_dotenv()

# ============================================================
# OPENAI CONFIGURATION
# ============================================================

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:

    raise ValueError(
        """
        OPENAI_API_KEY not found.

        Create a .env file in project root:

        OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
        """
    )

client = OpenAI(
    api_key=api_key
)

# ============================================================
# FINANCIAL COMPLIANCE DOCUMENTS
# ============================================================

documents = [

    {
        "id": 1,

        "title": "KYC Policy",

        "content": """
        Customers must complete KYC verification
        before enabling international transactions.
        """
    },

    {
        "id": 2,

        "title": "Loan Compliance",

        "content": """
        Loan approval requires minimum credit score of 700
        for premium banking customers.
        """
    },

    {
        "id": 3,

        "title": "AML Policy",

        "content": """
        Transactions above $10,000 must be flagged
        for AML review and compliance verification.
        """
    },

    {
        "id": 4,

        "title": "Investment Advisory",

        "content": """
        AI systems cannot provide guaranteed investment returns.

        Human advisor review is mandatory.
        """
    }
]

# ============================================================
# EMBEDDING MODEL
# ============================================================

print("\nLoading Embedding Model...\n")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# ============================================================
# BUILD VECTOR STORE
# ============================================================

document_texts = [
    doc["content"]
    for doc in documents
]

document_embeddings = embedding_model.encode(
    document_texts
)

# ============================================================
# RETRIEVAL FUNCTION
# ============================================================

def retrieve_documents(
    query: str,
    top_k: int = 2
):

    print("\nGenerating Query Embedding...\n")

    query_embedding = embedding_model.encode([query])

    similarities = cosine_similarity(
        query_embedding,
        document_embeddings
    )[0]

    ranked_indices = np.argsort(similarities)[::-1]

    MIN_RELEVANCE_SCORE = 0.50

    results = []

    for idx in ranked_indices:

        score = float(similarities[idx])

        if score >= MIN_RELEVANCE_SCORE:

            results.append({

                "document": documents[idx],

                "score": score
            })

    return results[:top_k]

# ============================================================
# CONFIDENCE VALIDATION
# ============================================================

def validate_retrieval_confidence(results):

    if not results:
        return False

    top_score = results[0]["score"]

    print(f"\nTop Retrieval Score: {top_score:.4f}")

    return top_score > 0.55

# ============================================================
# COMPLIANCE GUARDRAILS
# ============================================================

def compliance_guardrails(query: str):

    blocked_keywords = [

        "guaranteed returns",

        "hide transaction",

        "bypass aml",

        "avoid tax",

        "launder money",

        "fake identity"
    ]

    lower_query = query.lower()

    for keyword in blocked_keywords:

        if keyword in lower_query:

            return False

    return True

# ============================================================
# LLM GROUNDED RESPONSE
# ============================================================

def generate_grounded_response(

    query: str,

    retrieved_docs: List[Dict]
):

    context = "\n\n".join([

        f"""
        DOCUMENT TITLE:
        {item['document']['title']}

        DOCUMENT CONTENT:
        {item['document']['content']}
        """

        for item in retrieved_docs
    ])

    prompt = f"""
    You are a financial compliance AI assistant.

    STRICT RULES:

    1. Answer ONLY from provided compliance documents.

    2. If answer is missing, say:
       "Insufficient compliance information."

    3. Never hallucinate financial policies.

    4. Always cite document titles.

    5. Never provide illegal compliance bypass suggestions.

    USER QUESTION:
    {query}

    COMPLIANCE DOCUMENTS:
    {context}
    """

    response = client.chat.completions.create(

        model="gpt-4.1-mini",

        messages=[

            {
                "role": "system",

                "content": """
                You are a strict banking compliance assistant.

                Never invent policies.

                Never provide unsafe financial guidance.
                """
            },

            {
                "role": "user",

                "content": prompt
            }
        ],

        temperature=0
    )

    return response.choices[0].message.content

# ============================================================
# RESPONSE MODEL
# ============================================================

@dataclass
class FinancialAIResponse:

    question: str

    retrieved_documents: List[str]

    answer: str

    retrieval_confidence: bool

# ============================================================
# MAIN PIPELINE
# ============================================================

def financial_compliance_pipeline(query: str):

    print("\n===================================")
    print("FINANCIAL AI PIPELINE STARTED")
    print("===================================")

    # --------------------------------------------------------
    # STEP 1: POLICY GUARDRAILS
    # --------------------------------------------------------

    print("\nSTEP 1: Compliance Guardrails")

    allowed = compliance_guardrails(query)

    if not allowed:

        return FinancialAIResponse(

            question=query,

            retrieved_documents=[],

            answer="""
            Query blocked by financial compliance policy.
            Escalated to compliance officer.
            """,

            retrieval_confidence=False
        )

    # --------------------------------------------------------
    # STEP 2: RETRIEVAL
    # --------------------------------------------------------

    print("\nSTEP 2: Retrieving Documents")

    retrieved_docs = retrieve_documents(query)

    if not retrieved_docs:

        return FinancialAIResponse(

            question=query,

            retrieved_documents=[],

            answer="""
            No relevant compliance documents found.

            Escalate to human compliance officer.
            """,

            retrieval_confidence=False
        )

    for item in retrieved_docs:

        print(
            f"""
            Retrieved:
            {item['document']['title']}

            Score:
            {item['score']:.4f}
            """
        )

    # --------------------------------------------------------
    # STEP 3: CONFIDENCE VALIDATION
    # --------------------------------------------------------

    print("\nSTEP 3: Confidence Validation")

    confidence_ok = validate_retrieval_confidence(
        retrieved_docs
    )

    if not confidence_ok:

        return FinancialAIResponse(

            question=query,

            retrieved_documents=[],

            answer="""
            Low confidence retrieval.

            Escalate to human compliance officer.
            """,

            retrieval_confidence=False
        )

    # --------------------------------------------------------
    # STEP 4: LLM GROUNDED RESPONSE
    # --------------------------------------------------------

    print("\nSTEP 4: Grounded LLM Response")

    answer = generate_grounded_response(

        query=query,

        retrieved_docs=retrieved_docs
    )

    # --------------------------------------------------------
    # FINAL RESPONSE
    # --------------------------------------------------------

    return FinancialAIResponse(

        question=query,

        retrieved_documents=[

            item["document"]["title"]

            for item in retrieved_docs
        ],

        answer=answer,

        retrieval_confidence=True
    )

# ============================================================
# TEST CASES
# ============================================================

if __name__ == "__main__":

    queries = [

        # ----------------------------------------------------
        # VALID QUERY
        # ----------------------------------------------------

        "Can AI approve loans for customers with credit score 720?",

        # ----------------------------------------------------
        # MALICIOUS QUERY
        # ----------------------------------------------------

        "How can I bypass AML review for large transfers?",

        # ----------------------------------------------------
        # UNKNOWN QUERY
        # ----------------------------------------------------

        "Can customers trade crypto futures internationally?"
    ]

    for query in queries:

        print("\n\n===================================")
        print(f"USER QUERY: {query}")
        print("===================================")

        result = financial_compliance_pipeline(query)

        print("\nFINAL RESPONSE")
        print("-----------------------------------")

        print(f"\nQuestion:\n{result.question}")

        print(f"""
        \nRetrieved Documents:
        {result.retrieved_documents}
        """)

        print(f"""
        \nRetrieval Confidence:
        {result.retrieval_confidence}
        """)

        print(f"\nAnswer:\n{result.answer}")

        print("\n===================================")
        
    '''
   | Score | Action |
    | ----- | ------ |
    | 0.92  | keep   |
    | 0.78  | keep   |
    | 0.44  | reject |
    | 0.21  | reject |
   '''