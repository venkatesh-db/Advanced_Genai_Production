

import numpy as np
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from numpy.linalg import norm
from openai import OpenAI

# -------------------------
# Setup
# -------------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
client = OpenAI()

# -------------------------
# Utilities
# -------------------------
def cosine_similarity(a, b):
    if a is None or b is None:
        return -1
    return np.dot(a, b) / (norm(a) * norm(b))

def embed_batch(texts):
    return embedding_model.encode(texts, batch_size=32)

def deduplicate_chunks(chunks):
    return list(set(chunks))

# -------------------------
# RAPTOR Node
# -------------------------
class RaptorNode:
    def __init__(self, texts, level):
        self.texts = texts
        self.level = level
        self.embedding = None
        self.summary = None
        self.children = []

# -------------------------
# Summarization
# -------------------------
def summarize(texts):
    prompt = "Summarize these technical concepts into a high-level abstraction:\n\n"
    prompt += "\n".join(texts)

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

# -------------------------
# Build RAPTOR Tree
# -------------------------
def build_tree(texts, level=0, max_levels=2, cluster_size=2):
    node = RaptorNode(texts, level)

    embeddings = embed_batch(texts)
    node.embedding = np.mean(embeddings, axis=0)

    if level >= max_levels or len(texts) <= cluster_size:
        return node

    kmeans = KMeans(n_clusters=cluster_size, random_state=42)
    labels = kmeans.fit_predict(embeddings)

    clusters = {}
    for i, label in enumerate(labels):
        clusters.setdefault(label, []).append(texts[i])

    for cluster_texts in clusters.values():

        child_node = RaptorNode(cluster_texts, level + 1)

        child_embeddings = embed_batch(cluster_texts)
        child_node.embedding = np.mean(child_embeddings, axis=0)

        child_node.summary = summarize(cluster_texts)

        # Recursive build using summary abstraction
        summary_node = build_tree(
            texts=[child_node.summary],
            level=level + 1,
            max_levels=max_levels,
            cluster_size=cluster_size
        )

        child_node.children = summary_node.children
        node.children.append(child_node)

    return node

# -------------------------
# Top-K + Threshold Traversal
# -------------------------
def query_tree_topk(node, query_embedding, k=2, threshold=0.35):
    if not node.children:
        return node.texts

    scored_children = []

    for child in node.children:
        if child.embedding is None:
            continue

        score = cosine_similarity(child.embedding, query_embedding)

        # 🔥 filter noise
        if score >= threshold:
            scored_children.append((score, child))

    if not scored_children:
        return node.texts  # fallback

    scored_children.sort(reverse=True, key=lambda x: x[0])

    results = []

    for i in range(min(k, len(scored_children))):
        _, child = scored_children[i]
        results.extend(query_tree_topk(child, query_embedding, k, threshold))

    return results

# -------------------------
# Answer Generation
# -------------------------
def generate_answer(query, context_chunks):
    context_chunks = deduplicate_chunks(context_chunks)
    context = "\n".join(context_chunks)

    prompt = f"""
    Answer the question using ONLY the context below.

    Context:
    {context}

    Question:
    {query}
    """

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":

    documents = [
        "Microservices architecture improves scalability and independent deployment.",
        "Kubernetes orchestrates containers across clusters.",
        "Distributed systems must handle failures and partitions.",
        "Vector databases enable semantic search using embeddings.",
        "RAG combines retrieval with generation for better answers.",
        "Caching reduces latency and improves performance.",
        "Load balancing distributes incoming traffic efficiently.",
        "CI/CD pipelines automate build and deployment processes."
    ]

    print("Building RAPTOR tree...\n")

    root = build_tree(documents, max_levels=2)

    query = "How do distributed systems scale efficiently?"
    query_embedding = embed_batch([query])[0]

    # 🔥 Final retrieval (Top-K + Filtering)
    retrieved_chunks = query_tree_topk(root, query_embedding, k=2, threshold=0.35)

    answer = generate_answer(query, retrieved_chunks)

    print("\n--- RETRIEVED CONTEXT ---")
    for c in deduplicate_chunks(retrieved_chunks):
        print("-", c)

    print("\n--- FINAL ANSWER ---")
    print(answer)