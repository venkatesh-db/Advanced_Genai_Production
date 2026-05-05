import numpy as np
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# -------------------------
# Setup
# -------------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
client = OpenAI()

# -------------------------
# Sample Documents (simulate thousands)
# -------------------------
documents = [
    "Microservices architecture improves scalability and deployment independence.",
    "Kubernetes manages container orchestration at scale.",
    "Distributed systems face consistency and partition tolerance challenges.",
    "Vector databases store embeddings for semantic search.",
    "RAG combines retrieval with generation for better answers.",
    "CI/CD pipelines automate deployment and reduce human errors.",
    "Load balancing distributes traffic across servers.",
    "Caching improves performance by reducing repeated computation."
]

# -------------------------
# Step 1: Embed
# -------------------------
def embed_texts(texts):
    return embedding_model.encode(texts)

# -------------------------
# Step 2: Cluster
# -------------------------
def cluster_texts_kmeans(embeddings, n_clusters=2):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    return labels

# -------------------------
# Step 3: Summarize Cluster (LLM)
# -------------------------
def summarize_cluster(texts):
    prompt = "Summarize the following technical concepts into a high-level theme:\n\n"
    prompt += "\n".join(texts)

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# -------------------------
# Step 4: Build RAPTOR Tree (1 level for demo)
# -------------------------
def build_raptor_tree(texts):
    embeddings = embed_texts(texts)
    labels = cluster_texts_kmeans(embeddings, n_clusters=2)

    # Debugging: Print embeddings and labels
    print("Embeddings:", embeddings)
    print("Labels:", labels)

    clusters = {}
    for i, label in enumerate(labels):
        clusters.setdefault(label, []).append(texts[i])

    summaries = []
    for cluster_id, cluster_texts in clusters.items():
        summary = summarize_cluster(cluster_texts)
        summaries.append(summary)

    return summaries

# -------------------------
# Step 5: Query
# -------------------------
def query_raptor(query, summaries):
    summary_embeddings = embed_texts(summaries)
    query_embedding = embed_texts([query])[0]

    scores = np.dot(summary_embeddings, query_embedding)
    best_idx = np.argmax(scores)

    return summaries[best_idx]

# -------------------------
# RUN PIPELINE
# -------------------------
print("Building RAPTOR tree...\n")
summaries = build_raptor_tree(documents)

print("\n--- Cluster Summaries ---")
for s in summaries:
    print(s, "\n")

query = "How do large systems scale efficiently?"
answer = query_raptor(query, summaries)

print("\n--- RAPTOR Answer ---")
print(answer)