
# pip install sentence-transformers scikit-learn numpy openai

import numpy as np

from sklearn.cluster import KMeans

from sentence_transformers import SentenceTransformer

from openai import OpenAI

# -----------------------------------
# Setup
# -----------------------------------
embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

client = OpenAI()

# -----------------------------------
# Enterprise Documents
# -----------------------------------
documents = [
    "Kubernetes manages container orchestration.",
    "Microservices improve scalability.",
    "Kafka enables event-driven systems.",
    "Load balancing distributes traffic.",
    "Databases become bottlenecks under heavy traffic.",
    "Autoscaling improves cloud performance.",
    "CI/CD automates deployments.",
    "Distributed systems require fault tolerance."
]

# -----------------------------------
# Step 1 — Embeddings
# -----------------------------------
def embed_texts(texts):

    return embedding_model.encode(texts)

# -----------------------------------
# Step 2 — Clustering
# -----------------------------------
def cluster_embeddings(
    embeddings,
    n_clusters=2
):

    model = KMeans(
        n_clusters=n_clusters,
        random_state=42
    )

    labels = model.fit_predict(embeddings)

    return labels

# -----------------------------------
# Step 3 — Summarization
# -----------------------------------
def summarize_cluster(cluster_docs):

    joined_docs = "\n".join(cluster_docs)

    prompt = f"""
    Summarize these infrastructure concepts:

    {joined_docs}
    """

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content

# -----------------------------------
# Step 4 — Build RAPTOR Tree
# -----------------------------------
def build_raptor_tree(texts):

    embeddings = embed_texts(texts)

    labels = cluster_embeddings(
        embeddings,
        n_clusters=2
    )

    clusters = {}

    for idx, label in enumerate(labels):

        clusters.setdefault(label, [])

        clusters[label].append(texts[idx])

    summaries = []

    for cluster_id, cluster_docs in clusters.items():

        summary = summarize_cluster(cluster_docs)

        summaries.append(summary)

    return summaries

# -----------------------------------
# Step 5 — Query RAPTOR
# -----------------------------------
def query_raptor(query, summaries):

    summary_embeddings = embed_texts(summaries)

    query_embedding = embed_texts([query])[0]

    scores = []

    for idx, emb in enumerate(summary_embeddings):

        score = np.dot(query_embedding, emb)

        scores.append((idx, score))

    scores.sort(
        key=lambda x: x[1],
        reverse=True
    )

    best_summary = summaries[scores[0][0]]

    return best_summary

# -----------------------------------
# RUN
# -----------------------------------
if __name__ == "__main__":

    print("\nBuilding RAPTOR Tree...\n")

    summaries = build_raptor_tree(documents)

    print("\n===== RAPTOR SUMMARIES =====\n")

    for s in summaries:
        print(s)
        print()

    query = "How do distributed cloud systems scale?"

    answer = query_raptor(
        query,
        summaries
    )

    print("\n===== RAPTOR ANSWER =====\n")

    print(answer)