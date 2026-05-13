
# pip install sentence-transformers networkx numpy openai

import networkx as nx
import numpy as np

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
# Memory Documents
# -----------------------------------
memories = [
    "Kubernetes manages container orchestration.",
    "Microservices improve scalability.",
    "Kafka enables event-driven systems.",
    "Load balancing distributes traffic.",
    "Databases become bottlenecks under heavy traffic.",
    "Autoscaling improves cloud performance."
]

# -----------------------------------
# Step 1 — Create Memory Graph
# -----------------------------------
graph = nx.Graph()

# -----------------------------------
# Step 2 — Generate Embeddings
# -----------------------------------
memory_embeddings = embedding_model.encode(memories)

# -----------------------------------
# Step 3 — Add Memory Nodes
# -----------------------------------
for i, memory in enumerate(memories):

    graph.add_node(
        i,
        text=memory,
        embedding=memory_embeddings[i]
    )

# -----------------------------------
# Step 4 — Build Associations
# -----------------------------------
threshold = 0.45

for i in range(len(memories)):
    for j in range(i + 1, len(memories)):

        sim = np.dot(
            memory_embeddings[i],
            memory_embeddings[j]
        )

        if sim > threshold:

            graph.add_edge(i, j, weight=sim)

# -----------------------------------
# Step 5 — Associative Recall
# -----------------------------------
def associative_recall(query):

    query_embedding = embedding_model.encode([query])[0]

    scores = []

    for node in graph.nodes(data=True):

        idx = node[0]
        memory_embedding = node[1]["embedding"]

        score = np.dot(
            query_embedding,
            memory_embedding
        )

        scores.append((idx, score))

    scores.sort(
        key=lambda x: x[1],
        reverse=True
    )

    best_memory = scores[0][0]

    recalled_memories = []

    recalled_memories.append(
        graph.nodes[best_memory]["text"]
    )

    # -----------------------------------
    # Recall associated memories
    # -----------------------------------
    neighbors = graph.neighbors(best_memory)

    for n in neighbors:

        recalled_memories.append(
            graph.nodes[n]["text"]
        )

    return recalled_memories

# -----------------------------------
# Step 6 — LLM Reasoning
# -----------------------------------
def ask_hipporag(query):

    recalled = associative_recall(query)

    context = "\n".join(recalled)

    prompt = f"""
    You are an infrastructure AI assistant.

    Use these connected memories:

    {context}

    Answer this question:

    {query}
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
# RUN
# -----------------------------------
if __name__ == "__main__":

    query = "How do cloud systems scale?"

    answer = ask_hipporag(query)

    print("\n===== HippoRAG OUTPUT =====\n")

    print(answer)