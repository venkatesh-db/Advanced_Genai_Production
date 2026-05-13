
# pip install networkx matplotlib openai

import networkx as nx
from openai import OpenAI

# -----------------------------
# OpenAI Setup
# -----------------------------
client = OpenAI()

# -----------------------------
# Step 1 — Sample Enterprise Data
# -----------------------------
documents = [
    {
        "source": "API Gateway",
        "target": "Database",
        "relation": "depends_on"
    },
    {
        "source": "Database",
        "target": "Storage Cluster",
        "relation": "hosted_on"
    },
    {
        "source": "Storage Cluster",
        "target": "Network Switch",
        "relation": "connected_to"
    },
    {
        "source": "Network Switch",
        "target": "Latency Issue",
        "relation": "caused"
    }
]

# -----------------------------
# Step 2 — Build Knowledge Graph
# -----------------------------
graph = nx.DiGraph()

for doc in documents:
    graph.add_edge(
        doc["source"],
        doc["target"],
        relation=doc["relation"]
    )

# -----------------------------
# Step 3 — Graph Traversal
# -----------------------------
def get_related_nodes(node):
    related = []

    for neighbor in graph.neighbors(node):
        relation = graph[node][neighbor]["relation"]

        related.append({
            "from": node,
            "to": neighbor,
            "relation": relation
        })

    return related

# -----------------------------
# Step 4 — Build Context
# -----------------------------
def build_context(query_node):
    related_nodes = get_related_nodes(query_node)

    context = ""

    for item in related_nodes:
        context += (
            f"{item['from']} "
            f"{item['relation']} "
            f"{item['to']}\n"
        )

    return context

# -----------------------------
# Step 5 — Ask LLM
# -----------------------------
def graph_rag_query(query_node):

    context = build_context(query_node)

    prompt = f"""
    You are infrastructure AI assistant.

    Analyze this system graph:

    {context}

    Explain possible operational impact.
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

# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":

    query = "API Gateway"

    result = graph_rag_query(query)

    print("\n===== GraphRAG OUTPUT =====\n")
    print(result)