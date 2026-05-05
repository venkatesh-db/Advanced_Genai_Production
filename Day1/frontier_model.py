# frontier_model.py

from openai import OpenAI
import time

client = OpenAI()

def query_frontier(prompt):
    start = time.time()

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    latency = time.time() - start

    return response.choices[0].message.content, latency


if __name__ == "__main__":
    prompt = "Explain Nutanix cluster latency issues under load"

    output, latency = query_frontier(prompt)

    print("\n--- FRONTIER MODEL ---")
    print("Latency:", latency)
    print("Output:\n", output)