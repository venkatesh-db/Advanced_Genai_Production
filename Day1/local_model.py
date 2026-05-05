

# local_model.py
# local_model.py

import requests
import time

def query_local(prompt):
    start = time.time()

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3",
                "prompt": prompt,
                "stream": False
            },
            timeout=10
        )

        latency = time.time() - start
        data = response.json()

        # ✅ handle success
        if "response" in data:
            return data["response"], latency

        # ❌ handle model not found
        if "error" in data:
            return f"ERROR: {data['error']}", latency

        # ❌ unexpected structure
        return f"ERROR: Unexpected response {data}", latency

    except Exception as e:
        return f"ERROR: {str(e)}", 0