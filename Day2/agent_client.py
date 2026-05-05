

# agent_client.py

#uvicorn agent_server:app --reload
# python3 agent_client.py


import requests


def create_message():
    return {
        "sender": "InfraAgent",
        "receiver": "DebugAgent",
        "problem": "API latency spikes under high traffic",
        "context": {
            "cpu_usage": "85%",
            "memory": "70%",
            "requests_per_sec": 1200,
            "error_rate": "5%"
        },
        "metadata": {
            "type": "A2A_REQUEST",
            "version": "2.0"
        }
    }


def call_agent():
    url = "http://127.0.0.1:8000/process"

    message = create_message()

    print("\n[InfraAgent] Sending request...")
    print(message)

    try:
        response = requests.post(url, json=message, timeout=5)

        print("\n[InfraAgent] Response Status:", response.status_code)
        print("[InfraAgent] Response Data:", response.json())

    except requests.exceptions.RequestException as e:
        print("\n[InfraAgent] ERROR:", str(e))


if __name__ == "__main__":
    call_agent()