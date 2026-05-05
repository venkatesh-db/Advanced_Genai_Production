
#uvicorn agent_server:app --reload
#python3 agent_client.py


import requests
import time

URL = "http://127.0.0.1:8000/process"
HEADERS = {"x-api-key": "secret123"}
MAX_RETRIES = 3
TIMEOUT = 3

# simple circuit breaker
failure_count = 0
OPEN_THRESHOLD = 3
COOLDOWN = 5  # seconds
last_failure_time = 0

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
        "metadata": {"type": "A2A_REQUEST", "version": "2.2"}
    }

def circuit_open():
    global failure_count, last_failure_time
    if failure_count >= OPEN_THRESHOLD:
        if time.time() - last_failure_time < COOLDOWN:
            return True
        # half-open reset
        failure_count = 0
    return False

def call_agent():
    global failure_count, last_failure_time

    if circuit_open():
        print("⚠️ Circuit open. Skipping call.")
        return

    msg = create_message()
    print("\n[InfraAgent] Sending request...")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(URL, json=msg, headers=HEADERS, timeout=TIMEOUT)
            if resp.status_code == 200:
                print("\n[InfraAgent] Response:", resp.json())
                failure_count = 0
                return
            else:
                raise Exception(f"HTTP {resp.status_code}: {resp.text}")

        except Exception as e:
            print(f"Attempt {attempt} failed:", e)
            if attempt == MAX_RETRIES:
                failure_count += 1
                last_failure_time = time.time()
                print("❌ All retries failed. Circuit may open.")
            else:
                time.sleep(1)  # backoff

if __name__ == "__main__":
    call_agent()