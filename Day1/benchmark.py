
# benchmark.py

# ollama serve
# ollama run llama3
# python3 benchmark.py 



# benchmark.py

from local_model import query_local
from frontier_model import query_frontier

prompt = "Diagnose high latency in Nutanix distributed storage system"

local_out, local_lat = query_local(prompt)
front_out, front_lat = query_frontier(prompt)

print("\n===== BENCHMARK =====")

# 🔥 VALIDATION
if "ERROR" in local_out:
    print("\n⚠️ LOCAL MODEL FAILED:")
    print(local_out)
    print("\n👉 Fix by running: ollama pull llama3\n")
else:
    print("\nLOCAL MODEL")
    print("Latency:", local_lat)
    print(local_out[:300])

print("\nFRONTIER MODEL")
print("Latency:", front_lat)
print(front_out[:300])