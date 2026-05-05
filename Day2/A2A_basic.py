
# agent_service.py

class A2AService:
    def __init__(self, name="DebugAgent"):
        self.name = name

    def process(self, message: dict) -> dict:
        print(f"\n[{self.name}] Received message:")
        print(message)

        problem = message.get("problem", "")
        context = message.get("context", {})

        # Basic decision logic (replace with LLM later)
        cpu = context.get("cpu_usage", "0%")
        error_rate = context.get("error_rate", "0%")

        if cpu == "85%" or error_rate == "5%":
            solution = "Scale horizontally, add load balancer, optimize API"
        else:
            solution = "Run deeper diagnostics"

        response = {
            "status": "SUCCESS",
            "solution": solution,
            "handled_by": self.name,
            "metadata": {
                "type": "A2A_RESPONSE",
                "version": "1.0"
            }
        }

        print(f"\n[{self.name}] Sending response:")
        print(response)

        return response