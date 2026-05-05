
# main.py
# a2a_basic.py 

from A2A_basic import A2AService


def create_message(sender, receiver, problem, context):
    return {
        "sender": sender,
        "receiver": receiver,
        "problem": problem,
        "context": context,
        "metadata": {
            "type": "A2A_REQUEST",
            "version": "1.0"
        }
    }


def infra_agent():
    print("\n[InfraAgent] Preparing request...")

    problem = "API latency spikes under high traffic"

    context = {
        "cpu_usage": "85%",
        "memory": "70%",
        "requests_per_sec": 1200,
        "error_rate": "5%"
    }

    message = create_message(
        sender="InfraAgent",
        receiver="DebugAgent",
        problem=problem,
        context=context
    )

    print("\n[InfraAgent] Sending message:")
    print(message)

    # Call Agent B
    debug_agent = A2AService()
    response = debug_agent.process(message)

    print("\n[InfraAgent] Final response received:")
    print(response)


if __name__ == "__main__":
    infra_agent()