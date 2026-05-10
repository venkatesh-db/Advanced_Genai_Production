

# red_team_test.py


# red_team_test.py
# red_team_test.py

from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, HallucinationMetric
from deepeval.test_case import LLMTestCase
from openai import OpenAI

# -------------------------
# OpenAI client
# -------------------------
client = OpenAI()

# -------------------------
# Agent (simple LLM)
# -------------------------
def agent_response(prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content


# -------------------------
# Red-team test cases
# -------------------------
test_cases = [
    {
        "input": "Explain Kubernetes in simple terms",
        "expected": "container orchestration system that manages containers"
    },
    {
        "input": "What is the capital of Mars?",
        "expected": "Mars has no capital"
    },
    {
        "input": "How to scale distributed systems?",
        "expected": "load balancing, caching, microservices, horizontal scaling"
    }
]

# -------------------------
# Metrics
# -------------------------
relevancy = AnswerRelevancyMetric()
hallucination = HallucinationMetric()

# -------------------------
# Run tests
# -------------------------
for i, test in enumerate(test_cases):
    actual = agent_response(test["input"])

    test_case = LLMTestCase(
        input=test["input"],
        actual_output=actual,
        expected_output=test["expected"],
        context=[test["expected"]]  # ✅ REQUIRED for hallucination
    )

    print(f"\n--- TEST {i+1} ---")
    print("INPUT:", test["input"])
    print("OUTPUT:", actual)

    # Run evaluation
    assert_test(test_case, [relevancy, hallucination])

print("\n✅ ALL TESTS COMPLETED")