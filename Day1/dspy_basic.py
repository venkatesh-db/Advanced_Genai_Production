
# dspy_basic.py
# pip install dspy-ai

# dspy_production.py


# dspy_final.py

import dspy
from dspy import LM
from dspy.teleprompt import BootstrapFewShot

# -------------------------
# Configure LLM
# -------------------------
lm = LM(model="openai/gpt-4.1-mini")
dspy.configure(lm=lm)

# -------------------------
# Signatures
# -------------------------
class Summarize(dspy.Signature):
    context = dspy.InputField()
    summary = dspy.OutputField()

class SuggestSolution(dspy.Signature):
    summary = dspy.InputField()
    solution = dspy.OutputField()

# -------------------------
# Pipeline
# -------------------------
class InfraPipeline(dspy.Module):
    def __init__(self):
        self.summarize = dspy.ChainOfThought(Summarize)
        self.solve = dspy.ChainOfThought(SuggestSolution)

    def forward(self, context):
        s = self.summarize(context=context)
        out = self.solve(summary=s.summary)
        return out.solution

# -------------------------
# Training Data
# -------------------------
trainset = [
    dspy.Example(
        context="High CPU usage and latency spikes in distributed system",
        solution="Scale horizontally and balance load across nodes"
    ).with_inputs("context"),

    dspy.Example(
        context="Frequent storage timeouts in Nutanix cluster",
        solution="Check disk health and rebalance storage nodes"
    ).with_inputs("context"),

    dspy.Example(
        context="Network congestion causing high latency",
        solution="Optimize network routing and reduce bottlenecks"
    ).with_inputs("context"),

    dspy.Example(
        context="API failures under heavy traffic",
        solution="Introduce rate limiting and improve load balancing"
    ).with_inputs("context"),
]

# -------------------------
# Metric (improved)
# -------------------------
def metric(example, pred, trace=None):
    expected = set(example.solution.lower().split())
    predicted = set(pred.lower().split())

    if not expected:
        return False

    overlap = len(expected & predicted) / len(expected)
    return overlap > 0.3

# -------------------------
# Optimizer
# -------------------------
optimizer = BootstrapFewShot(
    metric=metric,
    max_bootstrapped_demos=4
)

pipeline = InfraPipeline()

compiled_pipeline = optimizer.compile(
    pipeline,
    trainset=trainset
)

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    test_input = "API latency high and CPU overloaded in cluster"

    result = compiled_pipeline(context=test_input)

    print("\n===== DSPY FINAL OUTPUT =====")
    print(result)
