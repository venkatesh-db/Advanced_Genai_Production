
'''

python3 -m venv modern_env
source modern_env/bin/activate

pip install langchain langchain-openai langgraph langfuse

export LANGFUSE_PUBLIC_KEY=pk-lf-xxxx
export LANGFUSE_SECRET_KEY=sk-lf-xxxx
export LANGFUSE_HOST=https://us.cloud.langfuse.com

'''
from typing import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

# -------------------------
# LLM
# -------------------------
llm = ChatOpenAI(model="gpt-4.1-mini")

# -------------------------
# State
# -------------------------
class State(TypedDict):
    input: str
    analysis: str
    solution: str

# -------------------------
# Nodes
# -------------------------
def analyze_node(state: State):
    prompt = f"Analyze: {state['input']}"
    res = llm.invoke(prompt)
    return {"analysis": res.content}

def solution_node(state: State):
    prompt = f"Solution: {state['analysis']}"
    res = llm.invoke(prompt)
    return {"solution": res.content}

# -------------------------
# Graph
# -------------------------
graph = StateGraph(State)
graph.add_node("analyze", analyze_node)
graph.add_node("solve", solution_node)

graph.set_entry_point("analyze")
graph.add_edge("analyze", "solve")
graph.add_edge("solve", END)

app = graph.compile()

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    result = app.invoke({
        "input": "High CPU usage causing API latency"
    })

    print("\nOUTPUT:\n", result["solution"])