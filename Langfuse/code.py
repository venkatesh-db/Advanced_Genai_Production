
'''

python3 -m venv clean_env
source clean_env/bin/activate

pip install langfuse==2.29.0
pip install langchain==0.0.350
pip install openai
pip install langgraph

'''



from typing import TypedDict
from langchain.chat_models import ChatOpenAI   # ✅ FIXED IMPORT
from langgraph.graph import StateGraph, END
from langfuse.callback import CallbackHandler

# Langfuse handler
handler = CallbackHandler(
    session_id="infra-debug",
    tags=["langgraph"]
)

# LLM
llm = ChatOpenAI(
    model_name="gpt-4.1-mini",   # ✅ FIXED PARAM
    callbacks=[handler]
)

# State
class State(TypedDict):
    input: str
    analysis: str
    solution: str

# Nodes
def analyze_node(state: State):
    res = llm.predict(f"Analyze: {state['input']}")
    return {"analysis": res}

def solution_node(state: State):
    res = llm.predict(f"Solution: {state['analysis']}")
    return {"solution": res}

# Graph
graph = StateGraph(State)
graph.add_node("analyze", analyze_node)
graph.add_node("solve", solution_node)

graph.set_entry_point("analyze")
graph.add_edge("analyze", "solve")
graph.add_edge("solve", END)

app = graph.compile()

# Run
if __name__ == "__main__":
    result = app.invoke({
        "input": "High CPU usage causing API latency"
    })

    print("\nOUTPUT:\n", result["solution"])