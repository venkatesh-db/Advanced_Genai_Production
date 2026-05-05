

from typing import TypedDict
from langgraph.graph import StateGraph, END
from crewai import Agent, Task, Crew
from openai import OpenAI

# -------------------------
# Setup
# -------------------------
client = OpenAI()

# -------------------------
# State Definition
# -------------------------
class RnDState(TypedDict):
    bug: str
    research_notes: str
    fix: str
    test_result: str
    iterations: int

# -------------------------
# Agents
# -------------------------
research_agent = Agent(
    role="Research Engineer",
    goal="Analyze bugs and identify root causes",
    backstory="Expert in debugging distributed systems"
)

debug_agent = Agent(
    role="Senior Debug Engineer",
    goal="Fix bugs using research insights",
    backstory="Writes clean and scalable fixes"
)

test_agent = Agent(
    role="QA Engineer",
    goal="Validate bug fixes",
    backstory="Ensures correctness and robustness"
)

# -------------------------
# Node 1: Research
# -------------------------
def research_node(state: RnDState):
    task = Task(
        description=f"Analyze this bug and list possible causes:\n{state['bug']}",
        agent=research_agent,
        expected_output="List of root causes with explanation"
    )

    crew = Crew(
        agents=[research_agent],
        tasks=[task],
        verbose=True
    )

    result = str(crew.kickoff())

    return {
        **state,
        "research_notes": result,
        "iterations": state["iterations"] + 1
    }

# -------------------------
# Node 2: Debug
# -------------------------
def debug_node(state: RnDState):
    task = Task(
        description=f"""
Bug:
{state['bug']}

Research:
{state['research_notes']}

Provide a FIX (code or explanation).
""",
        agent=debug_agent,
        expected_output="Clear fix with explanation or code"
    )

    crew = Crew(
        agents=[debug_agent],
        tasks=[task],
        verbose=True
    )

    result = str(crew.kickoff())

    return {
        **state,
        "fix": result
    }

# -------------------------
# Node 3: Test
# -------------------------
def test_node(state: RnDState):
    task = Task(
        description=f"""
Bug:
{state['bug']}

Fix:
{state['fix']}

Check if this fix resolves the issue.
Return PASS or FAIL with reason.
""",
        agent=test_agent,
        expected_output="PASS or FAIL with explanation"
    )

    crew = Crew(
        agents=[test_agent],
        tasks=[task],
        verbose=True
    )

    result = str(crew.kickoff())

    return {
        **state,
        "test_result": result
    }

# -------------------------
# Loop Control
# -------------------------
def should_continue(state: RnDState):
    if "PASS" in state["test_result"]:
        return "end"
    elif state["iterations"] >= 3:
        return "end"
    else:
        return "loop"

# -------------------------
# Build LangGraph
# -------------------------
graph = StateGraph(RnDState)

graph.add_node("research", research_node)
graph.add_node("debug", debug_node)
graph.add_node("test", test_node)

graph.set_entry_point("research")

graph.add_edge("research", "debug")
graph.add_edge("debug", "test")

graph.add_conditional_edges(
    "test",
    should_continue,
    {
        "loop": "research",
        "end": END
    }
)

app = graph.compile()

# -------------------------
# RUN
# -------------------------
if __name__ == "__main__":

    initial_state = {
        "bug": "API returns 500 error when multiple concurrent requests hit the service",
        "research_notes": "",
        "fix": "",
        "test_result": "",
        "iterations": 0
    }

    result = app.invoke(initial_state)

    print("\n--- FINAL RESULT ---")
    print("\nResearch Notes:\n", result["research_notes"])
    print("\nFix:\n", result["fix"])
    print("\nTest Result:\n", result["test_result"])