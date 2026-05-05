

# production_multiagent_isolated.py

import os
import json
import subprocess
import tempfile
from typing import TypedDict, Dict, Any

from langgraph.graph import StateGraph, END
from crewai import Agent, Task, Crew

# -------------------------
# Config
# -------------------------
MAX_ITERS = 3
TIMEOUT = 20

# -------------------------
# State
# -------------------------
class RnDState(TypedDict):
    bug: str
    research_notes: str
    fix_code: str
    test_code: str
    test_stdout: str
    test_stderr: str
    verdict: str
    iterations: int

# -------------------------
# Helpers
# -------------------------
def extract_json(text: str, default: Dict[str, Any]):
    try:
        s = text.find("{")
        e = text.rfind("}")
        if s != -1 and e != -1:
            return json.loads(text[s:e+1])
    except:
        pass
    return default

def ensure_imports(test_code: str):
    if "from solution import" not in test_code:
        return "from solution import *\n\n" + test_code
    return test_code

def ensure_pytest_installed():
    try:
        subprocess.run(["python3", "-m", "pytest", "--version"], capture_output=True)
    except:
        subprocess.run(["python3", "-m", "pip", "install", "pytest"], check=True)

def run_crew(agent, task):
    try:
        return str(Crew(agents=[agent], tasks=[task], verbose=True).kickoff())
    except Exception as e:
        return f"ERROR: {e}"

# -------------------------
# Execution
# -------------------------
def run_pytest(module_code, test_code):
    ensure_pytest_installed()

    with tempfile.TemporaryDirectory() as td:
        module_path = os.path.join(td, "solution.py")
        test_path = os.path.join(td, "test_solution.py")

        with open(module_path, "w") as f:
            f.write(module_code)

        with open(test_path, "w") as f:
            f.write(test_code)

        try:
            proc = subprocess.run(
                ["python3", "-m", "pytest", "-q"],
                cwd=td,
                capture_output=True,
                text=True,
                timeout=TIMEOUT
            )
            return proc.stdout, proc.stderr
        except subprocess.TimeoutExpired:
            return "", "TIMEOUT"

# -------------------------
# Agents
# -------------------------
research_agent = Agent(
    role="Research Engineer",
    goal="Find root causes of concurrency bugs",
    backstory="Expert in multithreading, race conditions, and scaling"
)

debug_agent = Agent(
    role="Senior Debug Engineer",
    goal="Produce FIX with NO global state and pytest tests",
    backstory="Writes production-safe concurrent systems"
)

# -------------------------
# Nodes
# -------------------------
def research_node(state: RnDState):
    task = Task(
        description=f"""
Analyze bug:

{state['bug']}

Return JSON:
{{ "root_causes": [], "strategy": "" }}
""",
        agent=research_agent,
        expected_output="Valid JSON"
    )

    result = run_crew(research_agent, task)
    data = extract_json(result, {"root_causes": [], "strategy": result})

    return {
        **state,
        "research_notes": json.dumps(data, indent=2),
        "iterations": state["iterations"] + 1
    }

def debug_node(state: RnDState):
    task = Task(
        description=f"""
Bug:
{state['bug']}

Research:
{state['research_notes']}

STRICT RULES:
- DO NOT use global variables
- Use dependency injection (classes/instances)
- Each test must be isolated
- Use thread-safe design
- Provide FULL working code
- Provide pytest tests ONLY
- Tests MUST include: from solution import *

Return JSON:
{{ "module_code": "", "tests": "" }}
""",
        agent=debug_agent,
        expected_output="Valid JSON"
    )

    result = run_crew(debug_agent, task)
    data = extract_json(result, {"module_code": "", "tests": ""})

    module_code = data["module_code"]
    test_code = ensure_imports(data["tests"])

    return {
        **state,
        "fix_code": module_code,
        "test_code": test_code
    }

def execute_node(state: RnDState):
    stdout, stderr = run_pytest(state["fix_code"], state["test_code"])
    return {
        **state,
        "test_stdout": stdout,
        "test_stderr": stderr
    }

def qa_node(state: RnDState):
    out = state["test_stdout"].lower()
    err = state["test_stderr"].lower()

    # 🔥 guardrail: reject globals
    if "global " in state["fix_code"]:
        return {**state, "verdict": "FAIL"}

    if "failed" in out or "error" in err:
        verdict = "FAIL"
    elif "passed" in out:
        verdict = "PASS"
    else:
        verdict = "LIKELY_PASS"

    return {**state, "verdict": verdict}

# -------------------------
# Control
# -------------------------
def should_continue(state: RnDState):
    if state["verdict"] == "PASS":
        return "end"
    if state["iterations"] >= MAX_ITERS:
        return "end"
    return "loop"

# -------------------------
# Graph
# -------------------------
graph = StateGraph(RnDState)

graph.add_node("research", research_node)
graph.add_node("debug", debug_node)
graph.add_node("execute", execute_node)
graph.add_node("qa", qa_node)

graph.set_entry_point("research")

graph.add_edge("research", "debug")
graph.add_edge("debug", "execute")
graph.add_edge("execute", "qa")

graph.add_conditional_edges(
    "qa",
    should_continue,
    {"loop": "research", "end": END}
)

app = graph.compile()

# -------------------------
# RUN
# -------------------------
if __name__ == "__main__":
    state = {
        "bug": "API returns 500 error under concurrent requests",
        "research_notes": "",
        "fix_code": "",
        "test_code": "",
        "test_stdout": "",
        "test_stderr": "",
        "verdict": "FAIL",
        "iterations": 0
    }

    result = app.invoke(state)

    print("\n===== FINAL RESULT =====")
    print("Verdict:", result["verdict"])
    print("\nTest Output:\n", result["test_stdout"])
    print("\nErrors:\n", result["test_stderr"])