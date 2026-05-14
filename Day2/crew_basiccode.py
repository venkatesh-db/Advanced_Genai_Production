

# crew_basiccode.py

from crewai import Agent
from crewai import Task
from crewai import Crew

# -----------------------------------
# Architect Agent
# -----------------------------------
architect = Agent(
    role="Software Architect",

    goal="""
    Design scalable Kubernetes
    architecture
    """,

    backstory="""
    Expert in distributed systems,
    Kubernetes,
    scalability,
    and cloud architecture.
    """,

    verbose=True
)

# -----------------------------------
# Coder Agent
# -----------------------------------
coder = Agent(
    role="DevOps Engineer",

    goal="""
    Create deployment strategy
    for Kubernetes platform
    """,

    backstory="""
    Expert in Docker,
    Kubernetes,
    CI/CD,
    and cloud deployment.
    """,

    verbose=True
)

# -----------------------------------
# Reviewer Agent
# -----------------------------------
reviewer = Agent(
    role="Infrastructure Reviewer",

    goal="""
    Review architecture
    for scalability and security
    """,

    backstory="""
    Expert in infrastructure
    reliability,
    security,
    and production systems.
    """,

    verbose=True
)

# -----------------------------------
# Task 1
# -----------------------------------
architecture_task = Task(
    description="""
    Design scalable Kubernetes
    architecture for
    microservices application.
    """,

    expected_output="""
    Detailed Kubernetes
    architecture plan.
    """,

    agent=architect
)

# -----------------------------------
# Task 2
# -----------------------------------
deployment_task = Task(
    description="""
    Create deployment strategy
    based on architecture.
    """,

    expected_output="""
    Kubernetes deployment
    implementation strategy.
    """,

    agent=coder
)

# -----------------------------------
# Task 3
# -----------------------------------
review_task = Task(
    description="""
    Review deployment architecture
    for:
    - scalability
    - reliability
    - security
    """,

    expected_output="""
    Infrastructure review report
    with recommendations.
    """,

    agent=reviewer
)

# -----------------------------------
# Crew
# -----------------------------------
crew = Crew(
    agents=[
        architect,
        coder,
        reviewer
    ],

    tasks=[
        architecture_task,
        deployment_task,
        review_task
    ],

    verbose=True
)

# -----------------------------------
# RUN
# -----------------------------------
result = crew.kickoff()

print("\n===== FINAL OUTPUT =====\n")

print(result)