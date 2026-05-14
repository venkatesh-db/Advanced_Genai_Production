
# llm_judge.py

from openai import OpenAI

client = OpenAI()

# -----------------------------------
# Agent LLM
# -----------------------------------
def agent_answer(question):

    response = client.chat.completions.create(

        model="gpt-4.1-mini",

        messages=[
            {
                "role": "user",
                "content": question
            }
        ]
    )

    return response.choices[0].message.content

# -----------------------------------
# Judge LLM
# -----------------------------------
def judge_answer(question, answer):

    prompt = f"""
    Evaluate this AI answer.

    Question:
    {question}

    Answer:
    {answer}

    Score from 1 to 10 for:
    - correctness
    - relevancy
    - clarity

    Return short evaluation.
    """

    response = client.chat.completions.create(

        model="gpt-4.1-mini",

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content

# -----------------------------------
# RUN
# -----------------------------------
if __name__ == "__main__":

    question = """
    How should Kubernetes APIs scale?
    """

    print("\n===== USER QUESTION =====\n")

    print(question)

    # -----------------------------------
    # Agent Generates Answer
    # -----------------------------------
    answer = agent_answer(question)

    print("\n===== AGENT ANSWER =====\n")

    print(answer)

    # -----------------------------------
    # Judge Evaluates Answer
    # -----------------------------------
    evaluation = judge_answer(
        question,
        answer
    )

    print("\n===== JUDGE EVALUATION =====\n")

    print(evaluation)