
# pip install langchain langchain-openai faiss-cpu

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# ---------------------------------------------
# 1️⃣ Create sample data for knowledge base
# ---------------------------------------------
bank_docs = [
    Document(page_content="You can check your account balance anytime using the mobile app."),
    Document(page_content="KYC verification is mandatory for all new customers."),
    Document(page_content="Interest rate for savings accounts is currently 4% per annum."),
    Document(page_content="Loan approvals depend on credit history and income stability."),
]

# ---------------------------------------------
# 2️⃣ Create embeddings and FAISS vector store
# ---------------------------------------------
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
split_docs = text_splitter.split_documents(bank_docs)

vectorstore = FAISS.from_documents(split_docs, embeddings)

# ---------------------------------------------
# 3️⃣ Create Conversational Memory
# ---------------------------------------------
memory = ConversationBufferMemory(
    memory_key="chat_history", 
    return_messages=True
)

# ---------------------------------------------
# 4️⃣ Create Conversational Retrieval Chain
# ---------------------------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
    memory=memory
)

# ---------------------------------------------
# 5️⃣ Run conversation (simulate chat)
# ---------------------------------------------
print("=== 🧠 Smart Banking Agent (with Context & Memory) ===\n")

queries = [
    "Hi, I’m Venkatesh. I want to open a new account.",
    "What documents do I need for KYC?",
    "Also, what’s my savings interest rate?",
    "Can you remind me what my name is?"
]

for q in queries:
    result = qa_chain.invoke({"question": q})
    print(f"👤 {q}")
    print(f"🤖 {result['answer']}\n")
