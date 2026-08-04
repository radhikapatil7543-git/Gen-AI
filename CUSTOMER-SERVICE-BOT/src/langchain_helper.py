import os

from dotenv import load_dotenv

from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import (ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings,)
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Load API key from .env file
load_dotenv()  

# Gemini LLM
llm = ChatGoogleGenerativeAI(model = "gemini-3.1-flash-lite", temperature = 0.1, google_api_key = os.getenv("GOOGLE_API_KEY"),)

# Gemini Embeddings
embeddings = GoogleGenerativeAIEmbeddings(model = "models/gemini-embedding-001", google_api_key = os.getenv("GOOGLE_API_KEY"),)

# FAISS database location
vectordb_file_path = "faiss_index"

# Create Vector Database
def create_vector_db():

    loader = CSVLoader(file_path = "../data/dataset.csv", source_column = "prompt", encoding = "utf-8",)
    data = loader.load()

    vectordb = FAISS.from_documents(documents = data, embedding = embeddings,)

    vectordb.save_local(vectordb_file_path)
    print("Knowledge Base Created Successfully")


def get_qa_chain():
    
    vectordb = FAISS.load_local(vectordb_file_path, embeddings, allow_dangerous_deserialization = True)

    retriever = vectordb.as_retriever(search_kwargs={"k":3})

    prompt_template = """

    You are a helpful AI customer service assistant.

    First, use the information provided in the context to answer the user's question.

    If the context does not contain the answer, then answer using your general knowledge.

    If you are unsure, clearly say that you are not certain.
    
    CONTEXT: {context}

    QUESTION: {question}

    Answer: """

    PROMPT = PromptTemplate(
        template = prompt_template, input_variables = ["context", "question"],
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
    )

    return chain

if __name__ == "__main__":
    create_vector_db()
    chain = get_qa_chain()
    response = chain.invoke({"query": "Do you provide job assistance?"})
    print("\n===== ANSWER =====")
    print(response["result"])
