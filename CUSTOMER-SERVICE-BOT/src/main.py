import streamlit as st
from langchain_helper import get_qa_chain

st.set_page_config("CUSTOMER SERVICE CHATBOT", page_icon = "🤖", layout = "centered")

st.title("CUSTOMER SERVICE CHATBOT 🤖")
st.write("Welcome! Ask any question about our services.")

question = st.text_input("Enter your question:")

if st.button("Ask"):
    if question.strip() == "":
        st.warning("Please enter a question.")
    else:
        chain = get_qa_chain()
        response = chain.invoke({"query":question})
        st.subheader("Answer")
        st.write(response["result"])
