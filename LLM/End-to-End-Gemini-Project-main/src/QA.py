from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

import streamlit as st
import os

from google import genai

client = genai.Client(api_key = os.getenv("GOOGLE_API_KEY"))

def get_gemini_response(question):

    response = client.models.generate_content(model = 'gemini-3.1-flash-lite', contents = question)
    return response.text


##initialize our streamlit app

st.set_page_config(page_title="GEMINI CHATBOT DEMO")

st.header("Gemini Application")

input = st.text_input("Input: ", key="input")

submit = st.button("Ask the question")

## If ask button is clicked

if submit:

    response = get_gemini_response(input)
    st.subheader("The Response is")
    st.write(response)
