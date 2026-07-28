# Q&A Chatbot
# from langchain.llms import OpenAI

from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

import streamlit as st
import os
from PIL import Image
from google import genai

#Create Gemini client
client = genai.Client(api_key = os.getenv('GOOGLE_API_KEY'))

## Function to get Gemini respones

def get_gemini_response(input, image):
    
    if input.strip():
       contents = [input, image]
    else:
       contents = [image]

       response = client.models.generate_content(model = 'gemini-3.1-flash-lite', contents = contents)
       return response.text
 

##initialize our streamlit app

st.set_page_config(page_title="Gemini vision bot Demo")

st.header("Gemini Application")
input = st.text_input("Input Prompt: ", key="input")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
image = ""
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", width="stretch")

submit = st.button("Tell me about the image")

## If ask button is clicked

if submit:

 if uploaded_file is None:
    st.warning("Please upload an image first.")
 else:
    response = get_gemini_response(input, image)
    st.subheader("The Response is")
    st.write(response) 
