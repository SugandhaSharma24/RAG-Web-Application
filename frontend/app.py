import streamlit as st
import requests
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import re

# FastAPI backend URL
API_URL = "http://127.0.0.1:8000"  # Update with your FastAPI URL

# Sidebar title and description
st.sidebar.title("Welcome to the PDF OCR & RAG System")
st.sidebar.write("""
This application allows you to upload PDF files, extract their content, and query it using a retrieval-augmented generation (RAG) system. 
Just upload a PDF, ask a question, and get relevant information extracted from the document.
""")

# Function to upload PDF to FastAPI backend
def upload_pdf(file):
    url = f"{API_URL}/upload_pdf"
    files = {"file": file}
    response = requests.post(url, files=files)
    
    if response.status_code == 200:
        return response.json()  # Returns filename and success message
    else:
        st.error(f"Failed to upload PDF: {response.text}")
        return None

# Function to query the RAG system via FastAPI backend
def query_rag_system(query):
    url = f"{API_URL}/query"
    payload = {"query": query}
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        return response.json()["response"]  # Returns query results
    else:
        st.error(f"Error during query: {response.text}")
        return None

# Streamlit UI elements
st.title("PDF OCR & RAG System")

# File upload
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])


# Check if file is uploaded and process it
if uploaded_file is not None:
    st.write(f"File uploaded: {uploaded_file.name}")

    # Upload the PDF file to the FastAPI backend
    upload_response = upload_pdf(uploaded_file)
    if upload_response:
        st.success(upload_response["message"])

    # Query input for the user
    query = st.text_input("Ask a question about the PDF:")

    # Initialize session state for query results if not already initialized
    if "query_results" not in st.session_state:
        st.session_state.query_results = None  # Start with no query results

    # Button to trigger query processing
    if st.button("Submit Query"):
        # Only process if the query is not empty
        if query:
            # Query the RAG system via FastAPI
            query_results = query_rag_system(query)
            
            if query_results:
                # Store the results in session state
                st.session_state.query_results = query_results
                st.write("Results:")
                for result in query_results:
                    st.write(f"Page {result['page']}: {result['text']}")
            else:
                st.write("No relevant information found.")
        else:
            st.warning("Please enter a query.")
 
       


