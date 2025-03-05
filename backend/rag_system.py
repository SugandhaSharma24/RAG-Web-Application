import os
import numpy as np
import cv2
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
import streamlit as st
import tempfile



class PDFImageOCRPipeline:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.images = []
        self.extracted_texts = []

    def extract_images_from_pdf(self):
        """Step 1: Extract images from the PDF"""
        self.images = convert_from_path(self.pdf_path)
        return self.images

    def preprocess_image(self, image):
        """Step 2: Preprocess image for better OCR results"""
        open_cv_image = np.array(image)
        gray = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2GRAY)
        processed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        return Image.fromarray(processed)

    def extract_text(self):
        """Step 3: Perform OCR on extracted images"""
        for i, img in enumerate(self.images):
            processed_img = self.preprocess_image(img)
            text = pytesseract.image_to_string(processed_img)
            self.extracted_texts.append({"page": i + 1, "text": text})
        return self.extracted_texts

    def store_text_in_chromadb(self):
        """Step 4: Store extracted text in a vector database (ChromaDB)"""
        documents = [Document(page_content=item["text"], metadata={"page": item["page"]}) for item in self.extracted_texts]
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        split_docs = text_splitter.split_documents(documents)

        # Use Sentence Transformers for efficient embeddings
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # Create Chroma instance, with a directory to store the vectors
        persist_directory = "./chroma_db"  # Directory for storing Chroma database
        vectorstore = Chroma.from_documents(split_docs, embeddings, persist_directory=persist_directory)

        return vectorstore

    def run_pipeline(self):
        """Run the full OCR and vectorization pipeline"""
        print("Extracting images from PDF...")
        self.extract_images_from_pdf()

        print("Extracting text from images using OCR...")
        self.extract_text()

        print("Storing extracted text in vector database...")
        db = self.store_text_in_chromadb()

        print("Pipeline completed successfully!")
        return db

# Function to handle PDF processing
def process_pdf(pdf_file):
    pipeline = PDFImageOCRPipeline(pdf_file)
    return pipeline.run_pipeline()

# Function to handle similarity search
def query_rag_system(query: str, db_path="./chroma_db", top_k=3):
    """Retrieve relevant text snippets from ChromaDB based on a user query."""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)
    retrieved_docs = vectorstore.similarity_search(query, k=top_k)

      # Remove duplicates based on the text content
    unique_results = []
    seen = set()
    for doc in retrieved_docs:
        page_content = doc.page_content
        if page_content not in seen:
            seen.add(page_content)
            unique_results.append({"page": doc.metadata["page"], "text": page_content})
    
    return unique_results

# Streamlit UI for uploading PDF
st.title("PDF OCR & RAG System")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    # Display the file name
    st.write(f"File uploaded: {uploaded_file.name}")

    # Temporary file for processing
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    # Run OCR and vectorization pipeline
    st.text("Processing the PDF...")
    db = process_pdf(temp_file_path)

    # Allow user to ask questions
    query = st.text_input("Ask a question about the PDF:")

    if query:
        results = query_rag_system(query)
        if results:
            for result in results:
                st.write(f"Page {result['page']}: {result['text']}")
        else:
            st.write("No relevant information found.")
