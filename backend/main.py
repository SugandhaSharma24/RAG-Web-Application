from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
from .rag_system import process_pdf, query_rag_system  # Assuming rag_system.py exists and has these functions
import sys
from fastapi.responses import StreamingResponse, JSONResponse
import matplotlib.pyplot as plt
import io
import json
import shutil
# Define the folder for storing uploaded PDFs
UPLOAD_FOLDER = "uploaded_pdf"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure the folder is created once

# Limit the file size (example: 10MB)
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Create the FastAPI app instance
app = FastAPI()

# CORS middleware settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)

# Utility function to check if the file is a PDF
def is_pdf(file: UploadFile):
    return file.content_type == "application/pdf"

# Upload PDF and process it
@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    contents = await file.read()
    # Process the PDF here (e.g., saving, parsing)
    return {"filename": file.filename, "message": "File uploaded successfully"}


# Visualization: Return text distribution across pages
@app.get("/visualize_text_distribution")
async def visualize_text_distribution(pdf_path: str):
    try:
        # Call the `process_pdf` method and get the text data
        db = process_pdf(pdf_path)
        # Simulate text extraction results, adjust based on your return structure
        word_counts = [len(doc.page_content.split()) for doc in db._collection.get()["documents"]]

        # Plot
        plt.figure(figsize=(8, 5))
        plt.bar(range(1, len(word_counts) + 1), word_counts, color='skyblue')
        plt.xlabel("Page Number")
        plt.ylabel("Word Count")
        plt.title("Text Distribution Across Pages")

        # Convert plot to PNG image and return
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating visualization: {str(e)}")

# API to return JSON Data for frontend charts
@app.get("/chart-data")
async def get_chart_data(pdf_path: str):
    try:
        # Call `process_pdf` and get the word counts
        db = process_pdf(pdf_path)
        word_counts = [len(doc.page_content.split()) for doc in db._collection.get()["documents"]]

        chart_data = {"pages": list(range(1, len(word_counts) + 1)), "word_counts": word_counts}
        return JSONResponse(content=chart_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving chart data: {str(e)}")


# Query the RAG system
class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def get_rag_response(request: QueryRequest):
    try:
        # Call `query_rag_system` with the user's query
        response = query_rag_system(request.query)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in query processing: {str(e)}")
