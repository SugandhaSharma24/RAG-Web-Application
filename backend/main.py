from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
from .rag_system import process_pdf, query_rag_system
import sys
from fastapi.responses import StreamingResponse, JSONResponse
import matplotlib.pyplot as plt
import io
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_model import query_rag_system 


# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



app = FastAPI()

# Allow all origins, you can restrict to specific origins if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)

# Define the folder for storing uploaded PDFs
UPLOAD_FOLDER = "uploaded_pdfs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Limit the file size (example: 10MB)
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Utility function to check if the file is a PDF
def is_pdf(file: UploadFile):
    return file.content_type == "application/pdf"

# Upload PDF and process it
@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        # Save file
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(file.file.read())

        # Process the uploaded file
        process_pdf(file_path)

        return {"message": "PDF processed successfully", "filename": file.filename}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")



# Visualization: Return text distribution across pages
@app.get("/visualize_text_distribution")
async def visualize_text_distribution():
    # Simulate text extraction results
    text_data = process_pdf("sample.pdf")  # Replace with actual stored results
    
    # Count words per page
    word_counts = [len(doc.page_content.split()) for doc in text_data._collection.get()["documents"]]
    
    # Plot
    plt.figure(figsize=(8, 5))
    plt.bar(range(1, len(word_counts) + 1), word_counts, color='skyblue')
    plt.xlabel("Page Number")
    plt.ylabel("Word Count")
    plt.title("Text Distribution Across Pages")

    # Convert plot to PNG image and return
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    return StreamingResponse(buf, media_type="image/png")

# API to return JSON Data for frontend charts
@app.get("/chart-data")
async def get_chart_data():
    text_data = process_pdf("sample.pdf")  # Replace with stored results
    word_counts = [len(doc.page_content.split()) for doc in text_data._collection.get()["documents"]]

    chart_data = {"pages": list(range(1, len(word_counts) + 1)), "word_counts": word_counts}
    return JSONResponse(content=chart_data)

# Query the RAG system
class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def get_rag_response(request: QueryRequest):
    try:
        response = query_rag_system(request.query)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in query processing: {str(e)}")

    

