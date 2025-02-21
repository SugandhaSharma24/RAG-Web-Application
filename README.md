# RAG-Web-Application
This system enables users to query and analyze reports with advanced Retrieval-Augmented Generation (RAG) methods. The app intelligently extracts valuable insights from diverse sources, offering results not only from text but also images and tables within PDFs.
# Project Setup

Before running the application, make sure to install the necessary system dependencies:

1. **Install distutils** (for Python 3.12 or above):
   ```bash
   sudo apt-get update
   sudo apt-get install python3-distutils
2.   Install Python dependencies:
     pip install -r requirements.txt
3. Run
uvicorn app.main:app --host 0.0.0.0 --port 8000
