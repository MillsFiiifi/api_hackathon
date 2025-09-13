import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import pdfplumber
import docx
from fpdf import FPDF
import requests
import json
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv
import shutil
from datetime import datetime, timedelta
import pptx  # For PowerPoint file processing

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="MCQ Generator API",
    description="API for generating multiple-choice questions from documents",
    version="1.0.0"
)

# Configure CORS (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'docx', 'ppt', 'pptx'}  # Added PowerPoint formats
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Create directories if they don't exist
Path(UPLOAD_FOLDER).mkdir(exist_ok=True)
Path(RESULTS_FOLDER).mkdir(exist_ok=True)

# Mount static files (for serving generated files)
app.mount("/results", StaticFiles(directory="results"), name="results")

# OpenRouter API configuration
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
YOUR_SITE_URL = os.getenv("YOUR_SITE_URL", "http://localhost:8000")
YOUR_SITE_NAME = os.getenv("YOUR_SITE_NAME", "MCQ Generator")

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_file(file_path: str) -> Optional[str]:
    try:
        ext = file_path.rsplit('.', 1)[1].lower()
        if ext == 'pdf':
            with pdfplumber.open(file_path) as pdf:
                text = ''.join([page.extract_text() or '' for page in pdf.pages])
            return text
        elif ext == 'docx':
            doc = docx.Document(file_path)
            text = ' '.join([para.text for para in doc.paragraphs])
            return text
        elif ext in ['ppt', 'pptx']:
            # Extract text from PowerPoint presentations
            presentation = pptx.Presentation(file_path)
            text = []
            for slide in presentation.slides:
                for shape in slide.shapes:
                    if hasattr(shape, "text"):
                        text.append(shape.text)
            return ' '.join(text)
        elif ext == 'txt':
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        return None
    except Exception as e:
        print(f"Error extracting text: {e}")
        return None

def generate_mcqs_with_openrouter(input_text: str, num_questions: int) -> str:
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="OpenRouter API key not configured")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": YOUR_SITE_URL,
        "X-Title": YOUR_SITE_NAME,
        "Content-Type": "application/json"
    }
    
    prompt = f"""
    Generate {num_questions} multiple-choice questions (MCQs) based on the following text.
    For each question, provide:
    - A clear question
    - Four answer options (labeled A, B, C, D)
    - The correct answer clearly indicated TWICE (repeat it)
    
    Format each question like this:
    ## MCQ
    Question: [question text]
    A) [option A]
    B) [option B]
    C) [option C]
    D) [option D]
    Correct Answer: [correct option]
    Correct Answer: [correct option]  # Repeat the correct answer
    
    Text to generate questions from:
    {input_text}
    """
    
    payload = {
        "model": os.getenv("OPENROUTER_MODEL", "openai/gpt-4"),
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 2000
    }
    
    try:
        response = requests.post(
            url=OPENROUTER_API_URL,
            headers=headers,
            data=json.dumps(payload),
            timeout=30
        )
        response.raise_for_status()
        
        generated_content = response.json()['choices'][0]['message']['content']
        return generated_content.strip()
        
    except requests.exceptions.RequestException as e:
        print(f"OpenRouter API error: {e}")
        raise HTTPException(status_code=502, detail=f"Error generating MCQs: {str(e)}")

def save_mcqs_to_file(mcqs: str, filename: str) -> str:
    results_path = os.path.join(RESULTS_FOLDER, filename)
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write(mcqs)
    return results_path

def create_pdf(mcqs: str, filename: str) -> str:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    for mcq in mcqs.split("## MCQ"):
        if mcq.strip():
            pdf.multi_cell(0, 10, mcq.strip())
            pdf.ln(5)

    pdf_path = os.path.join(RESULTS_FOLDER, filename)
    pdf.output(pdf_path)
    return pdf_path

def cleanup_old_files(directory: str, days: int = 1):
    """Remove files older than X days"""
    now = datetime.now()
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            modified_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            if now - modified_time > timedelta(days=days):
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"Error deleting file {file_path}: {e}")

@app.post("/generate")
async def generate_mcqs(
    file: UploadFile = File(...),
    num_questions: int = Form(..., gt=0, le=50)
):
    # Validate file
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="Invalid file format. Only PDF, TXT, DOCX, PPT, and PPTX files are allowed.")

    # Check file size
    file_size = 0
    temp_path = os.path.join(UPLOAD_FOLDER, f"temp_{file.filename}")
    
    try:
        with open(temp_path, "wb") as buffer:
            while content := await file.read(1024 * 1024):  # Read in 1MB chunks
                file_size += len(content)
                if file_size > MAX_FILE_SIZE:
                    raise HTTPException(status_code=413, detail="File too large. Max size is 10MB.")
                buffer.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")
    finally:
        await file.close()

    # Extract text
    text = extract_text_from_file(temp_path)
    os.remove(temp_path)  # Clean up temp file
    if not text:
        raise HTTPException(status_code=400, detail="Could not extract text from the file.")

    # Generate MCQs
    try:
        mcqs = generate_mcqs_with_openrouter(text, num_questions)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating MCQs: {str(e)}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    base_name = os.path.splitext(file.filename)[0]
    txt_filename = f"mcqs_{base_name}_{timestamp}.txt"
    pdf_filename = f"mcqs_{base_name}_{timestamp}.pdf"
    
    txt_path = save_mcqs_to_file(mcqs, txt_filename)
    pdf_path = create_pdf(mcqs, pdf_filename)

    # Cleanup old files
    cleanup_old_files(UPLOAD_FOLDER)
    cleanup_old_files(RESULTS_FOLDER, days=1)

    return JSONResponse({
        "mcqs": mcqs,
        "txt_url": f"/results/{txt_filename}",
        "pdf_url": f"/results/{pdf_filename}",
        "message": "MCQs generated successfully"
    })

@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join(RESULTS_FOLDER, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, filename=filename)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("RELOAD", "false").lower() == "true"
    )
