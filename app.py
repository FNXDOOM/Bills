# app.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from werkzeug.utils import secure_filename # More robust filename sanitization
from dotenv import load_dotenv
from PIL import Image # Pillow for image handling if needed by Gemini lib
import io # For handling file streams
import shutil
import os
import uuid
import json
import google.generativeai as genai
from datetime import datetime
from typing import List, Optional, Dict, Any 

# --- Configuration & Initialization ---
load_dotenv() # Load environment variables from .env file

# Configure Google Gemini AI
try:
    GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GEMINI_API_KEY:
        raise ValueError("GOOGLE_API_KEY not found in environment variables.")
    genai.configure(api_key=GEMINI_API_KEY)
    
    # using gemini 1.5 model
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
    print("Gemini AI configured successfully.")
except Exception as e:
    print(f"CRITICAL: Error configuring Gemini AI: {e}")
    gemini_model = None # None if configuration fails

# Initialize FastAPI
app = FastAPI(title="Bill Processor API")

# Configure CORS (Allow all origins for simplicity, restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define directories
UPLOAD_DIR = "uploads"
PROCESSED_DIR = "processed" # Maybe useful later
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'pdf'}

# Create necessary directories
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)


# --- Helper Functions ---
def allowed_file(filename):
    """Checks if the filename has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

async def extract_bill_data_with_gemini(file_content: bytes, mime_type: str) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Sends file content to Gemini AI for bill data extraction.

    Args:
        file_content: The byte content of the uploaded file.
        mime_type: The MIME type of the file.

    Returns:
        A tuple containing:
        - dict: Extracted data as a dictionary if successful.
        - str: An error message string if unsuccessful.
        Returns (None, error_message) on failure, (extracted_data, None) on success.
    """
    if not gemini_model:
        return None, "Gemini AI model is not configured or failed to initialize."

    print(f"Processing file with MIME type: {mime_type} using Gemini AI...")

    # Prepare the file part for the Gemini API
    # Ensure the MIME type is one supported by the Gemini File API
    if not mime_type.startswith(('image/', 'application/pdf')):
         return None, f"Unsupported file type for Gemini processing: {mime_type}"
    file_part = {"mime_type": mime_type, "data": file_content}

    # prompt for Gemini api 
    prompt = """
    You are an expert data extraction assistant specialized in invoices and bills.
    Analyze the provided document content (image or PDF).
    Extract the following information accurately:
    - Vendor Name (vendor_name)
    - Invoice Number (invoice_number)
    - Invoice Date (invoice_date) in YYYY-MM-DD format. Attempt conversion if needed.
    - Total Amount Due (total_amount) as a standard decimal number (e.g., 123.45). Remove currency symbols.

    Provide the output *strictly and only* in JSON format using the following exact keys:
    {
      "vendor_name": "...",
      "invoice_number": "...",
      "invoice_date": "YYYY-MM-DD",
      "total_amount": 0.00
    }

    If you cannot confidently find a specific piece of information, use null as the value for that field.
    Do not include any text before or after the JSON object, including markdown formatting like ```json.
    """

    try:
        # Call the Gemini API
        response = await gemini_model.generate_content_async([prompt, file_part]) # Use async version

        # Attempt to parse the JSON response
        json_text = response.text.strip()
        # Basic check for potential markdown (though prompt discourages it)
        if json_text.startswith("```json"):
            json_text = json_text[7:]
        if json_text.endswith("```"):
            json_text = json_text[:-3]
        json_text = json_text.strip()

        if not json_text:
             print("Warning: Gemini returned an empty response.")
             return None, "Extraction failed: AI returned an empty response."

        extracted_data = json.loads(json_text)

        # Basic validation (ensure expected keys exist, even if null)
        required_keys = {"vendor_name", "invoice_number", "invoice_date", "total_amount"}
        if not required_keys.issubset(extracted_data.keys()):
            print(f"Warning: Gemini response missing expected keys. Raw Text: {response.text} | Parsed: {extracted_data}")
            # Optionally, create a default structure with nulls
            default_data = {k: extracted_data.get(k) for k in required_keys}
            return default_data, "Extraction Warning: AI response structure incomplete, using null for missing fields."


        print(f"Gemini raw response text: {response.text}") # For debugging
        print(f"Successfully extracted data: {extracted_data}")
        return extracted_data, None # Success

    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode JSON from Gemini response: {e}")
        print(f"Gemini raw response: {response.text}")
        return None, f"Extraction failed: Invalid JSON response from AI. Raw text: {response.text}"
    except Exception as e:
        # Handle potential API errors (e.g., safety blocks, quota issues)
        # Check response attributes for more details if needed (response.prompt_feedback, response.candidates)
        print(f"Error during Gemini API call or processing: {e}")
        # Provide a more user-friendly error
        error_detail = f" ({type(e).__name__})" if str(e) else "" # Add exception type if no specific message
        return None, f"Extraction failed: An error occurred during AI processing{error_detail}."


# --- API Endpoints ---
@app.post("/upload-bill/")
async def upload_bill(
    file: UploadFile = File(...),
    company_name: Optional[str] = Form(None), # Use Optional for non-required fields
    notes: Optional[str] = Form(None)
):
    """
    Upload a bill file (PDF, JPG, PNG, WEBP) for processing with Gemini AI.
    """
    # --- Input Validation ---
    if not allowed_file(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # --- File Handling ---
    filename_secure = secure_filename(file.filename) # Sanitize filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_id = str(uuid.uuid4())[:8]
    extension = os.path.splitext(filename_secure)[1]
    saved_filename = f"{timestamp}_{file_id}{extension}"
    file_path = os.path.join(UPLOAD_DIR, saved_filename)

    try:
        # Read file content once
        file_content = await file.read()
        await file.seek(0) # Reset file pointer if needed elsewhere, though not strictly necessary here

        # Save the file asynchronously (optional but good practice)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

    except Exception as e:
        print(f"Error saving file {filename_secure}: {e}")
        raise HTTPException(status_code=500, detail=f"Could not save file: {filename_secure}")
    finally:
        await file.close() # Ensure file is closed

    # --- AI Processing ---
    extracted_data, processing_error = await extract_bill_data_with_gemini(
        file_content=file_content,
        mime_type=file.content_type # Get MIME type from UploadFile
    )

    # --- Prepare Response ---
    response_data = {
        "upload_status": "success",
        "message": "Bill uploaded successfully.",
        "file_id": file_id,
        "filename": saved_filename,
        "original_filename": file.filename,
        "content_type": file.content_type,
        "company_name": company_name,
        "notes": notes,
        "processing_status": "error" if processing_error else "success",
        "processing_message": processing_error,
        "extracted_data": extracted_data # Will be None if processing failed
    }

    if processing_error:
        print(f"AI Processing Error for {saved_filename}: {processing_error}")
        # Log the error, but return 200 OK as upload succeeded, processing failed
        # Alternatively, could return 207 Multi-Status or handle differently

    return response_data


@app.get("/")
async def root():
    """Root endpoint providing basic API status."""
    return {"message": "Bill Processing API with Gemini is running"}

if __name__ == "__main__":
    import uvicorn
    # Bind to 0.0.0.0 to make accessible on the network if needed
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)