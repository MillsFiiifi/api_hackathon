import os
import base64
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Optional
import cv2
import numpy as np
import tempfile

# Load environment variables
load_dotenv()

# Initialize OpenAI client
MODEL = 'GPT-4.1'

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("CHRISKEY")
)

# Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_methods=['*'],
    allow_credentials=True,
    allow_headers=['*'],
    allow_origins=['*'],
)

# Supported crops (can be expanded)
SUPPORTED_CROPS = [
    "rice", "maize", "tomato", "cassava", "yam", 
    "cocoa", "plantain", "banana", "pepper", "eggplant",
    "cabbage", "okra", "beans", "sorghum", "millet"
]

class ImageResponse(BaseModel):
    suggestions: str
    disease_info: Optional[Dict] = None
    translation: Optional[Dict] = None
    identified_crop: Optional[str] = None

class ChatbotResponse(BaseModel):
    response: str
    translation: Optional[Dict] = None

class WeatherInsightsResponse(BaseModel):
    insights: str
    translation: Optional[Dict] = None

class VideoResponse(BaseModel):
    analysis: List[Dict]
    translation: Optional[Dict] = None
    identified_crop: Optional[str] = None

def encode_image(image_file):
    return base64.b64encode(image_file).decode("utf-8")

def identify_crop(image_base64: str) -> str:
    """Identify the crop from the image"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are an agricultural expert. Identify the crop plant in this image from these options: " + ", ".join(SUPPORTED_CROPS)},
            {"role": "user", "content": [
                {"type": "text", "text": "What crop plant is this? Respond with just the crop name."},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{image_base64}"}
                }
            ]}
        ],
        temperature=0.0,
    )
    identified_crop = response.choices[0].message.content.lower()
    return identified_crop if identified_crop in SUPPORTED_CROPS else "unknown"

def translate_to_ghanaian(text: str, target_language: str = "twi") -> Dict:
    """Translate text to Ghanaian languages"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": f"You are a professional translator that translates English to Ghanaian languages. Translate accurately while maintaining agricultural terminology."},
            {"role": "user", "content": f"Translate this agricultural text to {target_language}:\n\n{text}"}
        ],
        temperature=0.0,
    )
    return {
        "original": text,
        "translated": response.choices[0].message.content,
        "language": target_language
    }

def get_disease_info(crop: str, disease: str) -> Dict:
    """Get structured disease information"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": f"You are an agricultural expert. Provide detailed information about {disease} in {crop} with this structure: description, symptoms, treatments, prevention."},
            {"role": "user", "content": f"Provide detailed information about {disease} in {crop}"}
        ],
        temperature=0.0,
    )
    return parse_disease_response(response.choices[0].message.content)

def parse_disease_response(text: str) -> Dict:
    """Parse the disease information into structured format"""
    sections = ["description", "symptoms", "treatments", "prevention"]
    result = {}
    current_section = None
    
    for line in text.split('\n'):
        line = line.strip().lower()
        if any(section in line for section in sections):
            current_section = next(section for section in sections if section in line)
            result[current_section] = []
        elif current_section and line:
            result[current_section].append(line)
    
    return result

def analyze_video_frames(video_path: str, crop_type: Optional[str] = None) -> List[Dict]:
    """Analyze video frames for disease detection"""
    cap = cv2.VideoCapture(video_path)
    frame_analyses = []
    frame_count = 0
    sample_rate = 10  # Analyze every 10th frame
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        if frame_count % sample_rate != 0:
            continue
            
        # Convert frame to base64
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Identify crop if not specified
        current_crop = crop_type if crop_type else identify_crop(frame_base64)
        
        # Analyze frame
        analysis = generate_suggestions(frame_base64, current_crop)
        frame_analyses.append({
            "frame": frame_count,
            "analysis": analysis,
            "diseases": extract_diseases(analysis),
            "identified_crop": current_crop
        })
    
    cap.release()
    return frame_analyses

def extract_diseases(text: str) -> List[str]:
    """Extract disease names from analysis text"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Extract plant disease names from this text. Return as JSON array."},
            {"role": "user", "content": text}
        ],
        temperature=0.0,
    )
    try:
        return eval(response.choices[0].message.content)
    except:
        return []

def generate_suggestions(image_base64: str, crop_type: str) -> str:
    """Generate suggestions for the uploaded image"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": f"You are a helpful plant doctor that provides suggestions for {crop_type} crops based on images of their leaves or other plant parts."},
            {"role": "user", "content": [
                {"type": "text", "text": f"Please analyze the condition of this {crop_type} plant and identify any diseases or issues."},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{image_base64}"}
                }
            ]}
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content

@app.post("/analyze-plant-image", response_model=ImageResponse)
async def analyze_plant_image(
    file: UploadFile = File(...),
    crop_type: Optional[str] = Form(None),
    language: Optional[str] = Form(None)
):
    image_data = await file.read()
    image_base64 = encode_image(image_data)
    
    # Identify crop if not specified
    identified_crop = crop_type if crop_type else identify_crop(image_base64)
    
    if identified_crop == "unknown":
        return JSONResponse(
            content={"error": "Could not identify the crop. Please specify the crop type."},
            status_code=400
        )
    
    suggestions = generate_suggestions(image_base64, identified_crop)
    
    # Extract potential diseases
    diseases = extract_diseases(suggestions)
    disease_info = {}
    
    for disease in diseases:
        disease_info[disease] = get_disease_info(identified_crop, disease)
    
    # Prepare response
    response_data = {
        "suggestions": suggestions,
        "disease_info": disease_info,
        "identified_crop": identified_crop
    }
    
    # Add translation if requested
    if language:
        translation = translate_to_ghanaian(suggestions, language)
        response_data["translation"] = translation
    
    return JSONResponse(content=response_data)

@app.post("/analyze-plant-video", response_model=VideoResponse)
async def analyze_plant_video(
    file: UploadFile = File(...),
    crop_type: Optional[str] = Form(None),
    language: Optional[str] = Form(None)
):
    # Save video to temporary file
    with tempfile.NamedTemporaryFile(delete=False) as temp_video:
        temp_video.write(await file.read())
        temp_video_path = temp_video.name
    
    # Analyze video frames
    analysis = analyze_video_frames(temp_video_path, crop_type)
    
    # Clean up
    os.unlink(temp_video_path)
    
    # Get the most frequently identified crop
    identified_crops = [frame.get("identified_crop") for frame in analysis if frame.get("identified_crop")]
    identified_crop = max(set(identified_crops), key=identified_crops.count) if identified_crops else "unknown"
    
    # Prepare response
    response_data = {
        "analysis": analysis,
        "identified_crop": identified_crop if identified_crop != "unknown" else None
    }
    
    # Add translation if requested
    if language:
        summary = "\n".join([f"Frame {item['frame']}: {item['analysis']}" for item in analysis])
        translation = translate_to_ghanaian(summary, language)
        response_data["translation"] = translation
    
    return JSONResponse(content=response_data)

@app.post("/agriculture-chatbot", response_model=ChatbotResponse)
async def agriculture_chatbot(
    query: str = Form(...),
    language: Optional[str] = Form(None)
):
    # Modified to handle all crops
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are an agriculture expert chatbot that provides advice and information to farmers about all crops. Provide detailed, practical information."},
            {"role": "user", "content": query}
        ],
        temperature=0.0,
    )
    chatbot_response = response.choices[0].message.content
    
    response_data = {"response": chatbot_response}
    
    if language:
        response_data["translation"] = translate_to_ghanaian(chatbot_response, language)
    
    return JSONResponse(content=response_data)

# Similarly modify other endpoints to be crop-agnostic
# ...

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)