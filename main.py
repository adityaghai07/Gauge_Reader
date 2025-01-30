from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
import shutil
import os
from datetime import datetime
from pathlib import Path
from services.yolo_functions import get_max_confidence_boxes, read_gauge
from services.digital_yolo_functions import process_digital_gauge_image

app = FastAPI()
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

app.mount("/data", StaticFiles(directory="data"), name="data")
app.mount("/results", StaticFiles(directory="results"), name="results")
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, detail="File must be an image")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    original_filename = Path(file.filename)
    new_filename = f"{timestamp}_{original_filename.stem}{original_filename.suffix}"
    file_path = DATA_DIR / new_filename
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # Try analog gauge detection first
        analog_model_path = r"models\combinedAnalogYOLOv8_float32.tflite"
        yolo_output, isGaugeThreshold , isSquare = get_max_confidence_boxes(analog_model_path, str(file_path))

        

        
        if isGaugeThreshold < 2:
            try:
                # Fall back to digital gauge detection
                digital_model_path = r'models/digitalYOLOv8_float32.tflite'
                digital_output_path = RESULTS_DIR / "digital_detection_result.jpg"
                
                digital_results = process_digital_gauge_image(
                    str(file_path),
                    digital_model_path,
                    str(digital_output_path)
                )
                
                isDigitalGauge = digital_results['isGauge']
                
                if isDigitalGauge < 0.7:
                    return JSONResponse({
                        "success": False,
                        "message": f"Neither analog nor digital gauge detected. Digital confidence: {isDigitalGauge}, Analog confidence: {isGaugeThreshold} found , threshold is 2"
                    })
                
                # If digital gauge is detected successfully
                return JSONResponse({
                    "success": True,
                    "display_text": digital_results['display_text'],
                    "image_path": f"/data/{new_filename}",
                    "detection_result": "/results/digital_detection_result.jpg"
                })
                
            except Exception as e:
                raise HTTPException(500, detail=f"Digital gauge processing failed: {str(e)}")
        
        # Process analog gauge if threshold is met
        value, min_val, max_val = read_gauge(str(file_path), yolo_output , isSquare)
        return JSONResponse({
            "success": True,
            "value": round(value, 1),
            "min_val": min_val,
            "max_val": max_val,
            "image_path": f"/data/{new_filename}",
            "detection_result": "/results/detection_result.jpg",
            "gauge_debug": "/results/gauge_debug.jpg"
        })
        
    except Exception as e:
        raise HTTPException(500, detail=str(e))