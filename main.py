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

app = FastAPI()


DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


app.mount("/data", StaticFiles(directory="data"), name="data")
app.mount("/results", StaticFiles(directory="results"), name="results")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
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
        
        model_path = 'models/bestNew.pt'
        yolo_output = get_max_confidence_boxes(model_path, str(file_path))
        
        
        value, min_val, max_val = read_gauge(str(file_path), yolo_output)
        
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