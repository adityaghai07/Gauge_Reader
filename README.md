# Gauge Reader

A web-based tool for automatic gauge reading using computer vision and deep learning. This application uses YOLOv8 object detection to locate gauges in images and processes them to extract readings automatically.

![Gauge Reader Results]("images\Screenshot 2024-12-23 125448.png")

## Features

- Upload images of gauges through a web interface
- Automatic gauge detection using YOLO
- Real-time gauge reading extraction
- Visual feedback with detection and debug images
- Support for various gauge types
- Range detection and value extraction

## Installation

### Setup Environment

1. Create a virtual environment:
```bash
python -m venv gauge_env
```

2. Activate the virtual environment:

On Windows:
```bash
gauge_env\Scripts\activate
```

On Unix or MacOS:
```bash
source gauge_env/bin/activate
```

3. Clone the repository:
```bash
git clone https://github.com/adityaghai07/Gauge_Reader.git
cd Gauge_Reader
```

4. Install requirements:
```bash
pip install -r requirements.txt
```



## Usage

1. Start the server:
```bash
uvicorn main:app --reload
```

2. Open your web browser and navigate to:
```
http://localhost:8000
```

3. Use the interface to:
   - Upload an image of a gauge
   - View the detection results
   - See the extracted gauge reading and range
   - Examine the debug visualization

## Output Images

The tool generates two types of output images:

1. `detection_result.jpg`: Shows the YOLO model's gauge detection
2. `gauge_debug.jpg`: Displays the gauge reading process visualization

## Customization

### Styling
- Modify `static/styles.css` to customize the appearance
- The interface is responsive and adapts to different screen sizes

### Model
- Replace `bestNew.pt` with your own trained YOLOv8 model
- Adjust detection parameters in `yolo_functions.py`



## Technical Details

### Backend
- FastAPI framework
- Static file serving
- File upload handling
- Image processing pipeline

## License

This project is licensed under the MIT License - see the LICENSE file for details.
