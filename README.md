# Gauge Reader

A web-based tool for automatic gauge reading using computer vision and deep learning. This application uses YOLOv8 object detection to locate gauges in images and processes them to extract readings automatically.

![Gauge_Reader_Results](https://github.com/user-attachments/assets/5149774e-3114-46f7-a034-f555871fb8c7)


## Features

- Upload images of gauges through a web interface
- Automatic gauge detection using YOLO
- Real-time gauge reading extraction
- Visual feedback with detection and debug images
- Support for various gauge types
- Range detection and value extraction
  


## YOLOv8 Training
The YOLOv8 model was trained using the **Ultralytics** library, a cutting-edge framework for object detection and computer vision tasks. The dataset used for training was sourced from **Roboflow**, a popular platform for dataset preparation and augmentation.

---

## Dataset Description
This [dataset](https://universe.roboflow.com/aditya07/gauge-detection-cohbz-zerqg) is specifically designed for a gauge detection task in a computer vision project. It provides the following characteristics:

### Key Features
- **Annotations**:
  - Needle tip position.
  - Needle base position.
  - Minimum and maximum values of the gauge.
- **Number of Images**:
  - Approximately 5,000 high-quality images.
- **Data Augmentation**:
  - To enhance the robustness of the model, various augmentations were applied to the dataset:
    - Rotation
    - Scaling
    - Brightness and contrast adjustments
    - Flipping and other transformations.

---

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

This project is licensed under the MIT License - see the [LICENSE](https://github.com/adityaghai07/Gauge_Reader?tab=MIT-1-ov-file) file for details.
