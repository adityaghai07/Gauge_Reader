from services.yolo_functions import get_max_confidence_boxes, read_gauge

if __name__ == "main": 
    model_path = 'bestNew.pt' 
    image_path = r"gauge-3.jpg" 
    yolo_output = get_max_confidence_boxes(model_path, image_path) 
    value, min_val, max_val = read_gauge(image_path, yolo_output) 

    print(f"Gauge reading: {value:.1f} (range: {min_val} - {max_val})")