import cv2
import easyocr
from ultralytics import YOLO
import numpy as np

def load_yolo_model(model_path, imgsz=640):


    model = YOLO(model_path)
    

    model.overrides['imgsz'] = imgsz
    model.overrides['conf'] = 0.25  
    model.overrides['iou'] = 0.45   
    
    return model



def detect_objects(image_path, model_path):
    """
    Run YOLOv8 detection on an image and return the results
    
    Args:
        image_path (str): Path to the input image
        model_path (str): Path to the YOLOv8 model weights
    
    Returns:
        results: YOLOv8 detection results
        image: Original image as numpy array
    """

    model = load_yolo_model(model_path, imgsz=800)

    image = cv2.imread(image_path)

    image = cv2.resize(image, (800, 800))

    results = model(image)[0]
    
    return results, image



def draw_boxes_and_save(image, results, save_path, conf_threshold=0.5):
    """
    Draw detection boxes on image and save it
    Only draws boxes with maximum confidence for each class
    
    Args:
        image (numpy.ndarray): Original image
        results: YOLOv8 detection results
        save_path (str): Path to save the annotated image
        conf_threshold (float): Confidence threshold for detections
    
    Returns:
        dict: Dictionary containing bounding boxes for display and gauge
    """

    annotated_img = image.copy()

    best_detections = {
        'display': {'conf': 0, 'box': None},
        'gauge': {'conf': 0, 'box': None}
    }

    for box in results.boxes:
        conf = float(box.conf[0])
        class_id = int(box.cls[0])
        class_name = results.names[class_id]
        
        # Only process display and gauge classes above threshold
        if conf < conf_threshold or class_name not in ['display', 'gauge']:
            continue
            

        if conf > best_detections[class_name]['conf']:
            best_detections[class_name]['conf'] = conf
            best_detections[class_name]['box'] = box.xyxy[0].cpu().numpy()

    colors = {'display': (0, 255, 0), 'gauge': (255, 0, 0)}
    
    for class_name, detection in best_detections.items():
        if detection['box'] is not None:
            box = detection['box']
            cv2.rectangle(
                annotated_img,
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                colors[class_name],
                2
            )
            # Add label
            label = f"{class_name}: {detection['conf']:.2f}"
            cv2.putText(
                annotated_img,
                label,
                (int(box[0]), int(box[1]-10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                colors[class_name],
                2
            )
    

    cv2.imwrite(save_path, annotated_img)
    
    return best_detections

def read_display_text(image, display_box):
    """
    Read text from the display region using EasyOCR
    
    Args:
        image (numpy.ndarray): Original image
        display_box (numpy.ndarray): Bounding box coordinates [x1, y1, x2, y2]
    
    Returns:
        str: Detected text from display
    """
    if display_box is None:
        return None
        

    x1, y1, x2, y2 = map(int, display_box)
    display_region = image[y1:y2, x1:x2]
    

    reader = easyocr.Reader(['en'])
    

    results = reader.readtext(display_region)
    

    text = ' '.join([result[1] for result in results])
    
    return text

def process_digital_gauge_image(image_path, model_path, output_path):
    """
    Main function to process the gauge image
    
    Args:
        image_path (str): Path to input image
        model_path (str): Path to YOLOv8 model
        output_path (str): Path to save annotated image
    
    Returns:
        dict: Dictionary containing detection results and display text
    """

    results, image = detect_objects(image_path, model_path)

    detections = draw_boxes_and_save(image, results, output_path)

    display_text = None
    if detections['display']['box'] is not None:
        display_text = read_display_text(image, detections['display']['box'])
    
    return {
        'detections': detections,
        'display_text': display_text,
        'isGauge': detections['display']['conf'] 
    }



# Example usage
# image_path = r"C:\Users\hp\Pictures\Screenshots\Screenshot 2025-01-15 231648.png"
# model_path = r"C:\Users\hp\Downloads\digitalYOLOv8_float32.tflite"
# output_path = "results/digital_detection_result.jpg"


# results = process_digital_gauge_image(image_path, model_path, output_path)
# print("Display text:", results['display_text'], "\n")
# print("gauge threshold recieved: ", results['isGauge'])