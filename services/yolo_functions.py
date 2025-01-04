import cv2
import numpy as np
import math
import easyocr
from ultralytics import YOLO
from services.helper import get_center, get_angle
from services.text_ocr import extract_text_value


def expand_bbox(bbox, expand_factor_x=1.4, expand_factor_y=1.4, left_weight=0.7, up_weight=0.7):
    """
    Expand bounding box with bias towards left and up
    left_weight and up_weight control how much of the expansion goes to left/up vs right/bottom
    Values > 0.5 favor left/up expansion
    """
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    
  
    width_increase = width * (expand_factor_x - 1)
    height_increase = height * (expand_factor_y - 1)
    
   
    left_expand = width_increase * left_weight
    right_expand = width_increase * (1 - left_weight)
    
    
    up_expand = height_increase * up_weight
    bottom_expand = height_increase * (1 - up_weight)
    
    
    new_x1 = max(0, int(x1 - left_expand))
    new_x2 = int(x2 + right_expand)
   
    new_y1 = max(0, int(y1 - up_expand))
    new_y2 = int(y2 + bottom_expand)
    
    return [new_x1, new_y1, new_x2, new_y2]


def get_max_confidence_boxes(model_path, image_path, conf_threshold=0.2):
  
    model = YOLO(model_path)
    
 
    results = model.predict(image_path, conf=conf_threshold,imgsz=(800, 800))
    
 
    yolo_output = {}

   
    for r in results:
        for box in r.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            coordinates = box.xyxy[0].tolist()
            
          
            class_name = model.names[class_id]
            
            # Update the dictionary only if the new confidence is higher
            if class_name not in yolo_output or confidence > yolo_output[class_name]['confidence']:
                yolo_output[class_name] = {
                    'coordinates': coordinates,
                    'confidence': confidence,
                    'class_id': class_id
                }
    
   
    image = cv2.imread(image_path)
    
    # Draw only the maximum confidence boxes
    for class_name, data in yolo_output.items():
        coords = data['coordinates']
        conf = data['confidence']
        
      
        x1, y1, x2, y2 = map(int, coords)
        
 
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
     
        label = f'{class_name}: {conf:.2f}'
        cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Save the result
    cv2.imwrite('results/detection_result.jpg', image)
    
   
    formatted_output = {class_name: data['coordinates'] for class_name, data in yolo_output.items()}
    
    return formatted_output



def read_gauge(image_path, yolo_output):
    """Read gauge value from image using YOLO detected points"""
    reader = easyocr.Reader(['en'])
    img = cv2.imread(image_path)
    
 
    base_center = get_center(yolo_output['base'])
    tip_center = get_center(yolo_output['tip'])
    min_center = get_center(yolo_output['minimum'])
    max_center = get_center(yolo_output['maximum'])
    

    min_angle = get_angle(base_center, min_center)
    max_angle = get_angle(base_center, max_center)
    current_angle = get_angle(base_center, tip_center)
    
    # Debug print angles
    print(f"Raw angles - Min: {min_angle:.1f}, Max: {max_angle:.1f}, Current: {current_angle:.1f}")
    


    min_bbox = expand_bbox(yolo_output['minimum'], 2.5)  
    max_bbox = expand_bbox(yolo_output['maximum'], 2.5)    # Expand boxes by 2.5X
    
    min_region = img[int(min_bbox[1]):int(min_bbox[3]), 
                    int(min_bbox[0]):int(min_bbox[2])]
    max_region = img[int(max_bbox[1]):int(max_bbox[3]), 
                    int(max_bbox[0]):int(max_bbox[2])]
    
    min_value = extract_text_value(min_region, reader)
    max_value = extract_text_value(max_region, reader)
    
    
    min_value = min_value if min_value is not None else 0
    max_value = max_value if max_value is not None else 100

    if max_value==0:
        max_value = 100
    
    print(f"OCR values - Min: {min_value}, Max: {max_value}")
    
   
    if min_angle <= current_angle <= max_angle:
        relative_angle = 0
    elif current_angle < min_angle:
        relative_angle = min_angle - current_angle
    else:  
        relative_angle = 360 - (current_angle - min_angle)
        
    
    value_range = max_value - min_value
    angle_range = 360 - (max_angle - min_angle)
    current_value = (value_range/angle_range) * relative_angle + min_value
    
    print(f"Angle range: {angle_range:.1f}, Relative angle: {relative_angle:.1f}")
    
    # Create debug visualization
    debug_img = img.copy()
    
    # Draw points and lines
    cv2.circle(debug_img, base_center, 5, (0, 255, 0), -1)
    cv2.circle(debug_img, tip_center, 5, (0, 0, 255), -1)
    cv2.circle(debug_img, min_center, 5, (255, 0, 0), -1)
    cv2.circle(debug_img, max_center, 5, (255, 0, 0), -1)
    
    cv2.line(debug_img, base_center, tip_center, (0, 255, 0), 2)
    cv2.line(debug_img, base_center, min_center, (255, 0, 0), 2)
    cv2.line(debug_img, base_center, max_center, (255, 0, 0), 2)
    
    
    cv2.putText(debug_img, f"Min: {min_value} ({min_angle:.1f})", 
                min_center, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.putText(debug_img, f"Max: {max_value} ({max_angle:.1f})", 
                max_center, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.putText(debug_img, f"Current: {current_value:.1f} ({current_angle:.1f})", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(debug_img, f"Range: {angle_range:.1f}", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    cv2.imwrite('results/gauge_debug.jpg', debug_img)
    
    return current_value, min_value, max_value