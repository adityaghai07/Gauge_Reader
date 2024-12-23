import math

def get_center(bbox):
    """Get center point from bounding box coordinates"""
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))

def get_angle(point1, point2):
    """Calculate angle between two points relative to horizontal"""
    dx = point2[0] - point1[0]
    dy = point1[1] - point2[1]  
    angle = math.degrees(math.atan2(dy, dx))
    if angle < 0:
        angle += 360
    return angle

def normalize_angle(angle):
    """Normalize angle to be between 0 and 360"""
    return angle % 360

def extract_number(text):
    """Extract the first number from OCR text"""
    numbers = ''.join(c for c in text if c.isdigit() or c == '.')
    try:
        return float(numbers)
    except ValueError:
        return 0
    
def get_shortest_angle(angle1, angle2):
    diff = (angle2 - angle1) % 360
    if diff > 180:
        diff = diff - 360
    return diff