import cv2
from services.helper import extract_number


def preprocess_ocr_image(img):
    """Apply preprocessing to improve OCR accuracy"""
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
    
    # Increase contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    
    # Resize to larger size
    return cv2.resize(enhanced, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)


def extract_text_value(img_region, reader):
    """Extract numeric value from image region with multiple preprocessing attempts"""
    # Try different preprocessing combinations
    preprocessed = preprocess_ocr_image(img_region)
    
    attempts = [
        (img_region, "Original"),
        (preprocessed, "Preprocessed"),
        (255 - preprocessed, "Inverted"),
        (cv2.GaussianBlur(preprocessed, (3,3), 0), "Blurred")
    ]
    
    for img_attempt, name in attempts:
        # Try to read text
        results = reader.readtext(img_attempt, detail=0)
        if results:
            # Try to extract number from each result
            for text in results:
                number = extract_number(text)
                if number is not None:
                    print(f"Successfully read value {number} using {name} image")
                    return number
    
    print("Failed to read value from all attempts")
    return None