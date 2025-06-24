from roboflow import Roboflow
import cv2
import numpy as np

def load_model():
    """Initialize Roboflow and load the model"""
    rf = Roboflow(api_key="YOUR_API_KEY")
    project = rf.workspace().project("YOUR_PROJECT_NAME")
    model = project.version(1).model
    return model

def detect_objects(image_path, model):
    """Perform object detection on an image"""
    # Load image
    image = cv2.imread(image_path)
    
    # Perform prediction
    predictions = model.predict(image_path, confidence=40, overlap=30)
    
    # Process predictions
    for prediction in predictions:
        # Get bounding box coordinates
        x = prediction['x']
        y = prediction['y']
        width = prediction['width']
        height = prediction['height']
        
        # Get class and confidence
        class_name = prediction['class']
        confidence = prediction['confidence']
        
        # Draw bounding box
        cv2.rectangle(image, 
                     (int(x - width/2), int(y - height/2)),
                     (int(x + width/2), int(y + height/2)),
                     (0, 255, 0), 2)
        
        # Add label
        label = f"{class_name}: {confidence:.2f}"
        cv2.putText(image, label, 
                   (int(x - width/2), int(y - height/2 - 10)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return image

def main():
    # Initialize model
    model = load_model()
    
    # Perform detection
    image_path = "path_to_your_image.jpg"
    result_image = detect_objects(image_path, model)
    
    # Display result
    cv2.imshow("Object Detection", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 