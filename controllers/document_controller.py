from fastapi import HTTPException, UploadFile
import shutil
import os
import requests
import cv2
from PIL import Image
from roboflow import Roboflow

from cloudinary_config import cloudinary
from utils.core import text_extractor  
from utils.document_formatter import DocumentFormatter

def convert_to_jpg(temp_file):
    """Convert PNG to JPG if necessary."""
    img = Image.open(temp_file)
    if img.mode != "RGB":
        img = img.convert("RGB")
    jpg_path = temp_file.rsplit(".", 1)[0] + ".jpg"
    img.save(jpg_path, "JPEG")
    return jpg_path

def detect_objects(image_url: str, card_type: str):
    """Detect objects in the document using Roboflow"""
    try:
        if card_type not in ["Aadhar", "PAN", "Driving License"]:
            raise HTTPException(status_code=400, detail="Unsupported card type")

        image_path = "temp_image.jpg"
        response = requests.get(image_url, stream=True)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to download image from Cloudinary")

        with open(image_path, "wb") as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)

        if card_type == "Aadhar":
            api_key = "LFlCmvBFgaFn7jH9yIWs"
            project_name = "adhar_obj_detection"
            expected_fields = ['aadhar no', 'details', 'gov', 'logo', 'photo', 'qr']
        elif card_type == "PAN":
            api_key = "l8euJldexdnlej6ptiMb"
            project_name = "pancard-mp1jt"
            expected_fields = ['details', 'goi', 'pan', 'photo', 'qr', 'symbol']
        else:  # Driving License
            api_key = "K8xsKfZpKUg9anaKIw1I"
            project_name = "minorproject-5blne"
            expected_fields = ["Address", "DOB", "Exp date", "First name", "Issue date", "Last name", "License number", "Sex"]

        rf = Roboflow(api_key=api_key)
        project = rf.workspace().project(project_name)
        model = project.version(1).model

        predictions = model.predict(image_path, confidence=10, overlap=30).json()
        if "predictions" not in predictions:
            raise HTTPException(status_code=500, detail="Failed to process image with Roboflow")

        detected_fields = list(set(prediction["class"] for prediction in predictions["predictions"]))

        image = cv2.imread(image_path)

        # ✅ Use shared EasyOCR reader via text_extractor
        extracted_text = text_extractor.extract_document_text(image, predictions["predictions"])

        formatted_data = DocumentFormatter.format_document_data(card_type, extracted_text)

        missing_fields = [field for field in expected_fields if field not in detected_fields]

        os.remove(image_path)

        return {
            "predictions": predictions,
            "detected_fields": detected_fields,
            "missing_fields": missing_fields,
            "extracted_text": extracted_text,
            "formatted_data": formatted_data
        }

    except Exception as e:
        print(f"Error in detect_objects: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

async def process_document(card_type: str, file: UploadFile):
    """Process uploaded document"""
    print(f"\n=== Processing {card_type} Document ===")
    print(f"File name: {file.filename}")

    temp_file = f"temp_{file.filename}"
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    if file.filename.lower().endswith(".png"):
        temp_file = convert_to_jpg(temp_file)
        print("Converted PNG to JPG format")

    upload_result = cloudinary.uploader.upload(temp_file)
    cloudinary_url = upload_result.get("secure_url")

    if not cloudinary_url:
        os.remove(temp_file)
        raise HTTPException(status_code=500, detail="Failed to upload image to Cloudinary")

    os.remove(temp_file)
    print("Image uploaded to Cloudinary successfully")

    result = detect_objects(cloudinary_url, card_type)

    if result["missing_fields"]:
        print(f"\nMissing fields detected: {result['missing_fields']}")
        raise HTTPException(status_code=400, detail={
            "message": "Required objects not detected in the image.",
            "predictions": result["predictions"],
            "detected_fields": result["detected_fields"],
            "missing_fields": result["missing_fields"],
            "extracted_text": result["extracted_text"],
            "formatted_data": result["formatted_data"]
        })

    print("\n=== Document Processing Complete ===")
    print(f"Detected fields: {result['detected_fields']}")
    print("================================\n")

    return {
        "message": "Success",
        "cloudinary_url": cloudinary_url,
        "detected_fields": result["detected_fields"],
        "missing_fields": result["missing_fields"],
        "predictions": result["predictions"],
        "extracted_text": result["extracted_text"],
        "formatted_data": result["formatted_data"]
    }
