from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
import cv2
import io

# Initialize FastAPI app
app = FastAPI()

# Load the trained model (.keras format)
model = tf.keras.models.load_model('best_model.keras')

# Load Haar Cascade for eye detection
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Image size that the model expects
IMAGE_SIZE = 224

def preprocess_image(image_data):
    """Preprocess the uploaded image to the required format."""
    # Convert the image bytes to a NumPy array
    image = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = image / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def contains_eye(image):
    """Check if the image contains an eye using Haar Cascade."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return len(eyes) > 0  # Return True if at least one eye is detected

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """Handle image upload and make predictions."""
    try:
        # Read the image file
        image_data = await file.read()
        image = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        # Validate if the uploaded image contains an eye
        if not contains_eye(image):
            return JSONResponse(content={"error": "Uploaded image must contain a human eye."}, status_code=400)

        # Preprocess the image for prediction
        processed_image = preprocess_image(image_data)

        # Make prediction
        prediction = model.predict(processed_image)[0][0]

        # Determine the class
        if prediction > 0.5:
            result = {"class": "RG", "probability": float(prediction)}
        else:
            result = {"class": "NRG", "probability": float(1 - prediction)}

        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Example route to test the API
@app.get("/")
def root():
    return {"message": "Welcome to the EyePACS Diabetes Prediction API"}
