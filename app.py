from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import cv2
import io

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from any origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model (.keras format)
model = tf.keras.models.load_model('best_model.keras')

# Image size that the model expects
IMAGE_SIZE = 224

# Load Haar cascade for eye detection (OpenCV)
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

def preprocess_image(image_data):
    """Preprocess the uploaded image to the required format."""
    # Convert the image bytes to a NumPy array
    image = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = image / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def detect_eye(image_data):
    """Check if the uploaded image contains a human eye."""
    # Convert the image bytes to a NumPy array
    image = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # Convert to grayscale for eye detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect eyes
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Return True if at least one eye is detected, else False
    return len(eyes) > 0

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """Handle image upload and make predictions."""
    try:
        # Read the image file
        image_data = await file.read()

        # Validate the image contains a human eye
        if not detect_eye(image_data):
            raise HTTPException(status_code=400, detail="No human eye detected in the image.")

        # Preprocess the image for the model
        image = preprocess_image(image_data)

        # Make prediction
        prediction = model.predict(image)[0][0]

        # Determine the class
        if prediction > 0.5:
            result = {"class": "RG", "probability": float(prediction)}
        else:
            result = {"class": "NRG", "probability": float(1 - prediction)}

        return JSONResponse(content=result)

    except HTTPException as e:
        return JSONResponse(content={"error": e.detail}, status_code=e.status_code)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Example route to test the API
@app.get("/")
def root():
    return {"message": "Welcome to the EyePACS Diabetes Prediction API"}
