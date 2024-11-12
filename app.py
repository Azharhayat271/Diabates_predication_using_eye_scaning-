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

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """Handle image upload and make predictions."""
    try:
        # Read the image file
        image_data = await file.read()
        image = preprocess_image(image_data)

        # Make prediction
        prediction = model.predict(image)[0][0]

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
